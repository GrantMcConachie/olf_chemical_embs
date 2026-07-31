"""
Multimodel LoRA transformer model

TODO: Speed up MHA layers (https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
"""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn


class LORAX(nn.Module):
    """
    LORAX model
    """

    def __init__(
        self,
        model_config,
        smi_model,
        prot_model,
        no_cross_attn=False,
        no_prot_model_ft=False,
        lin_proj=False,
    ):
        super().__init__()
        self.model_config = model_config
        self.no_cross_attn = no_cross_attn

        # Cache frozen protein embeddings. When the protein model is not
        # fine-tuned its output is a deterministic function of the sequence,
        # so we memoize last_hidden_state by sequence string and skip the
        # (expensive) ESM2 forward on repeat proteins.
        self.cache_prot = no_prot_model_ft
        self.prot_cache = {}

        # create loara models
        lora_config_smi = self.create_lora_config(model_config)
        lora_config_prot = self.create_lora_config(model_config)
        self.smi_lora_model = get_peft_model(smi_model, lora_config_smi)
        self.prot_lora_model = get_peft_model(prot_model, lora_config_prot)

        print("smiles foudation model:")
        self.smi_lora_model.print_trainable_parameters()

        if no_prot_model_ft:
            for param in self.prot_lora_model.parameters():
                param.requires_grad = False
            print("No protein fine-tuning")

        print("protein foundation model:")
        self.prot_lora_model.print_trainable_parameters()

        # multiheaded cross attention blocks
        self.smi_MHA = nn.MultiheadAttention(
            self.smi_lora_model.config.hidden_size,
            model_config["combine"]["num_heads"],
            dropout=model_config["combine"]["comb_dropout"],
            kdim=self.prot_lora_model.config.hidden_size,
            vdim=self.prot_lora_model.config.hidden_size,
            batch_first=True,
        )
        self.prot_MHA = nn.MultiheadAttention(
            self.prot_lora_model.config.hidden_size,
            model_config["combine"]["num_heads"],
            dropout=model_config["combine"]["comb_dropout"],
            kdim=self.smi_lora_model.config.hidden_size,
            vdim=self.smi_lora_model.config.hidden_size,
            batch_first=True,
        )

        # layer norms
        self.smi_layer_norm = nn.LayerNorm(self.smi_lora_model.config.hidden_size)
        self.prot_layer_norm = nn.LayerNorm(self.prot_lora_model.config.hidden_size)

        if lin_proj:
            # linear projection
            self.proj = nn.Linear(
                self.smi_lora_model.config.hidden_size
                + self.prot_lora_model.config.hidden_size,
                1,
            )

        else:
            # create combination MLP
            self.proj = nn.Sequential(
                nn.Linear(
                    self.smi_lora_model.config.hidden_size
                    + self.prot_lora_model.config.hidden_size,
                    model_config["combine"]["mlp_hidden_dim"],
                ),
                nn.ReLU(),
                nn.Linear(
                    model_config["combine"]["mlp_hidden_dim"],
                    model_config["combine"]["mlp_hidden_dim"],
                ),
                nn.ReLU(),
                nn.Linear(model_config["combine"]["mlp_hidden_dim"], 1),
            )

    def forward(self, smi_token, prot_token, pocket_mask=None):
        # unpack attention masks
        smi_mask = smi_token["attention_mask"]
        prot_mask = prot_token["attention_mask"]

        # pass foundation model tokens through lora models
        smi_rep_lora = self.smi_lora_model(**smi_token)
        prot_rep_lora = self._protein_embeddings(prot_token)

        # projecting smiles representation to consistent dimension
        if self.model_config["combine"]["full_smiles_sequence"]:
            smi_rep_new = smi_rep_lora.last_hidden_state
        else:
            smi_rep_new = smi_rep_lora.pooler_output.unsqueeze(-2)

        if self.no_cross_attn:
            smi_rep = smi_rep_new.clone()
            prot_rep = prot_rep_lora.clone()

        else:
            # passing representations through cross attention
            smi_attn, _ = self.smi_MHA(
                query=smi_rep_new,
                key=prot_rep_lora,
                value=prot_rep_lora,
                key_padding_mask=(prot_mask == 0),
            )
            prot_attn, _ = self.prot_MHA(
                query=prot_rep_lora,
                key=smi_rep_new,
                value=smi_rep_new,
                key_padding_mask=(smi_mask == 0),
            )

            # residual connection + layer norm
            smi_rep = smi_attn + smi_rep_new
            prot_rep = prot_attn + prot_rep_lora
            smi_rep = self.smi_layer_norm(smi_rep)
            prot_rep = self.prot_layer_norm(prot_rep)

        # mean pool
        smi_mask = smi_mask.float()
        prot_mask = prot_mask.float()
        if pocket_mask is None:
            prot_full_mask = prot_mask
        else:
            prot_full_mask = prot_mask * pocket_mask
        smi_rep = (smi_rep * smi_mask.unsqueeze(-1)).sum(dim=1) / (
            smi_mask.unsqueeze(-1).sum(dim=1) + 1e-8
        )
        prot_rep = (prot_rep * prot_full_mask.unsqueeze(-1)).sum(dim=1) / (
            prot_full_mask.unsqueeze(-1).sum(dim=1) + 1e-8
        )

        # concatenate
        cat_rep = torch.cat((smi_rep, prot_rep), -1)

        # passing through combination mlp
        output = self.proj(cat_rep)

        return (output, smi_rep_new, smi_mask, cat_rep, prot_rep_lora, prot_mask)

    def _protein_embeddings(self, prot_token):
        """
        Returns the protein model's last_hidden_state for a batch.

        When the protein model is frozen (`no_prot_model_ft`) its output is a
        deterministic function of the input tokens, so we memoize it by the
        token ids and only run the (expensive) ESM2 forward on proteins we
        have not seen yet. When the protein model is fine-tuned its output
        changes every step, so we always run it live.
        """
        if not self.cache_prot:
            return self.prot_lora_model(**prot_token).last_hidden_state

        # raw token bytes form a hashable, sequence-unique cache key per row
        keys = [ids.tobytes() for ids in prot_token["input_ids"].cpu().numpy()]
        miss = [i for i, k in enumerate(keys) if k not in self.prot_cache]

        if miss:
            sub = {k: v[miss] for k, v in prot_token.items()}
            was_training = self.prot_lora_model.training
            self.prot_lora_model.eval()  # canonical frozen features (no dropout)
            with torch.no_grad():
                emb = self.prot_lora_model(**sub).last_hidden_state
            if was_training:
                self.prot_lora_model.train()
            for j, i in enumerate(miss):
                self.prot_cache[keys[i]] = emb[j]

        return torch.stack([self.prot_cache[k] for k in keys])

    def create_lora_config(self, model_config):

        lora_config = LoraConfig(
            inference_mode=model_config["lora_module"]["inference_mode"],
            r=model_config["lora_module"]["r"],  # rank of the matrices
            lora_alpha=model_config["lora_module"][
                "lora_alpha"
            ],  # scaling weights by alpha/r
            bias=model_config["lora_module"]["bias"],
            use_rslora=model_config["lora_module"]["use_rslora"],
            modules_to_save=model_config["lora_module"]["modules_to_save"],
            target_modules=model_config["lora_module"]["target_modules"],
            lora_dropout=model_config["lora_module"]["lora_dropout"],
        )

        return lora_config

    def get_cross_attn_weights(self, smi_token, prot_token, average_attn_weights):
        """
        Method for getting cross attn weights for visualization.
        """
        smi_mask = smi_token["attention_mask"]
        prot_mask = prot_token["attention_mask"]

        # pass foundation model tokens through lora models
        smi_rep_lora = self.smi_lora_model(**smi_token)
        prot_rep_lora = self._protein_embeddings(prot_token)

        # projecting smiles representation to consistent dimension
        if self.model_config["combine"]["full_smiles_sequence"]:
            smi_rep_new = smi_rep_lora.last_hidden_state
        else:
            smi_rep_new = smi_rep_lora.pooler_output.unsqueeze(-2)

        if self.no_cross_attn:
            raise Exception(
                "Cannot get cross attn weights without cross attn layers. "
                "Set 'no_cross_attn' to 'False'."
            )

        else:
            # passing representations through cross attention
            smi_attn, smi_attn_weights = self.smi_MHA(
                query=smi_rep_new,
                key=prot_rep_lora,
                value=prot_rep_lora,
                key_padding_mask=(prot_mask == 0),
                average_attn_weights=average_attn_weights,
            )
            prot_attn, prot_attn_weights = self.prot_MHA(
                query=prot_rep_lora,
                key=smi_rep_new,
                value=smi_rep_new,
                key_padding_mask=(smi_mask == 0),
                average_attn_weights=average_attn_weights,
            )

        return smi_attn_weights, prot_attn_weights
