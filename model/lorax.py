"""
Multimodel LoRA transformer model

TODO: Speed up MHA layers (https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model


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
            lin_proj=False
    ):
        super(LORAX, self).__init__()
        self.model_config = model_config
        self.no_cross_attn = no_cross_attn

        # create lora models
        lora_config_smi = self.create_lora_config(model_config, model_type='smi')
        lora_config_prot = self.create_lora_config(model_config, model_type='prot')
        self.smi_lora_model = get_peft_model(smi_model, lora_config_smi) 
        self.prot_lora_model = get_peft_model(prot_model, lora_config_prot)

        # making a hidden size attribute for unimol
        if model_config['smi_model_card'] == 'unimol':
            self.smi_lora_model.config = copy.deepcopy(self.prot_lora_model.config)
            self.smi_lora_model.config.hidden_size = smi_model.args.encoder_embed_dim

        print('smiles foudation model:')
        self.smi_lora_model.print_trainable_parameters()

        if no_prot_model_ft:
            for param in self.prot_lora_model.parameters():
                param.requires_grad = False
            print("No protein fine-tuning")

        print('protein foundation model:')
        self.prot_lora_model.print_trainable_parameters()

        # multiheaded cross attention blocks
        self.smi_MHA = nn.MultiheadAttention(
            self.smi_lora_model.config.hidden_size,
            model_config['combine']['num_heads'],
            dropout=model_config['combine']['comb_dropout'],
            kdim=self.prot_lora_model.config.hidden_size,
            vdim=self.prot_lora_model.config.hidden_size,
            batch_first=True
        )
        self.prot_MHA = nn.MultiheadAttention(
            self.prot_lora_model.config.hidden_size,
            model_config['combine']['num_heads'],
            dropout=model_config['combine']['comb_dropout'],
            kdim=self.smi_lora_model.config.hidden_size,
            vdim=self.smi_lora_model.config.hidden_size,
            batch_first=True
        )

        # layer norms
        self.smi_layer_norm = nn.LayerNorm(self.smi_lora_model.config.hidden_size)
        self.prot_layer_norm = nn.LayerNorm(self.prot_lora_model.config.hidden_size)

        if lin_proj:
            # linear projection
            self.proj = nn.Linear(
                self.smi_lora_model.config.hidden_size + self.prot_lora_model.config.hidden_size,
                1
            )

        else:
            # create combination MLP
            self.proj = nn.Sequential(
                nn.Linear(
                    self.smi_lora_model.config.hidden_size + self.prot_lora_model.config.hidden_size,
                    model_config['combine']['mlp_hidden_dim']
                ),
                nn.ReLU(),
                nn.Linear(model_config['combine']['mlp_hidden_dim'], model_config['combine']['mlp_hidden_dim']),
                nn.ReLU(),
                nn.Linear(model_config['combine']['mlp_hidden_dim'], 1)
            )

    def forward(self, smi_token, prot_token):
        # unpack attention masks
        if self.model_config['smi_model_card'] == 'unimol':
            pad_len = 512
            smi_mask = smi_token['atom_mask']
            smi_mask = F.pad(smi_mask, (0, pad_len - smi_mask.shape[1]), 'constant', 0)  # pad sequence length
        else:
            smi_mask = smi_token['attention_mask']
        prot_mask = prot_token['attention_mask']

        # pass foundation model tokens through lora models
        if self.model_config['smi_model_card'] == 'unimol':
            smi_rep_lora = self.smi_lora_model(**smi_token, return_repr=True, return_atomic_reprs=True)
            smi_rep_lora = [F.pad(i, (0, 0, 0, pad_len - i.shape[0]), 'constant', 0) for i in smi_rep_lora['atomic_reprs']]
            smi_rep_lora = torch.stack(smi_rep_lora)
        else:
            smi_rep_lora = self.smi_lora_model(**smi_token)
        prot_rep_lora = self.prot_lora_model(**prot_token).last_hidden_state

        # projecting smiles representation to consistent dimension
        if self.model_config['combine']['full_smiles_sequence']:
            if self.model_config['smi_model_card'] == 'unimol':
                smi_rep_new = smi_rep_lora
            else:
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
                key_padding_mask=(prot_mask == 0)
            )
            prot_attn, _ = self.prot_MHA(
                query=prot_rep_lora,
                key=smi_rep_new,
                value=smi_rep_new,
                key_padding_mask=(smi_mask == 0)
            )

            # residual connection + layer norm
            smi_rep = smi_attn + smi_rep_new
            prot_rep = prot_attn + prot_rep_lora
            smi_rep = self.smi_layer_norm(smi_rep)
            prot_rep = self.prot_layer_norm(prot_rep)

        # mean pool
        smi_mask = smi_mask.float()
        prot_mask = prot_mask.float()
        smi_rep = (smi_rep * smi_mask.unsqueeze(-1)).sum(dim=1) / (smi_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
        prot_rep = (prot_rep * prot_mask.unsqueeze(-1)).sum(dim=1) / (prot_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
        
        # concatenate
        cat_rep = torch.cat((smi_rep, prot_rep), -1)

        # passing through combination mlp
        output = self.proj(cat_rep)

        return (
            output,
            smi_rep_new,
            smi_mask,
            cat_rep,
            prot_rep_lora,
            prot_mask
        )

    def create_lora_config(self, model_config, model_type):

        lora_config = LoraConfig(
            inference_mode=model_config['lora_module']['inference_mode'],
            r=model_config['lora_module']['r'],  # rank of the matrices
            lora_alpha=model_config['lora_module']['lora_alpha'],  # scaling weights by alpha/r 
            bias=model_config['lora_module']['bias'],
            use_rslora=model_config['lora_module']['use_rslora'],
            modules_to_save=model_config['lora_module']['modules_to_save'],
            target_modules=model_config['lora_module'][f'target_{model_type}_modules'],
            lora_dropout=model_config['lora_module']['lora_dropout']
        )

        return lora_config
    
    def get_cross_attn_weights(self, smi_token, prot_token, average_attn_weights):
        """
        Method for getting cross attn weights for visualization.
        """
        smi_mask = smi_token['attention_mask']
        prot_mask = prot_token['attention_mask']

        # pass foundation model tokens through lora models
        smi_rep_lora = self.smi_lora_model(**smi_token)
        prot_rep_lora = self.prot_lora_model(**prot_token).last_hidden_state

        # projecting smiles representation to consistent dimension
        if self.model_config['combine']['full_smiles_sequence']:
            smi_rep_new = smi_rep_lora.last_hidden_state
        else:
            smi_rep_new = smi_rep_lora.pooler_output.unsqueeze(-2)

        if self.no_cross_attn:
            raise Exception(
                "Cannot get cross attn weights without cross attn layers. " \
                "Set 'no_cross_attn' to 'False'."
            )

        else:
            # passing representations through cross attention
            smi_attn, smi_attn_weights = self.smi_MHA(
                query=smi_rep_new,
                key=prot_rep_lora,
                value=prot_rep_lora,
                key_padding_mask=(prot_mask == 0),
                average_attn_weights=average_attn_weights
            )
            prot_attn, prot_attn_weights = self.prot_MHA(
                query=prot_rep_lora,
                key=smi_rep_new,
                value=smi_rep_new,
                key_padding_mask=(smi_mask == 0),
                average_attn_weights=average_attn_weights
            )
        
        return smi_attn_weights, prot_attn_weights
