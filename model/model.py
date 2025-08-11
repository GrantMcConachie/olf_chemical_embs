"""
Multimodel LoRA transformer model
"""

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model


class ProteinSmilesLoraModel(nn.Module):
    def __init__(
            self,
            model_config,
            smi_model,
            prot_model,
    ):
        super(ProteinSmilesLoraModel, self).__init__()
        self.model_config = model_config

        # create loara models
        lora_config = self.create_lora_config(model_config)
        self.smi_lora_model = get_peft_model(smi_model, lora_config) 
        self.prot_lora_model = get_peft_model(prot_model, lora_config)

        print('smiles foudation model:')
        self.smi_lora_model.print_trainable_parameters()
        print('protien foundation model:')
        self.prot_lora_model.print_trainable_parameters()

        # linear projection of molecule embeddings into the same space to be
        # compared with each other
        self.lin_proj = nn.Linear(
            self.smi_lora_model.config.hidden_size,
            model_config['combine']['smiles_hidden_dim']
        )

        # multiheaded cross attention blocks
        self.smi_MHA = nn.MultiheadAttention(
            model_config['combine']['smiles_hidden_dim'],
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
            kdim=model_config['combine']['smiles_hidden_dim'],
            vdim=model_config['combine']['smiles_hidden_dim'],
            batch_first=True
        )

        # layer norms
        self.smi_layer_norm = nn.LayerNorm(model_config['combine']['smiles_hidden_dim'])
        self.prot_layer_norm = nn.LayerNorm(self.prot_lora_model.config.hidden_size)

        # create combination MLP
        self.mlp = nn.Sequential(
            nn.Linear(
                model_config['combine']['smiles_hidden_dim'] + self.prot_lora_model.config.hidden_size,
                model_config['combine']['mlp_hidden_dim']
            ),
            nn.ReLU(),
            nn.Linear(model_config['combine']['mlp_hidden_dim'], model_config['combine']['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Linear(model_config['combine']['mlp_hidden_dim'], 1)
        )

    def forward(self, smi_token, prot_token):
        # unpack attention masks
        smi_mask = smi_token['attention_mask']
        prot_mask = prot_token['attention_mask']

        # pass foundation model tokens through lora models
        smi_rep = self.smi_lora_model(**smi_token)
        prot_rep = self.prot_lora_model(**prot_token).last_hidden_state

        # projecting smiles representation to consistent dimension
        if self.model_config['combine']['full_smiles_sequence']:
            smi_rep_new = self.lin_proj(smi_rep.last_hidden_state)
        else:
            smi_rep_new = self.lin_proj(smi_rep.pooler_output.unsqueeze(-2))

        # passign representations through cross attention
        smi_attn, _ = self.smi_MHA(
            query=smi_rep_new,
            key=prot_rep,
            value=prot_rep,
            key_padding_mask=(prot_mask == 0)
        )
        prot_attn, _ = self.prot_MHA(
            query=prot_rep,
            key=smi_rep_new,
            value=smi_rep_new,
            key_padding_mask=(smi_mask == 0)
        )

        # residual connection + layer norm
        smi_rep = smi_attn + smi_rep_new
        prot_rep += prot_attn
        smi_rep = self.smi_layer_norm(smi_rep)
        prot_rep = self.prot_layer_norm(prot_rep)

        # mean pool and predict
        smi_mask = smi_mask.float()
        prot_mask = prot_mask.float()
        smi_rep = (smi_rep * smi_mask.unsqueeze(-1)).sum(dim=1) / (smi_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
        prot_rep = (prot_rep * prot_mask.unsqueeze(-1)).sum(dim=1) / (prot_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
        cat_rep = torch.cat((smi_rep, prot_rep), -1)

        # passing through combination mlp
        output = self.mlp(cat_rep)

        return output, smi_rep_new, smi_token['attention_mask']

    def create_lora_config(self, model_config):

        lora_config = LoraConfig(
            inference_mode=model_config['lora_module']['inference_mode'],
            r=model_config['lora_module']['r'],  # rank of the matrices
            lora_alpha=model_config['lora_module']['lora_alpha'],  # scaling weights by alpha/r 
            bias=model_config['lora_module']['bias'],
            use_rslora=model_config['lora_module']['use_rslora'],
            modules_to_save=model_config['lora_module']['modules_to_save'],
            target_modules=model_config['lora_module']['target_modules'],
            lora_dropout=model_config['lora_module']['lora_dropout']
        )

        return lora_config
