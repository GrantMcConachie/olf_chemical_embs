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

        # create combination MLP
        self.mlp = nn.Sequential(
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
        # pass foundation model tokens through lora models
        smi_rep = self.smi_lora_model(**smi_token)
        prot_rep = self.prot_lora_model(**prot_token)

        # concatenate representations
        cat_rep = torch.cat((smi_rep.pooler_output, prot_rep.pooler_output), -1)

        # passing through combination mlp
        output = self.mlp(cat_rep)

        return output

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
