"""
Dataset utilities for the model.
"""

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ProteinSmilesDataset(Dataset):
    def __init__(
            self,
            dir,
            smi_model,
            prot_model,
            smi_model_card,
            prot_model_card,
            num_special_tokens=2  # excluding cls and end tokens NOTE: check on this for switching out models
    ):
        # data
        self.df = pd.read_csv(dir)

        # length of tokens 
        self.smi_max_len = smi_model.config.max_position_embeddings-num_special_tokens
        self.prot_max_len = prot_model.config.max_position_embeddings-num_special_tokens

        # tokenizers
        self.smi_tokenizer = AutoTokenizer.from_pretrained(smi_model_card)
        self.prot_tokenizer = AutoTokenizer.from_pretrained(prot_model_card)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        smi_token = self.smi_tokenizer(
            self.df['SMILES'][index],
            padding='max_length',  # may need to specify this for particular models
            max_length=self.smi_max_len,
            truncation=True,
            return_tensors='pt'
        )
        prot_token = self.prot_tokenizer(
            self.df['Protein sequence'][index],
            padding='max_length',
            max_length=self.prot_max_len,
            truncation=True,
            return_tensors='pt'
        )
        output = self.df['output'][index]

        # squeeze dim to batchsize x embedding length
        smi_token['input_ids'] = smi_token['input_ids'].squeeze()
        smi_token['attention_mask'] = smi_token['attention_mask'].squeeze()
        prot_token['input_ids'] = prot_token['input_ids'].squeeze()
        prot_token['attention_mask'] = prot_token['attention_mask'].squeeze()

        return (
            smi_token,
            prot_token,
            torch.tensor(output, dtype=torch.float32)
        )

    def get_unique_smiles_rep(self):
        # init
        smiles = []
        smi_tokens = []

        # get unique smiles
        unique_smiles = self.df['SMILES'].unique()

        # generate tokens
        for smi in unique_smiles:
            smi_token = self.smi_tokenizer(
                smi,
                padding='max_length',  # may need to specify this for particular models
                max_length=self.smi_max_len,
                truncation=True,
                return_tensors='pt'
            )
            smiles.append(smi)
            smi_tokens.append(smi_token)

        return smiles, smi_tokens
