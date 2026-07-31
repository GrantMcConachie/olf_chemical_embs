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
        num_special_tokens=2,  # excluding cls and end tokens NOTE: could change if HF models change
    ):
        # data
        self.df = pd.read_csv(dir)
        self.smile_ids, unique_smiles = pd.factorize(self.df["SMILES"])
        self.protein_ids, unique_proteins = pd.factorize(self.df["Protein sequence"])

        # length of tokens
        self.smi_max_len = smi_model.config.max_position_embeddings - num_special_tokens
        self.prot_max_len = (
            prot_model.config.max_position_embeddings - num_special_tokens
        )
        self.smi_max_len = min(
            self.smi_max_len, max(len(s) for s in unique_smiles) + num_special_tokens
        )
        self.prot_max_len = min(
            self.prot_max_len, max(len(s) for s in unique_proteins) + num_special_tokens
        )

        # tokenizers
        self.smi_tokenizer = AutoTokenizer.from_pretrained(smi_model_card)
        self.prot_tokenizer = AutoTokenizer.from_pretrained(prot_model_card)

        self.tokenized_smiles = self.smi_tokenizer(
            list(unique_smiles),
            padding="max_length",  # NOTE: may need to specify this for particular models
            max_length=self.smi_max_len,
            truncation=True,
            return_tensors="pt",
        )
        self.tokenized_proteins = self.prot_tokenizer(
            list(unique_proteins),
            padding="max_length",
            max_length=self.prot_max_len,
            truncation=True,
            return_tensors="pt",
        )

        # build pocket weight, hard code file path for testing
        df_bp = pd.read_csv("data/CC/binding_sites/binding_sites.csv")
        self.pocket_masks = {}
        for seq, group in df_bp.groupby("sequence"):
            positions = group.loc[group["pocket"] == "pocket1", "residue"].values
            positions = torch.tensor(positions, dtype=torch.long)
            pocket_mask = torch.zeros(self.prot_max_len)
            pocket_mask[positions[positions < self.prot_max_len]] = 1.0
            self.pocket_masks[seq] = pocket_mask

        missing = set(self.df["Protein sequence"]) - set(self.pocket_masks)
        assert not missing, f"{len(missing)} proteins missing from binding_sites.csv"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        smile_id = self.smile_ids[index]
        protein_id = self.protein_ids[index]
        smi_token = {
            "input_ids": self.tokenized_smiles["input_ids"][smile_id],
            "attention_mask": self.tokenized_smiles["attention_mask"][smile_id],
        }
        prot_token = {
            "input_ids": self.tokenized_proteins["input_ids"][protein_id],
            "attention_mask": self.tokenized_proteins["attention_mask"][protein_id],
        }

        prot_seq = self.df["Protein sequence"][index]

        return (
            smi_token,
            prot_token,
            self.pocket_masks[prot_seq],
            torch.tensor(self.df["output"][index], dtype=torch.float32),
            self.df["SMILES"][index],  # smiles
            prot_seq,
        )

    def get_unique_smiles_rep(self):
        # init
        smiles = []
        smi_tokens = []

        # get unique smiles
        unique_smiles = self.df["SMILES"].unique()

        # generate tokens
        for smi in unique_smiles:
            smi_token = self.smi_tokenizer(
                smi,
                padding="max_length",  # may need to specify this for particular models
                max_length=self.smi_max_len,
                truncation=True,
                return_tensors="pt",
            )
            smiles.append(smi)
            smi_tokens.append(smi_token)

        return smiles, smi_tokens

    def get_unique_prot_rep(self):
        # init
        prots = []
        prot_tokens = []

        # get unique smiles
        unique_prots = self.df["Protein sequence"].unique()

        # generate tokens
        for prot in unique_prots:
            prot_token = self.prot_tokenizer(
                prot,
                padding="max_length",  # may need to specify this for particular models
                max_length=self.prot_max_len,
                truncation=True,
                return_tensors="pt",
            )
            prots.append(prot)
            prot_tokens.append(prot_token)

        return prots, prot_tokens
