"""
Implementing special loss functions for M2OR dataset
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class M2ORWeightedCrossEntropyLoss(nn.Module):
    """
    Weighted loss function used in Hladiš et. al. to account for data quality
    for the M2OR dataset
    """
    def __init__(self, data_path):
        super(M2ORWeightedCrossEntropyLoss, self).__init__()

        # loading full data
        full_dat = os.path.join(
            os.path.dirname(data_path),
            'raw',
            'full_data.csv'
        )
        full_dat_df = pd.read_csv(full_dat)

        ## Data quality weights
        primary_pos_weight = 0.40
        primary_neg_weight = 0.69
        secondary_pos_weight = 0.72
        secondary_neg_weight = 0.77
        ec50_weight = 1.0

        # create list of conditions
        conditions = [
            full_dat_df['_DataQuality'] == 'ec50',
            (full_dat_df['_DataQuality'] == 'primaryScreening') & (full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'primaryScreening') & (~full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'secondaryScreening') & (full_dat_df['Responsive']),
            (full_dat_df['_DataQuality'] == 'secondaryScreening') & (~full_dat_df['Responsive']),
        ]

        # list of correcponding weights
        choices = [
            ec50_weight,
            primary_pos_weight,
            primary_neg_weight,
            secondary_pos_weight,
            secondary_neg_weight
        ]

        # create column of weights
        full_dat_df['data_quality_w'] = np.select(conditions, choices, default=np.nan)

        # Check for Incorrect data quality label
        if full_dat_df['data_quality_w'].isnull().any():
            raise Exception("Incorrect data quality label")

        # build weight dict
        self.data_quality_w = dict(
            zip(
                zip(full_dat_df['SMILES'], full_dat_df['Protein sequence']),
                full_dat_df['data_quality_w']
            )
        )

        ## calculating class imbalance weights
        num_responsive = (full_dat_df['Responsive'] == 1).sum()
        num_non_responsive = (full_dat_df['Responsive'] == 0).sum()
        self.pos_class_weight = num_non_responsive / num_responsive
        self.neg_class_weight = 1.0

        ## pair imbalance
        self.num_mol_per_prot = full_dat_df['Protein sequence'].value_counts().to_dict()
        self.num_prot_per_mol = full_dat_df['SMILES'].value_counts().to_dict()
        self.K = 100.0
    
    def pair_imbalance_weight(self, mol, prot):
        """
        Eq 5 in supplement C of Hladiš et. al.
        """
        w_pair = np.log(1 + self.K / 2 * (1/self.num_mol_per_prot[prot] + 1/self.num_prot_per_mol[mol]))
        return w_pair

    def forward(self, pred, target, smiles, proteins):
        batch_size = pred.shape[0]

        # Calculate a weight for each sample in the batch based on the SMILES and protein
        weights = []
        for i in range(batch_size):
            smi = smiles[i]
            prot = proteins[i]

            # get individual weights
            w_quality = self.data_quality_w[(smiles[i], proteins[i])]
            w_class = self.pos_class_weight if target[i] else self.neg_class_weight
            w_pair = self.pair_imbalance_weight(smi, prot)

            # get total weighting append to weights
            w_tot = w_quality * w_class * w_pair
            weights.append(w_tot)

        # Convert to tensor, same device/type as pred
        weights = torch.tensor(weights, dtype=pred.dtype, device=pred.device)

        # Compute BCE loss per sample (no reduction)
        bce = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
        loss = bce(pred, target)

        return loss
