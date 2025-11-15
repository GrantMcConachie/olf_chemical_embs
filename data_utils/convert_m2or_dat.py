"""
This is a script that takes the existing rand split 5 fold M2OR data from
"MATCHING RECEPTOR TO ODORANT WITH PROTEIN
LANGUAGE AND GRAPH NEURAL NETWORKS" (Hladiš et. al.), and converts it to
a LORAX useable format.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from rdkit import Chem


def generate_dicts(m2or_raw_fp):
    """
    Generates dictionaries from mol_id and seq_id
    """
    mol_fp = os.path.join(m2or_raw_fp, 'mols.csv')
    mol_df = pd.read_csv(mol_fp, delimiter=";")
    mol_dict = {}

    # loop through all the mols
    for idx, row in tqdm(mol_df.iterrows(), desc='getting smiles strings'):
        if pd.isna(row['canonicalSMILES']):
            pubchem_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{row['InChI Key']}/property/SMILES/TXT'  # NOTE: not SMILES not CanonicalSMILES
            smi = requests.get(pubchem_url).text.split('\n')[0]
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            time.sleep(0.2)  # so api doesn't get mad

        else:
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(row['canonicalSMILES']))
        
        # TODO: Add something for sum of isomers

        # append to dictionary
        mol_dict[row['mol_id']] = canon_smi

    # loop through all the proteins
    prot_fp = os.path.join(m2or_raw_fp, 'seqs.csv')
    prot_df = pd.read_csv(prot_fp, delimiter=";")
    prot_dict = {}

    # loop through all the mols
    for idx, row in prot_df.iterrows():
        # append to dictionary
        prot_dict[row['seq_id']] = {
            'seq': row['mutated_Sequence'],
            'Uniprot ID': row['Uniprot ID']
        }

    return mol_dict, prot_dict


def main(m2or_split_fp, m2or_raw_fp):
    # create dicts for mols and protein sequences
    smi_dict, prot_dict = generate_dicts(
        m2or_raw_fp
    )
    print('here')

    # # loop through all csv files and append necessary info
    # for split in os.listdir(m2or_split_fp):
    #     split_dir = os.path.join(m2or_split_fp, split)
    #     for file in os.listdir(split_dir):
    #         fp = os.path.join(split_dir, file)
    #         df = pd.read_csv(fp, delimiter=';')

    #         # append seqence and smiles to the csv
    #         smiles = []
    #         prot_seq = []
    #         uniprot_id = []
    #         for i, row in df.iterrows():
    #             smiles.append(smi_dict[row['mol_id']])
    #             prot_seq.append(prot_dict[row['seq_id']]['seq'])
    #             uniprot_id.append(prot_dict[row['seq_id']]['Uniprot ID'])
            
    #         # append to df and save
    #         df['SMILES'] = smiles
    #         df['Protein sequence'] = prot_seq
    #         df['Uniprot id'] = uniprot_id

    #         df.to_csv(fp, index=False)


if __name__ == '__main__':
    m2or_split_fp = './data/M2OR/rand_splits'
    m2or_raw_fp = './data/M2OR/raw'
    main(m2or_split_fp, m2or_raw_fp)
