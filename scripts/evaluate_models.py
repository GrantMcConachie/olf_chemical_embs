import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from data_utils.prot_smi_dataset import ProteinSmilesDataset
from model.lorax import LORAX

DEVICE = "cuda"

MODEL_DIR = "results/LORO_sweep"
model_dirs = sorted(
    d
    for d in os.listdir(MODEL_DIR)
    if (
        os.path.isdir(os.path.join(MODEL_DIR, d))
        and not os.path.exists(f"{MODEL_DIR}/{d}/ChemBERTa-77M-MTR/val_pred.csv")
    )
)
print("Model directories:", model_dirs)
config_file = "configs/config_default.yaml"
config = yaml.safe_load(open(config_file, "r"))

smi_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(DEVICE).eval()
prot_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(DEVICE).eval()

model = (
    LORAX(
        model_config=config["model"],
        smi_model=smi_model,
        prot_model=prot_model,
        no_cross_attn=config["model"]["combine"]["no_cross_attn"],
        no_prot_model_ft=config["model"]["combine"]["no_prot_model_ft"],
        lin_proj=config["model"]["combine"]["lin_proj"],
    )
    .to(DEVICE)
    .eval()
)

for model_dir in tqdm(model_dirs):
    model_path = f"{MODEL_DIR}/{model_dir}/ChemBERTa-77M-MTR/{model_dir}/ChemBERTa-77M-MTR_esm2_t33_650M_UR50D_{model_dir}.pt"
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    data_path = f"data/CC/LORO/{model_dir}/val_df.csv"
    val_dataset = ProteinSmilesDataset(
        data_path,
        smi_model,
        prot_model,
        config["model"]["smi_model_card"],
        config["model"]["prot_model_card"],
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    preds = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            smi_token, prot_token, y, smiles, prot = batch
            smi_token = {k: v.to(DEVICE) for k, v in smi_token.items()}
            prot_token = {k: v.to(DEVICE) for k, v in prot_token.items()}

            out = model(smi_token, prot_token)
            preds.append(out[0].squeeze(-1).to("cpu"))
    # Add predictions to validation dataset
    preds = torch.cat(preds)
    val_csv = pd.read_csv(data_path)
    assert len(preds) == len(val_csv)
    val_csv["prediction"] = preds.numpy()
    val_csv.to_csv(
        f"{MODEL_DIR}/{model_dir}/ChemBERTa-77M-MTR/val_pred.csv", index=False
    )
