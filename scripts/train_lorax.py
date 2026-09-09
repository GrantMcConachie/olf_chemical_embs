"""
Script to train the lora model
"""

import os

# Must be set before the first CUDA/cuBLAS call so deterministic cuBLAS GEMM
# kernels can be selected (required by torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random

import matplotlib
import numpy as np
import yaml
from tqdm import tqdm

matplotlib.use("Agg")  # non-interactive backend, safe on headless cluster nodes
import matplotlib.pyplot as plt
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel

import wandb
from data_utils.prot_smi_dataset import ProteinSmilesDataset
from loss_functions.loss_functions import M2ORWeightedCrossEntropyLoss
from model.lorax import LORAX


def set_seed(seed, deterministic=True):
    """
    Seed every RNG that influences training so runs are reproducible.

    Covers Python, NumPy and PyTorch (CPU + all CUDA devices). When
    ``deterministic`` is set we additionally pin cuDNN and request
    deterministic algorithms everywhere they exist. ``warn_only=True`` keeps
    ops that lack a deterministic kernel from raising -- they warn and fall
    back instead, so this never crashes a run. The performance cost here is
    minor: the large protein model is frozen, cached and run under no_grad,
    so only the small trainable modules are affected.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """
    Seed a DataLoader worker deterministically from the base torch seed so
    runs stay reproducible if ``num_workers`` > 0 is ever enabled.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_model(model, config):
    """
    Saves the model's state dict.

    When the protein model is frozen (``no_prot_model_ft``) its weights never
    change from the pretrained ESM2 checkpoint, so persisting them for every
    experiment/split wastes a large amount of storage (~650M params). In that
    case we drop the protein sub-model (``prot_lora_model.*``) from the saved
    state dict; it is restored from ``AutoModel.from_pretrained`` when the
    checkpoint is loaded (see the loader in ``scripts/train_GB.py``). The
    trainable heads -- including the protein-side cross-attention
    (``prot_MHA``) and layer norm (``prot_layer_norm``) -- are separate
    top-level modules and are always saved.
    """
    os.makedirs(config["training"]["save_path"], exist_ok=True)

    state_dict = model.state_dict()
    if config["model"]["combine"]["no_prot_model_ft"]:
        # frozen protein branch == pretrained ESM2, so don't persist it
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("prot_lora_model.")
        }

    torch.save(state_dict, os.path.join(config["training"]["save_path"], "model.pt"))


def evaluate(config, model, dataloader, device, loss_fn, epoch, writer, dataset):
    """
    Evaluate model against the validation set
    """
    model.eval()
    with torch.no_grad():
        running_loss = []
        preds = []
        ground_truth = []

        for i in dataloader:
            # unpack data
            smi_token, prot_token, pocket_mask, y, smiles, prot = i
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            pocket_mask = pocket_mask.to(device)
            y = y.to(device)

            # pass through model
            out = model(smi_token, prot_token, pocket_mask)
            pred = out[0].squeeze()

            # calculate loss
            if "data/M2OR" in config["training"]["data_path"]:
                loss = loss_fn(pred, y, smiles, prot)
            else:
                loss = loss_fn(pred, y)

            # bookeeping
            preds.append(pred)
            ground_truth.append(y)
            running_loss.append(loss.item())

        # metrics
        avg_loss = sum(running_loss) / len(running_loss)
        ground_truth = torch.concat(ground_truth).cpu().numpy()

        # change metrics if binary data
        if "data/M2OR" in config["training"]["data_path"]:
            logits = torch.concat(preds)
            preds = torch.nn.Sigmoid()(logits).cpu().numpy()
            bin_preds = (preds >= 0.5).astype(int)
            ave_p = average_precision_score(ground_truth, preds)
            precision = precision_score(ground_truth, bin_preds)
            recall = recall_score(ground_truth, bin_preds)
            f_score = f1_score(ground_truth, bin_preds)
            mcc = matthews_corrcoef(ground_truth, bin_preds)
            auroc = roc_auc_score(ground_truth, preds)
            print(
                f"Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} AveP: {ave_p:.4f} | {dataset} Precision: {precision:.4f} | {dataset} Recall: {recall:.4f} | {dataset} F1 score: {f_score:.4f} | {dataset} MCC: {mcc:.4f} | {dataset} AUROC: {auroc:.4f}"
            )

            # write to tensorboard
            writer.add_scalar(f"Loss/{dataset}", avg_loss, epoch)
            writer.add_scalar(f"{dataset}_metrics/AveP", ave_p, epoch)
            writer.add_scalar(f"{dataset}_metrics/Precision", precision, epoch)
            writer.add_scalar(f"{dataset}_metrics/Recall", recall, epoch)
            writer.add_scalar(f"{dataset}_metrics/F1", f_score, epoch)
            writer.add_scalar(f"{dataset}_metrics/MCC", mcc, epoch)
            writer.add_scalar(f"{dataset}_metrics/AUROC", auroc, epoch)

            # write to wandb
            wandb.log(
                {
                    f"Loss/{dataset}": avg_loss,
                    f"{dataset}_metrics/AveP": ave_p,
                    f"{dataset}_metrics/Precision": precision,
                    f"{dataset}_metrics/Recall": recall,
                    f"{dataset}_metrics/F1": f_score,
                    f"{dataset}_metrics/MCC": mcc,
                    f"{dataset}_metrics/AUROC": auroc,
                },
                step=epoch,
            )

        else:
            preds = torch.concat(preds).cpu().numpy()
            r2 = r2_score(ground_truth, preds)
            ci = concordance_index(ground_truth, preds)
            # Precision@k
            k = 10
            true_top = set(np.argsort(ground_truth)[::-1][:k])
            pred_top = set(np.argsort(preds)[::-1][:k])
            precision_k = len(true_top & pred_top) / k
            print(
                f"Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} R2: {r2:.4f} | {dataset} CI: {ci:.4f} | Precision@10: {precision_k:.1f}"
            )

            # write to tensorboard
            writer.add_scalar(f"Loss/{dataset}", avg_loss, epoch)
            writer.add_scalar(f"{dataset}_metrics/{dataset}/R2", r2, epoch)
            writer.add_scalar(f"{dataset}_metrics/{dataset}/CI", ci, epoch)

            # scatter of predictions vs. actual
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(ground_truth, preds, s=8, alpha=0.5)
            lims = [
                min(ground_truth.min(), preds.min()),
                max(ground_truth.max(), preds.max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=1)  # y = x reference
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{dataset} (R2={r2:.3f})")
            writer.add_figure(f"{dataset}_scatter/pred_vs_actual", fig, epoch)

            # write to wandb
            wandb.log(
                {
                    f"Loss/{dataset}": avg_loss,
                    f"{dataset}_metrics/{dataset}/R2": r2,
                    f"{dataset}_metrics/{dataset}/CI": ci,
                    f"{dataset}_metrics/{dataset}/Precision@{k}": precision_k,
                    f"{dataset}_scatter/pred_vs_actual": wandb.Image(fig),
                },
                step=epoch,
            )

            plt.close(fig)

    return avg_loss


def train(config):
    """
    Main training loop for LORAX
    """
    # setup device and splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility settings (defaults keep older configs working)
    split_seed = config["training"].get("seed", 42)
    deterministic = config["training"].get("deterministic", True)

    # get foudation models
    smi_model_card = config["model"]["smi_model_card"]
    prot_model_card = config["model"]["prot_model_card"]
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    # Seed per split using a stable, GPU-assignment-independent offset so a
    # given split trains identically regardless of how many GPUs are used
    # or which one it lands on. This reseeds before model init and data
    # shuffling, the two RNG-consuming steps below.
    set_seed(split_seed, deterministic=deterministic)
    data_generator = torch.Generator()
    data_generator.manual_seed(split_seed)

    # create tensorboard log
    writer = SummaryWriter(log_dir=config["training"]["save_path"])

    # create wandb run (one per split)
    wandb.init(
        project=config["training"].get("wandb_project", "lorax"),
        name=f"{config['training']['save_path'].split('/')[-1]}",
        group=f"{config['training']['save_path'].split('/')[-2]}",
        config=config,
        reinit=True,
    )

    # dataloaders
    train_data = ProteinSmilesDataset(
        os.path.join(config["training"]["data_path"], "train_df.csv"),
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card,
    )
    val_data = ProteinSmilesDataset(
        os.path.join(config["training"]["data_path"], "val_df.csv"),
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card,
    )
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config["train_lorax"]["batch_size"],
        num_workers=0,
        pin_memory=True,
        generator=data_generator,  # deterministic shuffle order
        worker_init_fn=seed_worker,
    )
    val_batch_size = config["train_lorax"].get(
        "val_batch_size", config["train_lorax"]["batch_size"]
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # init model
    model = LORAX(
        model_config=config["model"],
        smi_model=smi_model,
        prot_model=prot_model,
        no_cross_attn=config["model"]["combine"]["no_cross_attn"],
        no_prot_model_ft=config["model"]["combine"]["no_prot_model_ft"],
        lin_proj=config["model"]["combine"]["lin_proj"],
    ).to(device)

    # init optimizer and loss_fn
    optim = torch.optim.Adam(params=model.parameters(), lr=config["train_lorax"]["lr"])

    # use special loss function for M2OR data
    if "data/M2OR" in config["training"]["data_path"]:
        loss_fn = M2ORWeightedCrossEntropyLoss(config["training"]["data_path"])
    else:
        loss_fn = torch.nn.MSELoss()

    # look at parameters
    tot_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot_params = sum(p.numel() for p in model.parameters())
    print("combo model:")
    print(
        "trainable params:",
        f"{tot_train_params:,}",
        " || all params:",
        f"{tot_params:,}",
        " || trainable%:",
        f"{(tot_train_params / tot_params) * 100:0.04}",
    )

    # training loop
    best_loss = 1e10
    for epoch in range(config["train_lorax"]["train_epochs"]):
        model.train()
        running_loss = []

        for dat in tqdm(train_dataloader, desc=f"Epoch {epoch} | batch"):
            # zero gradients
            optim.zero_grad()

            # unpack data
            smi_token, prot_token, pocket_mask, y, smiles, prot = dat
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            pocket_mask = pocket_mask.to(device)
            y = y.to(device)

            # pass through model
            out = model(smi_token, prot_token, pocket_mask)
            pred = out[0].squeeze()

            # calculate loss and backprop
            if "data/M2OR" in config["training"]["data_path"]:
                loss = loss_fn(pred, y, smiles, prot)
            else:
                loss = loss_fn(pred, y)

            loss.backward()
            optim.step()

            # bookeeping
            running_loss.append(loss.item())

        # evaluate
        avg_val_loss = evaluate(
            config, model, val_dataloader, device, loss_fn, epoch, writer, "val"
        )

        # save model when the best validation loss happens
        if avg_val_loss < best_loss:
            save_model(model, config)
            writer.add_scalar("Model/best_model", epoch, epoch)

            # update best loss
            best_loss = avg_val_loss

        # report train loss
        avg_loss = sum(running_loss) / len(running_loss)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        wandb.log({"Loss/train": avg_loss}, step=epoch)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # write all pending events to log and close
    writer.flush()
    writer.close()
    wandb.finish()


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file", required=True
    )
    args = parser.parse_args()

    # load config
    config = yaml.safe_load(open(args.config, "r"))

    # call main training function
    train(config)

    # save config
    config_save_pth = os.path.join(config["training"]["save_path"], "config.yaml")
    with open(config_save_pth, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
