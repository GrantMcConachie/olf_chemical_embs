"""
Script to train the lora model
"""

import os
# Must be set before the first CUDA/cuBLAS call so deterministic cuBLAS GEMM
# kernels can be selected (required by torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml
import random
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe on headless cluster nodes
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from lifelines.utils import concordance_index

import torch
import torch.multiprocessing as mp
from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from model.lorax import LORAX
from data_utils.prot_smi_dataset import ProteinSmilesDataset
from loss_functions.loss_functions import M2ORWeightedCrossEntropyLoss


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


def save_molecular_rep(train_data, val_data, test_data, model, config, device, designation):
    """
    Saves molecular representations over training
    """
    # init arrays
    smiles = []
    smi_reps = []
    rep_attn_masks = []

    # getting smiles tokens
    smi_train, smi_reps_train = train_data.get_unique_smiles_rep()
    smi_val, smi_reps_val = val_data.get_unique_smiles_rep()
    smi_test, smi_reps_test = test_data.get_unique_smiles_rep()

    # combine smiles tokens
    smi_tokens = {}
    all_smi = smi_train + smi_val + smi_test
    all_reps = smi_reps_train + smi_reps_val + smi_reps_test
    for smi, rep in zip(all_smi, all_reps):
        if smi not in smi_tokens:
            smi_tokens[smi] = rep

    # get a random protein token
    _, prot_token, _, _, _ = next(iter(train_data))
    prot_token['input_ids'] = prot_token['input_ids'].unsqueeze(0)
    prot_token['attention_mask'] = prot_token['attention_mask'].unsqueeze(0)
    prot_token = {k: v.to(device) for k, v in prot_token.items()}

    # get model representation of all smiles
    model.eval()
    with torch.no_grad():
        for key, value in smi_tokens.items():
            smi_token = value
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            out = model(smi_token, prot_token)
            smi_rep = out[1]
            rep_attn_mask = out[2]

            # append arrays
            smiles.append(key)
            smi_reps.append(smi_rep.cpu())
            rep_attn_masks.append(rep_attn_mask.cpu())

    # save representations
    save_path = os.path.join(
        config['training']['results_path'],
        config['model']['smi_model_card'].split('/')[-1],
        'saved_representations'
    )
    os.makedirs(save_path, exist_ok=True)

    pkl.dump(smiles, open(os.path.join(save_path, f'smiles_{designation}.pkl'), 'wb'))
    pkl.dump(smi_reps, open(os.path.join(save_path, f'smi_reps_{designation}.pkl'), 'wb'))
    pkl.dump(rep_attn_masks, open(os.path.join(save_path, f'rep_attn_masks_{designation}.pkl'), 'wb'))


def save_model(model, config, split):
    """
    Saves model
    """
    save_path = os.path.join(
        config['training']['results_path'],
        config['model']['smi_model_card'].split('/')[-1],
        split
    )
    os.makedirs(save_path, exist_ok=True)
    model_fp = f'{config['model']['smi_model_card'].split('/')[-1]}_{config['model']['prot_model_card'].split('/')[-1]}_{split}.pt'
    torch.save(
        model.state_dict(), os.path.join(save_path, model_fp)
    )


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
            smi_token, prot_token, y, smiles, prot = i
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            y = y.to(device)

            # pass through model
            out = model(smi_token, prot_token)
            pred = out[0].squeeze()
            
            # calculate loss
            if 'data/M2OR' in config['training']['data_path']:
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
        if 'data/M2OR' in config['training']['data_path']:
            logits = torch.concat(preds)
            preds = torch.nn.Sigmoid()((logits)).cpu().numpy()
            bin_preds = (preds >= 0.5).astype(int)
            ave_p = average_precision_score(ground_truth, preds)
            precision = precision_score(ground_truth, bin_preds)
            recall = recall_score(ground_truth, bin_preds)
            f_score = f1_score(ground_truth, bin_preds)
            mcc = matthews_corrcoef(ground_truth, bin_preds)
            auroc = roc_auc_score(ground_truth, preds)
            print(f'Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} AveP: {ave_p:.4f} | {dataset} Precision: {precision:.4f} | {dataset} Recall: {recall:.4f} | {dataset} F1 score: {f_score:.4f} | {dataset} MCC: {mcc:.4f} | {dataset} AUROC: {auroc:.4f}')

            # write to tensorboard
            writer.add_scalar(f'Loss/{dataset}', avg_loss, epoch)
            writer.add_scalar(f'{dataset}_metrics/AveP', ave_p, epoch)
            writer.add_scalar(f'{dataset}_metrics/Precision', precision, epoch)
            writer.add_scalar(f'{dataset}_metrics/Recall', recall, epoch)
            writer.add_scalar(f'{dataset}_metrics/F1', f_score, epoch)
            writer.add_scalar(f'{dataset}_metrics/MCC', mcc, epoch)
            writer.add_scalar(f'{dataset}_metrics/AUROC', auroc, epoch)

            # write to wandb
            wandb.log({
                f'Loss/{dataset}': avg_loss,
                f'{dataset}_metrics/AveP': ave_p,
                f'{dataset}_metrics/Precision': precision,
                f'{dataset}_metrics/Recall': recall,
                f'{dataset}_metrics/F1': f_score,
                f'{dataset}_metrics/MCC': mcc,
                f'{dataset}_metrics/AUROC': auroc,
            }, step=epoch)

        else:
            preds = torch.concat(preds).cpu().numpy()
            r2 = r2_score(ground_truth, preds)
            ci = concordance_index(ground_truth, preds)
            print(f'Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} R2: {r2:.4f} | {dataset} CI: {ci:.4f}')

            # write to tensorboard
            writer.add_scalar(f'Loss/{dataset}', avg_loss, epoch)
            writer.add_scalar(f'{dataset}_metrics/{dataset}/R2', r2, epoch)
            writer.add_scalar(f'{dataset}_metrics/{dataset}/CI', ci, epoch)

            # scatter of predictions vs. actual
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(ground_truth, preds, s=8, alpha=0.5)
            lims = [min(ground_truth.min(), preds.min()),
                    max(ground_truth.max(), preds.max())]
            ax.plot(lims, lims, 'k--', linewidth=1)  # y = x reference
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{dataset} (R2={r2:.3f})')
            writer.add_figure(f'{dataset}_scatter/pred_vs_actual', fig, epoch)

            # write to wandb
            wandb.log({
                f'Loss/{dataset}': avg_loss,
                f'{dataset}_metrics/{dataset}/R2': r2,
                f'{dataset}_metrics/{dataset}/CI': ci,
                f'{dataset}_scatter/pred_vs_actual': wandb.Image(fig),
            }, step=epoch)

            plt.close(fig)

    return avg_loss


def get_dataloaders(
        config,
        split,
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card,
        generator=None
):
    """
    generates dataloaders for training
    """
    train_data = ProteinSmilesDataset(
        os.path.join(config['training']['data_path'], split, "train_df.csv"),
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card
    )
    val_data = ProteinSmilesDataset(
        os.path.join(config['training']['data_path'], split, "val_df.csv"),
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card
    )
    test_data = ProteinSmilesDataset(
        os.path.join(config['training']['data_path'], split, "test_df.csv"),
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card
    )
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config['train_lorax']['batch_size'],
        generator=generator,  # deterministic shuffle order
        worker_init_fn=seed_worker
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=config['train_lorax']['batch_size']
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config['train_lorax']['batch_size']
    )
    
    return (
        train_data,
        val_data,
        test_data,
        train_dataloader,
        val_dataloader,
        test_dataloader
    )


def train(gpu_id, config, split_batches, splits):
    """
    Main training loop for LORAX
    """
    # setup device and splits
    splits_for_this_gpu = split_batches[gpu_id]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    print(f'Training loop for split {splits_for_this_gpu} on {device}')

    # reproducibility settings (defaults keep older configs working)
    base_seed = config['training'].get('seed', 42)
    deterministic = config['training'].get('deterministic', True)

    # get foudation models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    # loop through splits assigned to this gpu
    for split in splits_for_this_gpu:
        print(f'Split: {split}')

        # Seed per split using a stable, GPU-assignment-independent offset so a
        # given split trains identically regardless of how many GPUs are used
        # or which one it lands on. This reseeds before model init and data
        # shuffling, the two RNG-consuming steps below.
        split_seed = base_seed + splits.index(split)
        set_seed(split_seed, deterministic=deterministic)
        data_generator = torch.Generator()
        data_generator.manual_seed(split_seed)

        # create tensorboard log
        log_dir = os.path.join(
            config['training']['log_path'],
            smi_model_card.split('/')[-1],
            'lorax',
            split
        )
        writer = SummaryWriter(log_dir=log_dir)

        # create wandb run (one per split)
        wandb.init(
            project=config['training'].get('wandb_project', 'lorax'),
            name=f"{smi_model_card.split('/')[-1]}-lorax-{split}",
            group=smi_model_card.split('/')[-1],
            config=config,
            reinit=True,
        )

        # dataloaders
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config,
            split,
            smi_model,
            prot_model,
            smi_model_card,
            prot_model_card,
            generator=data_generator
        )

        print('reloading models to remove old adapters')
        smi_model = AutoModel.from_pretrained(smi_model_card).to(device).eval()
        prot_model = AutoModel.from_pretrained(prot_model_card).to(device).eval()

        # init model
        model = LORAX(
            model_config=config['model'],
            smi_model=smi_model,
            prot_model=prot_model,
            no_cross_attn=config['model']['combine']['no_cross_attn'],
            no_prot_model_ft=config['model']['combine']['no_prot_model_ft'],
            lin_proj=config['model']['combine']['lin_proj']
        ).to(device)

        # init optimizer and loss_fn
        optim = torch.optim.Adam(
            params=model.parameters(),
            lr=config['train_lorax']['lr']
        )

        # use special loss function for M2OR data
        if 'data/M2OR' in config['training']['data_path']:
            loss_fn = M2ORWeightedCrossEntropyLoss(
                config['training']['data_path']
            )
        else:
            loss_fn = torch.nn.MSELoss()

        # look at parameters
        tot_train_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        tot_params = sum(p.numel() for p in model.parameters())
        print('combo model:')
        print(
            'trainable params:',
            '{:,}'.format(tot_train_params),
            ' || all params:',
            '{:,}'.format(tot_params),
            ' || trainable%:',
            f'{(tot_train_params/tot_params)*100:0.04}'
        )

        # training loop
        best_loss = 1e10
        first_split = splits[0]
        for epoch in range(config['train_lorax']['train_epochs']):

            # save initial representation
            if epoch == 0 and split == first_split:
                print('Saving initial molecular representation')
                save_molecular_rep(train_data, val_data, test_data, model, config, device, epoch)

            model.train()
            running_loss = []

            for dat in tqdm(train_dataloader, desc=f'Epoch {epoch} | batch'):
                # zero gradients
                optim.zero_grad()

                # unpack data
                smi_token, prot_token, y, smiles, prot = dat
                smi_token = {k: v.to(device) for k, v in smi_token.items()}
                prot_token = {k: v.to(device) for k, v in prot_token.items()}
                y = y.to(device)

                # pass through model
                out = model(smi_token, prot_token)
                pred = out[0].squeeze()

                # calculate loss and backprop
                if 'data/M2OR' in config['training']['data_path']:
                    loss = loss_fn(pred, y, smiles, prot)
                else:
                    loss = loss_fn(pred, y)

                loss.backward()
                optim.step()

                # bookeeping
                running_loss.append(loss.item())

            # evaluate
            avg_val_loss = evaluate(
                config,
                model,
                val_dataloader,
                device,
                loss_fn,
                epoch,
                writer,
                'val'
            )
            _ = evaluate(
                config,
                model,
                test_dataloader,
                device,
                loss_fn,
                epoch,
                writer,
                'test'
            )

            # save model when the best validation loss happens
            if avg_val_loss < best_loss:
                save_model(model, config, split)
                writer.add_scalar("Model/best_model", epoch, epoch)
                
                # save final molecular representions
                if split == first_split:
                    print('Saving final molecular representation')
                    save_molecular_rep(
                        train_data,
                        val_data,
                        test_data,
                        model,
                        config,
                        device,
                        "final"
                    )

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

        # cleanup
        del model, train_data, val_data, test_data, train_dataloader, val_dataloader, _


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, help='Path to config file', required=True
    )
    args = parser.parse_args()

    # load config
    config = yaml.safe_load(open(args.config, 'r'))

    # split dir
    splits = sorted(os.listdir(config['training']['data_path']))

    # distribute across gpus
    n_gpus = torch.cuda.device_count()

    if n_gpus != 0:
        split_batches = [[] for _ in range(n_gpus)]
        for i, split in enumerate(splits):
            split_batches[i % n_gpus].append(split)
        
        # call main training function
        mp.spawn(
            train,
            args=(
                config,
                split_batches,
                splits
            ),
            nprocs=n_gpus,
            join=True
        )

    # cpu only
    else:
        split_batches = [splits]

        # call main training function
        train(0, config, split_batches, splits)

    # save config
    config_save_pth = os.path.join(
        config['training']['results_path'],
        config['model']['smi_model_card'].split('/')[-1],
        'config.yaml'
    )
    with open(config_save_pth, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    main()
