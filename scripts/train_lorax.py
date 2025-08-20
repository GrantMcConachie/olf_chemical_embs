#!/usr/bin/env python3
"""
Script to train the lora model

export HF_HOME=/projectnb/depaqlab/Grant/lora/saved_models
"""

import os
import yaml
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index

import torch
import torch.multiprocessing as mp
from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.lorax import LORAX
from data_utils.prot_smi_dataset import ProteinSmilesDataset


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
            smi_reps.append(smi_rep)
            rep_attn_masks.append(rep_attn_mask)

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


def evaluate(model, val_dataloader, device, loss_fn, epoch, writer):
    """
    Evaluate model against the validation set
    """
    model.eval()
    with torch.no_grad():
        running_loss = []
        preds = []
        ground_truth = []

        for i in val_dataloader:
            # unpack data
            smi_token, prot_token, y, _, _ = i
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            y = y.to(device)

            # pass through model
            out = model(smi_token, prot_token)
            pred = out[0].squeeze()
            
            # calculate loss and backpropogate
            loss = loss_fn(pred, y)

            # bookeeping
            preds.append(pred)
            ground_truth.append(y)
            running_loss.append(loss.item())
        
        ground_truth = torch.concat(ground_truth).cpu().numpy()
        preds = torch.concat(preds).cpu().numpy()
        val_r2 = r2_score(ground_truth, preds)
        avg_val_loss = sum(running_loss) / len(running_loss)
        val_CI = concordance_index(ground_truth, preds)
        print(f'Epoch {epoch} | Avg Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f} | Val CI: {val_CI:.4f}')

        # write to tensorboard
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar('Metrics/val_R2', val_r2, epoch)
        writer.add_scalar('Metrics/val_CI', val_CI, epoch)

    return avg_val_loss


def get_dataloaders(
        config,
        split,
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card
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
        batch_size=config['train_lorax']['batch_size']
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
    Main training loop for the lora model
    """
    # setup device and splits
    splits_for_this_gpu = split_batches[gpu_id]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    print(f'Training loop for split {splits_for_this_gpu} on {device}')

    # get foudation models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    # loop through splits assigned to this gpu
    for split in splits_for_this_gpu:
        # create tensorboard log
        log_dir = os.path.join(
            config['training']['log_path'],
            smi_model_card.split('/')[-1],
            'lorax',
            split
        )
        writer = SummaryWriter(log_dir=log_dir)

        # dataloaders
        train_data, val_data, test_data, train_dataloader, val_dataloader, _ = get_dataloaders(
            config,
            split,
            smi_model,
            prot_model,
            smi_model_card,
            prot_model_card
        )

        # init model
        model = LORAX(
            model_config=config['model'],
            smi_model=smi_model,
            prot_model=prot_model,
            no_cross_attn=config['model']['combine']['no_cross_attn']
        ).to(device)

        # init optimizer and loss_fn
        optim = torch.optim.Adam(
            params=model.parameters(),
            lr=config['train_lorax']['lr']
        )
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
                smi_token, prot_token, y, _, _ = dat
                smi_token = {k: v.to(device) for k, v in smi_token.items()}
                prot_token = {k: v.to(device) for k, v in prot_token.items()}
                y = y.to(device)

                # pass through model
                out = model(smi_token, prot_token)
                pred = out[0].squeeze()

                # calculate loss and backpropogate
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()

                # bookeeping
                running_loss.append(loss.item())

            # evaluate
            avg_val_loss = evaluate(
                model,
                val_dataloader,
                device,
                loss_fn,
                epoch,
                writer
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
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        # write all pending events to log and close
        writer.flush()
        writer.close()

        # cleanup
        del model, train_data, val_data, test_data, train_dataloader, val_dataloader, _


def main():
    # load config
    config = yaml.safe_load(open('configs/no_rslora_config.yaml', 'r'))

    # split dir
    splits = sorted(os.listdir(config['training']['data_path']))

    # distribute across gpus
    n_gpus = torch.cuda.device_count()

    if n_gpus != 0:
        split_batches = [[] for _ in range(n_gpus)]
        for i, split in enumerate(splits):
            split_batches[i % n_gpus].append(split)
        
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
