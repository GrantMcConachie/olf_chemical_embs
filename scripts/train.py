"""
Script to train the lora model

export HF_HOME=/projectnb/depaqlab/Grant/lora/saved_models
"""

import os
import yaml
import pickle as pkl
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index

import torch
from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model import ProteinSmilesLoraModel
from data_utils.prot_smi_dataset import ProteinSmilesDataset


def save_molecular_rep(train_data, val_data, test_data, model, config, device):
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
    _, prot_token, _ = next(iter(train_data))
    prot_token = {k: v.to(device) for k, v in prot_token.items()}

    # get model representation of all smiles
    model.eval()
    with torch.no_grad():
        for key, value in smi_tokens.items():
            smi_token = value
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            _, smi_rep, rep_attn_mask = model(smi_token, prot_token).squeeze()

            # append arrays
            smiles.append(key)
            smi_reps.append(smi_rep)
            rep_attn_masks.append(rep_attn_mask)

    # save representations
    save_path = os.path.join(
        config['training']['results_path'],
        config['smi_model_card'].split('/')[-1]
    )
    os.makedirs(save_path, exist_ok=True)

    pkl.dump(smiles, open(os.path.join(save_path, 'smiles.pkl'), 'wb'))
    pkl.dump(smi_reps, open(os.path.join(save_path, 'smi_reps.pkl'), 'wb'))
    pkl.dump(rep_attn_masks, open(os.path.join(save_path, 'rep_attn_masks.pkl'), 'wb'))


def save_model(model, config, split):
    """
    Saves model
    """
    save_path = os.path.join(
        config['training']['results_path'],
        config['smi_model_card'].split('/')[-1],
        split
    )
    os.makedirs(save_path, exist_ok=True)
    model_fp = f'{config['smi_model_card'].split('/')[-1]}_{config['prot_model_card'].split('/')[-1]}_{split}.pt'
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
            smi_token, prot_token, y = i
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            y = y.to(device)

            # pass through model
            pred, _, _ = model(smi_token, prot_token).squeeze()
            
            # calculate loss and backpropogate
            loss = loss_fn(pred, y)

            # bookeeping
            preds.append(pred.squeeze())
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


def train(config):
    """
    Main training loop for the lora model
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    for split in os.listdir(config['training']['data_path']):
        print(f'Training loop for split {split}')

        # create tensorboard log
        log_dir = os.path.join(
            config['training']['log_path'],
            smi_model_card.split('/')[-1],
            split
        )
        writer = SummaryWriter(log_dir=log_dir)

        # dataloaders
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
            batch_size=config['training']['batch_size']
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=config['training']['batch_size']
        )

        # init model
        model = ProteinSmilesLoraModel(
            model_config=config['model'],
            smi_model=smi_model,
            prot_model=prot_model,
        ).to(device)

        # init optimizer and loss_fn
        optim = torch.optim.Adam(
            params=model.parameters(),
            lr=config['training']['lr']
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
        for epoch in range(config['training']['train_epochs']):
            model.train()
            running_loss = []

            for dat in tqdm(train_dataloader, desc=f'epoch {epoch} | batch'):
                # zero gradients
                optim.zero_grad()

                # unpack data
                smi_token, prot_token, y = dat
                smi_token = {k: v.to(device) for k, v in smi_token.items()}
                prot_token = {k: v.to(device) for k, v in prot_token.items()}
                y = y.to(device)

                # pass through model
                pred, _, _ = model(
                    smi_token, prot_token
                )
                pred = pred.squeeze()

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

            # report train loss
            avg_loss = sum(running_loss) / len(running_loss)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            # save molecular representions
            if split == os.listdir(config['training']['data_path'])[-1]:
                print('Saving molecular representation')
                save_molecular_rep(train_data, val_data, test_data, model, config, device)


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    train(config)
