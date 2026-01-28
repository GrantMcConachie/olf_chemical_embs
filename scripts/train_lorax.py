"""
Script to train the lora model
"""

import os
import yaml
import argparse
import pickle as pkl
from tqdm import tqdm
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
from unimol_tools.models.unimolv2 import UniMolV2Model
from unimol_tools.data.conformer import UniMolV2Feature

from model.lorax import LORAX
from data_utils.prot_smi_dataset import ProteinSmilesDataset
from loss_functions.loss_functions import M2ORWeightedCrossEntropyLoss


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

        # get unimol featurizer
        if config['model']['smi_model_card'] == 'unimol':
            smi_model = UniMolV2Model(
                pretrained_model_path='/projectnb/depaqlab/Grant/lora/saved_models/unimol/checkpoint.pt',  # TODO: put this in the config file instead of hard code
                data_type='molecule'
            )
            smi_featurizer_unimol = UniMolV2Feature(multi_process=False)


        for i in dataloader:
            # unpack data
            smi_token, prot_token, y, smiles, prot = i

            if config['model']['smi_model_card'] == 'unimol':
                smi_token = smi_featurizer_unimol.transform(smiles)[0]
                smi_token = [(f, None) for f in smi_token]
                smi_token, _ = smi_model.batch_collate_fn(smi_token)
                smi_token = {k: v.to(device) for k, v in smi_token.items()}
            else:
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
            auroc = roc_auc_score(ground_truth, bin_preds)
            print(f'Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} AveP: {ave_p:.4f} | {dataset} Precision: {precision:.4f} | {dataset} Recall: {recall:.4f} | {dataset} F1 score: {f_score:.4f} | {dataset} MCC: {mcc:.4f} | {dataset} AUROC: {auroc:.4f}')

            # write to tensorboard
            writer.add_scalar(f'Loss/{dataset}', avg_loss, epoch)
            writer.add_scalar(f'{dataset}_metrics/AveP', ave_p, epoch)
            writer.add_scalar(f'{dataset}_metrics/Precision', precision, epoch)
            writer.add_scalar(f'{dataset}_metrics/Recall', recall, epoch)
            writer.add_scalar(f'{dataset}_metrics/F1', f_score, epoch)
            writer.add_scalar(f'{dataset}_metrics/MCC', mcc, epoch)
            writer.add_scalar(f'{dataset}_metrics/AUROC', auroc, epoch)

        else:
            preds = torch.concat(preds).cpu().numpy()
            r2 = r2_score(ground_truth, preds)
            ci = concordance_index(ground_truth, preds)
            print(f'Epoch {epoch} | Avg {dataset} Loss: {avg_loss:.4f} | {dataset} R2: {r2:.4f} | {dataset} CI: {ci:.4f}')

            # write to tensorboard
            writer.add_scalar(f'Loss/{dataset}', avg_loss, epoch)
            writer.add_scalar(f'{dataset}_metrics/{dataset}/R2', r2, epoch)
            writer.add_scalar(f'{dataset}_metrics/{dataset}/CI', ci, epoch)

    return avg_loss


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
    Main training loop for LORAX
    """
    # setup device and splits
    splits_for_this_gpu = split_batches[gpu_id]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    print(f'Training loop for split {splits_for_this_gpu} on {device}')

    # get foudation models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    if smi_model_card == 'unimol':
        smi_model = UniMolV2Model(
            pretrained_model_path='/projectnb/depaqlab/Grant/lora/saved_models/unimol/checkpoint.pt',  # TODO: put this in the config file instead of hard code
            data_type='molecule'
        )
        smi_featurizer_unimol = UniMolV2Feature(multi_process=False)
    else:
        smi_model = AutoModel.from_pretrained(smi_model_card, force_download=True).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card, force_download=True).to(device)

    # loop through splits assigned to this gpu
    for split in splits_for_this_gpu:
        print(f'Split: {split}')

        # create tensorboard log
        log_dir = os.path.join(
            config['training']['log_path'],
            smi_model_card.split('/')[-1],
            'lorax',
            split
        )
        writer = SummaryWriter(log_dir=log_dir)

        # dataloaders
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config,
            split,
            smi_model,
            prot_model,
            smi_model_card,
            prot_model_card
        )

        print('reloading models to remove old adapters')
        if smi_model_card == 'unimol':
            smi_model = UniMolV2Model(
                pretrained_model_path='/projectnb/depaqlab/Grant/lora/saved_models/unimol/checkpoint.pt',  # TODO: put this in the config file instead of hard code
                data_type='molecule'
            )
        else:
            smi_model = AutoModel.from_pretrained(smi_model_card, force_download=True).to(device)
        prot_model = AutoModel.from_pretrained(prot_model_card, force_download=True).to(device).eval()

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
            # if epoch == 0 and split == first_split:
            #     print('Saving initial molecular representation')
            #     save_molecular_rep(train_data, val_data, test_data, model, config, device, epoch)

            model.train()
            running_loss = []

            for dat in tqdm(train_dataloader, desc=f'Epoch {epoch} | batch'):
                # zero gradients
                optim.zero_grad()

                # unpack data
                smi_token, prot_token, y, smiles, prot = dat

                # dealing with unimol
                if smi_model_card == 'unimol':
                    with torch.no_grad():
                        smi_token = smi_featurizer_unimol.transform(smiles)[0]
                        smi_token = [(f, None) for f in smi_token]
                        smi_token, _ = smi_model.batch_collate_fn(smi_token)
                        smi_token = {k: v.to(device) for k, v in smi_token.items()}
                else:
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
                    # save_molecular_rep(
                    #     train_data,
                    #     val_data,
                    #     test_data,
                    #     model,
                    #     config,
                    #     device,
                    #     "final"
                    # )

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
