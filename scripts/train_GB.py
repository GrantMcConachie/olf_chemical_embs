"""
Trains a gradient boosted descision tree on top of lorax
"""

import os
import yaml
import argparse
import pandas as pd
import pickle as pkl
from functools import partial
from lifelines.utils import concordance_index
from sklearn.metrics import (
    r2_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    mean_squared_error
)

import torch
import torch.multiprocessing as mp
from transformers import AutoModel
from torch.utils.tensorboard import SummaryWriter
from unimol_tools.models.unimolv2 import UniMolV2Model
from unimol_tools.data.conformer import UniMolV2Feature

import xgboost as xgb
from hyperopt import fmin, hp, rand

from model.lorax import LORAX
from scripts.train_lorax import get_dataloaders
from loss_functions.loss_functions import M2ORWeightedCrossEntropyLoss


def hyperparam_objective(param, train_data, val_data, device, config):
    """
    objective to optimize hyperparameters for the xgboost model. 
    Adapted from prosmith
    """
    # setting params
    num_round = int(param["num_rounds"])
    param["tree_method"] = "hist"
    param['device'] = device
    param["sampling_method"] = "gradient_based"

    # changing objective between binary and regression tasks
    if 'data/M2OR' in config['training']['data_path']:
        param['objective'] = 'binary:logistic'
    else:
        param['objective'] = 'reg:squarederror'

    del param["num_rounds"]
    del param["weight"]

    # training xgboost
    bst = xgb.train(param, train_data, num_round)
    mse = mean_squared_error(val_data.get_label(), bst.predict(val_data))

    return mse


def generate_model_reps(
        model,
        dataloader,
        smi_reps,
        prot_reps,
        device,
        config
):
    """
    generates lorax model representations to feed into xgboost
    """
    # init
    cls_reps_dat = []
    smi_reps_dat = []
    prot_reps_dat = []
    ys = []
    smi_prot = []

    # add weighting if M2OR data
    if 'data/M2OR' in config['training']['data_path']:
        loss = M2ORWeightedCrossEntropyLoss(
                config['training']['data_path']
            )
        weights = []
    else:
        weights = None

    # loop through data
    with torch.no_grad():
        # unimol specific featurizer and model
        smi_featurizer_unimol = UniMolV2Feature(multi_process=False)
        smi_model = UniMolV2Model(
            pretrained_model_path='/projectnb/depaqlab/Grant/lora/saved_models/unimol/checkpoint.pt',  # TODO: put this in the config file instead of hard code
            data_type='molecule'
        )

        for dat in dataloader:
            smi_token, prot_token, y, smis, prots = dat
            # dealing with unimol
            if config['model']['smi_model_card'] == 'unimol':
                smi_token = smi_featurizer_unimol.transform(smis)[0]
                smi_token = [(f, None) for f in smi_token]
                smi_token, _ = smi_model.batch_collate_fn(smi_token)
                smi_token = {k: v.to(device) for k, v in smi_token.items()}
            else:
                smi_token = {k: v.to(device) for k, v in smi_token.items()}

            prot_token = {k: v.to(device) for k, v in prot_token.items()}

            out = model(smi_token, prot_token)

            for i, j, k, l in zip(out[3], smis, prots, y):
                cls_reps_dat.append(i.detach().cpu())
                ys.append(l)
                smi_prot.append((j, k))
                if not config['train_GB']['use_lorax_embs']:
                    # using smi and protein representations from the original
                    # foundation models
                    smi_reps_dat.append(smi_reps[j])
                    prot_reps_dat.append(prot_reps[k])
                
                if 'data/M2OR' in config['training']['data_path']:
                    # assign weights to the data if using the m2or dataset
                    w_quality = loss.data_quality_w[(j, k)]
                    w_class = loss.pos_class_weight if l else loss.neg_class_weight
                    w_pair = loss.pair_imbalance_weight(j, k)

                    # get total weighting append to weights
                    w_tot = w_quality * w_class * w_pair
                    weights.append(w_tot)

            if config['train_GB']['use_lorax_embs']:
                # using protein and smiles representations from the model
                # itself
                for s_rep, s_mask, p_rep, p_mask in zip(out[1], out[2], out[4], out[5]):
                    smi_rep = (s_rep * s_mask.unsqueeze(-1)).sum(dim=0) / (s_mask.unsqueeze(-1).sum(dim=0) + 1e-8)
                    prot_rep = (p_rep * p_mask.unsqueeze(-1)).sum(dim=0) / (p_mask.unsqueeze(-1).sum(dim=0) + 1e-8)
                    smi_reps_dat.append(smi_rep.detach().cpu())
                    prot_reps_dat.append(prot_rep.detach().cpu())                    

        # convert to xgb.DMatrix
        cls_reps_dat = torch.stack(cls_reps_dat)
        prot_reps_dat = torch.stack(prot_reps_dat).squeeze()
        smi_reps_dat = torch.stack(smi_reps_dat).squeeze()
        labels = torch.stack(ys)
        if 'data/M2OR' in config['training']['data_path']:
            weights = torch.tensor(weights, dtype=torch.float, device='cpu')

        cls_reps = xgb.DMatrix(cls_reps_dat, label=labels, weight=weights)
        prot_smi_reps = xgb.DMatrix(
            torch.concat((prot_reps_dat, smi_reps_dat), axis=-1),
            label=labels,
            weight=weights
        )
        prot_smi_cls_reps = xgb.DMatrix(
            torch.concat((prot_reps_dat, smi_reps_dat, cls_reps_dat), axis=-1),
            label=labels,
            weight=weights
        )

    return cls_reps, prot_smi_reps, prot_smi_cls_reps, smi_prot


def generate_foundation_reps(smi_model, prot_model, train_data, val_data, test_data, config, device, mode='train'):
    """
    generates foundation model representations to feed into xgboost
    """
    # init
    smi_reps = {}
    prot_reps = {}

    # get smiles and prots
    smi_train, smi_tokens_train = train_data.get_unique_smiles_rep()
    prot_train, prot_tokens_train = train_data.get_unique_prot_rep()
    smi_val, smi_tokens_val = val_data.get_unique_smiles_rep()
    prot_val, prot_tokens_val = val_data.get_unique_prot_rep()
    smi_test, smi_tokens_test = test_data.get_unique_smiles_rep()
    prot_test, prot_tokens_test = test_data.get_unique_prot_rep()

    # combine smiles tokens
    with torch.no_grad():
        all_smi = smi_train + smi_val + smi_test
        all_smi_tokens = smi_tokens_train + smi_tokens_val + smi_tokens_test

        if config['model']['smi_model_card'] == 'unimol':
            featurizer = UniMolV2Feature(multi_process=False)
            for smi in all_smi:
                if smi not in smi_reps:
                    feats = featurizer.transform([smi])[0]
                    feats = [(f, None) for f in feats]  # add dummy label for batching
                    batch_dict, labels = smi_model.batch_collate_fn(feats)
                    rep = smi_model(**batch_dict, return_repr=True)
                    smi_reps[smi] = rep.detach().cpu()
        else:
            for smi, token in zip(all_smi, all_smi_tokens):
                if smi not in smi_reps:
                    token = {k: v.to(device) for k, v in token.items()}
                    smi_mask = token['attention_mask']
                    if mode == 'train':
                        rep = smi_model(**token).pooler_output
                    if mode == 'inference':
                        rep = smi_model(**token).last_hidden_state  # embedding
                        rep = (rep * smi_mask.unsqueeze(-1)).sum(dim=1) / (smi_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
                    smi_reps[smi] = rep.detach().cpu()

        # combine prot tokens
        all_prot = prot_train + prot_val + prot_test
        all_prot_tokens = prot_tokens_train + prot_tokens_val + prot_tokens_test
        for prot, token in zip(all_prot, all_prot_tokens):
            if prot not in prot_reps:
                token = {k: v.to(device) for k, v in token.items()}
                prot_mask = token['attention_mask']
                if mode == 'train':
                    rep = prot_model(**token).pooler_output 
                if mode == 'inference':
                    rep = prot_model(**token).last_hidden_state  # embedding
                    rep = (rep * prot_mask.unsqueeze(-1)).sum(dim=1) / (prot_mask.unsqueeze(-1).sum(dim=1) + 1e-8)
                prot_reps[prot] = rep.detach().cpu()

    return smi_reps, prot_reps


def get_xgboost_preds(param, train_dat, test_dat, config, split, device, save, tree=None):
    """
    Generates xgboost model predictions and saves models
    """
    param["tree_method"] = "hist"
    param['device'] = device
    if 'data/M2OR' in config['training']['data_path']:
        param['objective'] = 'binary:logistic'
    else:
        param['objective'] = 'reg:squarederror'

    # generate predictions
    bst = xgb.train(param, train_dat, int(param['num_rounds']))

    # save model
    if save:
        xg_tree = "all_lorax_tree" if config['train_GB']['use_lorax_embs'] else "tree"
        save_path = os.path.join(
            config['training']['results_path'],
            config['model']['smi_model_card'].split('/')[-1],
            split,
            'xgboost',
            xg_tree
        )
        os.makedirs(save_path, exist_ok=True)
        model_fp = f'{config['model']['smi_model_card'].split('/')[-1]}_{config['model']['prot_model_card'].split('/')[-1]}_{split}_{tree}_GB.pkl'
        with open(os.path.join(save_path, model_fp), 'wb') as f:
            pkl.dump((bst, param), f)
        f.close()

    return bst.predict(test_dat)


def find_best_proportion(
        val_preds_cls,
        val_preds_prot_smi,
        val_preds_prot_smi_cls,
        val_labels,
        config,
        split
):
    """
    Adapted from prosmith. Finds the best proportion of
    models to use in the final prediction.
    """
    best_loss = -1000
    best_i, best_j, best_k = 0,0,0
    for i in [k/100 for k in range(0,100)]:
        for j in [k/100 for k in range(0,100)]:
            if i+j <=1:
                k = (1-i-j)
                y_val_pred = i * val_preds_cls + j * val_preds_prot_smi  + k * val_preds_prot_smi_cls

                if 'data/M2OR' in config['training']['data_path']:
                    loss = matthews_corrcoef(val_labels, (y_val_pred >= 0.5).astype(int))
                else:
                    loss = -mean_squared_error(val_labels, y_val_pred)

                if loss > best_loss:
                    best_loss = loss
                    best_i, best_j, best_k = i, j, k

    # report best proportion
    print(f'Best Proportion: {best_i} [cls] | {best_j} [prot + smi] | {best_k} [prot + smi + cls]')

    # saving
    save_path = os.path.join(
        config['training']['results_path'],
        config['model']['smi_model_card'].split('/')[-1],
        split
    )
    os.makedirs(save_path, exist_ok=True)
    model_fp = f'GB_proportion_{split}.pkl'
    with open(os.path.join(save_path, model_fp), 'wb') as f:
        pkl.dump((best_i, best_j, best_k), f)
    f.close()

    return best_i, best_j, best_k


def save_predictions(config, split, preds, test_smi_prots):
    """
    Saves predictions of best model
    """
    # get test df
    df = pd.read_csv(
        os.path.join(config['training']['data_path'], split, 'test_df.csv')
    )
    
    # append predictions to test df
    y_preds = []
    for i, row in df.iterrows():
        smi_prot = (row['SMILES'], row['Protein sequence'])
        idx = test_smi_prots.index(smi_prot)
        y_preds.append(preds[idx])

    # append to df
    df['Model Predictions'] = y_preds

    # save predictions
    save_path = os.path.join(
        config['training']['results_path'],
        config['model']['smi_model_card'].split('/')[-1],
        split
    )
    os.makedirs(save_path, exist_ok=True)
    model_fp = f'preds_{split}.csv'
    df.to_csv(os.path.join(save_path, model_fp))


def train(gpu_id, config, split_batches):
    """
    training function for the GB descision tree
    """
    # Setup device and splits
    splits_for_this_gpu = split_batches[gpu_id]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
    print(f'Training loop for split {splits_for_this_gpu} on {device}')

    # load models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    if smi_model_card == 'unimol':
        smi_model = UniMolV2Model(
            pretrained_model_path='/projectnb/depaqlab/Grant/lora/saved_models/unimol/checkpoint.pt',  # TODO: put this in the config file instead of hard code
            data_type='molecule'
        )
    else:
        smi_model = AutoModel.from_pretrained(smi_model_card, force_download=True).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card, force_download=True).to(device).eval()

    # loop through data splits
    for i, split in enumerate(splits_for_this_gpu):
        print(f'Training loop for split {split}')

        # create tensorboard log
        if config['train_GB']['use_lorax_embs']:
            log_dir = os.path.join(
                config['training']['log_path'],
                smi_model_card.split('/')[-1],
                'all_lorax_tree',
                split
            )

        else:
            log_dir = os.path.join(
                config['training']['log_path'],
                smi_model_card.split('/')[-1],
                'tree',
                split
            )

        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # generate representations to put into xgboost
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config,
            split,
            smi_model,
            prot_model,
            smi_model_card,
            prot_model_card
        )

        # get all LMM and LPM represnetations from first batch
        if i == 0:
            print('generating foundation model representations')
            if config['train_GB']['use_lorax_embs']:
                smi_reps, prot_reps = None, None
            else:
                smi_reps, prot_reps = generate_foundation_reps(
                    smi_model,
                    prot_model,
                    train_data,
                    val_data,
                    test_data,
                    config,
                    device,
                    'train'
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

        # load in trained model
        model = LORAX(
            model_config=config['model'],
            smi_model=smi_model,
            prot_model=prot_model,
            no_cross_attn=config['model']['combine']['no_cross_attn'],
            no_prot_model_ft=config['model']['combine']['no_prot_model_ft'],
            lin_proj=config['model']['combine']['lin_proj']
        ).to(device)
        save_path = os.path.join(
            config['training']['results_path'],
            config['model']['smi_model_card'].split('/')[-1],
            split
        )
        state_dict = f'{config['model']['smi_model_card'].split('/')[-1]}_{config['model']['prot_model_card'].split('/')[-1]}_{split}.pt'
        state_dict = torch.load(os.path.join(save_path, state_dict), map_location=device)

        # update state_dict to v2 of model
        if 'proj.0.weight' not in state_dict.keys():
            state_dict['proj.0.weight'] = state_dict['mlp.0.weight']
            state_dict['proj.0.bias'] = state_dict['mlp.0.bias']
            state_dict['proj.2.weight'] = state_dict['mlp.2.weight']
            state_dict['proj.2.bias'] = state_dict['mlp.2.bias']
            state_dict['proj.4.weight'] = state_dict['mlp.4.weight']
            state_dict['proj.4.bias'] = state_dict['mlp.4.bias']

            del state_dict['mlp.0.weight']
            del state_dict['mlp.0.bias']
            del state_dict['mlp.2.weight']
            del state_dict['mlp.2.bias']
            del state_dict['mlp.4.weight']
            del state_dict['mlp.4.bias']

        model.load_state_dict(state_dict)
        model.eval()

        # generate model representation
        print('generating LORAX representations')
        train_cls, train_prot_smi, train_prot_smi_cls, _ = generate_model_reps(
            model,
            train_dataloader,
            smi_reps,
            prot_reps,
            device,
            config
        )
        val_cls, val_prot_smi, val_prot_smi_cls, _ = generate_model_reps(
            model,
            val_dataloader,
            smi_reps,
            prot_reps,
            device,
            config
        )
        test_cls, test_prot_smi, test_prot_smi_cls, test_smi_prots = generate_model_reps(
            model,
            test_dataloader,
            smi_reps,
            prot_reps,
            device,
            config
        )
        train_val_cls, train_val_prot_smi, train_val_prot_smi_cls, _ = generate_model_reps(
            model,
            list(train_dataloader) + list(val_dataloader),
            smi_reps,
            prot_reps,
            device,
            config
        )

        # optimize xgboost hyperparams
        space_search = {  # taken from prosmith hyperparam optimization
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
            "max_depth": hp.choice("max_depth", [6,7,8,9,10,11,12,13,14]),
            "reg_lambda": hp.uniform("reg_lambda", 0, 5),
            "reg_alpha": hp.uniform("reg_alpha", 0, 5),
            "max_delta_step": hp.uniform("max_delta_step", 0, 5),
            "min_child_weight": hp.uniform("min_child_weight", 0.1, 15),
            "num_rounds":  hp.uniform("num_rounds", 30, 1000),
            "weight" : hp.uniform("weight", 0.01, 0.99)
        }
        print('optimizing xgboost [cls] hyperparams')
        best_cls = fmin(
            fn=partial(hyperparam_objective, train_data=train_cls, val_data=val_cls, device=device, config=config),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb] hyperparams')
        best_prot_smi = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi, val_data=val_prot_smi, device=device, config=config),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb + cls] hyperparams')
        best_prot_smi_cls = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi_cls, val_data=val_prot_smi_cls, device=device, config=config),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        # get validation predictions
        val_preds_cls = get_xgboost_preds(
            best_cls, train_cls, val_cls, config, split, device, save=False
        )
        val_preds_prot_smi = get_xgboost_preds(
            best_prot_smi, train_prot_smi, val_prot_smi, config, split, device, save=False
        )
        val_preds_prot_smi_cls = get_xgboost_preds(
            best_prot_smi_cls, train_prot_smi_cls, val_prot_smi_cls, config, split, device, save=False
        )

        # getting best proportion of xgboost models to test
        best_i, best_j, best_k = find_best_proportion(
            val_preds_cls, val_preds_prot_smi, val_preds_prot_smi_cls, val_cls.get_label(), config, split
        )
        
        # get test predictions
        test_preds_cls = get_xgboost_preds(
            best_cls, train_val_cls, test_cls, config, split, device, save=True, tree='cls'
        )
        test_preds_prot_smi = get_xgboost_preds(
            best_prot_smi, train_val_prot_smi, test_prot_smi, config, split, device, save=True, tree='prot_smi'
        )
        test_preds_prot_smi_cls = get_xgboost_preds(
            best_prot_smi_cls, train_val_prot_smi_cls, test_prot_smi_cls, config, split, device, save=True, tree='prot_smi_cls'
        )

        # getting results
        preds = best_i * test_preds_cls + best_j * test_preds_prot_smi + best_k * test_preds_prot_smi_cls
        ground_truth = test_cls.get_label()
        
        if 'data/M2OR' in config['training']['data_path']:
            # metrics for M2OR
            bin_preds = (preds >= 0.5).astype(int)
            ave_p = average_precision_score(ground_truth, preds)
            precision = precision_score(ground_truth, bin_preds)
            recall = recall_score(ground_truth, bin_preds)
            f_score = f1_score(ground_truth, bin_preds)
            mcc = matthews_corrcoef(ground_truth, bin_preds)
            auroc = roc_auc_score(ground_truth, bin_preds)
            print(f'Epoch {split} | AveP: {ave_p:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 score: {f_score:.4f} | MCC: {mcc:.4f} | AUROC: {auroc:.4f}')

            # write to tensorboard
            writer.add_scalar(f'GB_metrics/AveP', ave_p, 0)
            writer.add_scalar(f'GB_metrics/Precision', precision, 0)
            writer.add_scalar(f'GB_metrics/Recall', recall, 0)
            writer.add_scalar(f'GB_metrics/F1', f_score, 0)
            writer.add_scalar(f'GB_metrics/MCC', mcc, 0)
            writer.add_scalar(f'GB_metrics/AUROC', auroc, 0)

        else:
            # Metrics for everything else
            r2 = r2_score(ground_truth, preds)
            CI = concordance_index(ground_truth, preds)
            mse = mean_squared_error(ground_truth, preds)
            print(f'{split} | Test MSE: {mse:.4f} | Test R2: {r2:.4f} | Test CI: {CI:.4f}')
            writer.add_scalar("GB_metrics/test_R2", r2, 0)
            writer.add_scalar("GB_metrics/test_CI", CI, 0)
            writer.add_scalar("GB_metrics/test_mse", mse, 0)

        # save predictions
        save_predictions(config, split, preds, test_smi_prots)


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
        
        mp.spawn(
            train,
            args=(
                config,
                split_batches
            ),
            nprocs=n_gpus,
            join=True
        )

    # cpu only
    else:
        split_batches = [splits]
        train(0, config, split_batches)


if __name__ == '__main__':
    main()
