#!/usr/bin/env python3
"""
Trains a gradient boosted descision tree on top of lorax

export HF_HOME=/projectnb/depaqlab/Grant/lora/saved_models
"""

import os
import yaml
import pandas as pd
import pickle as pkl
from functools import partial
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error

import torch
import torch.multiprocessing as mp
from transformers import AutoModel
from torch.utils.tensorboard import SummaryWriter

import xgboost as xgb
from hyperopt import fmin, hp, rand

from model.lorax import LORAX
from scripts.train_lorax import get_dataloaders


def hyperparam_objective(param, train_data, val_data, device):
    """
    objective to optimize hyperparameters for the xgboost model. 
    Adapted from prosmith
    """
    # setting params
    num_round = int(param["num_rounds"])
    param["tree_method"] = "hist"
    param['device'] = device
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'reg:squarederror' # NOTE: this is only for non-binary tasks
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
        device
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

    # loop through data
    with torch.no_grad():
        for dat in dataloader:
            smi_token, prot_token, y, smis, prots = dat
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}

            out = model(smi_token, prot_token)

            for i, j, k, l in zip(out[-1], smis, prots, y):
                cls_reps_dat.append(i.detach().cpu())
                smi_reps_dat.append(smi_reps[j])
                prot_reps_dat.append(prot_reps[k])
                ys.append(l)
                smi_prot.append((j, k))

        # convert to xgb.DMatrix
        cls_reps_dat = torch.stack(cls_reps_dat)
        prot_reps_dat = torch.stack(prot_reps_dat).squeeze()
        smi_reps_dat = torch.stack(smi_reps_dat).squeeze()
        labels = torch.stack(ys)

        cls_reps = xgb.DMatrix(cls_reps_dat, label=labels)
        prot_smi_reps = xgb.DMatrix(
            torch.concat((prot_reps_dat, smi_reps_dat), axis=-1),
            label=labels
        )
        prot_smi_cls_reps = xgb.DMatrix(
            torch.concat((prot_reps_dat, smi_reps_dat, cls_reps_dat), axis=-1),
            label=labels
        )

    return cls_reps, prot_smi_reps, prot_smi_cls_reps, smi_prot


def generate_foundation_reps(smi_model, prot_model, train_data, val_data, test_data, device):
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
        for smi, token in zip(all_smi, all_smi_tokens):
            if smi not in smi_reps:
                token = {k: v.to(device) for k, v in token.items()}
                rep = smi_model(**token).pooler_output  # embedding
                smi_reps[smi] = rep.detach().cpu()

        # combine prot tokens
        all_prot = prot_train + prot_val + prot_test
        all_prot_tokens = prot_tokens_train + prot_tokens_val + prot_tokens_test
        for prot, token in zip(all_prot, all_prot_tokens):
            if prot not in prot_reps:
                token = {k: v.to(device) for k, v in token.items()}
                rep = prot_model(**token).pooler_output  # embedding
                prot_reps[prot] = rep.detach().cpu()

    return smi_reps, prot_reps


def get_xgboost_preds(param, train_dat, test_dat, config, split, device, save, tree=None):
    """
    Generates xgboost model predictions and saves models
    """
    param["tree_method"] = "hist"
    param['device'] = device

    # generate predictions
    bst = xgb.train(param, train_dat, int(param['num_rounds']))

    # save model
    if save:
        save_path = os.path.join(
            config['training']['results_path'],
            config['model']['smi_model_card'].split('/')[-1],
            split,
            'xgboost'
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
    best_mse = 1000
    best_i, best_j, best_k = 0,0,0
    for i in [k/100 for k in range(0,100)]:
        for j in [k/100 for k in range(0,100)]:
            if i+j <=1:
                k = (1-i-j)
                y_val_pred = i * val_preds_cls + j * val_preds_prot_smi  + k * val_preds_prot_smi_cls
                mse = mean_squared_error(val_labels, y_val_pred)
                if mse < best_mse:
                    best_mse = mse
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
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device).eval()
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device).eval()

    # loop through data splits
    for i, split in enumerate(splits_for_this_gpu):
        print(f'Training loop for split {split}')

        # create tensorboard log
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
            smi_reps, prot_reps = generate_foundation_reps(
                smi_model,
                prot_model,
                train_data,
                val_data,
                test_data,
                device
            )

        # load in trained model
        model = LORAX(
            model_config=config['model'],
            smi_model=smi_model,
            prot_model=prot_model,
            no_cross_attn=config['model']['combine']['no_cross_attn']
        ).to(device)
        save_path = os.path.join(
            config['training']['results_path'],
            config['model']['smi_model_card'].split('/')[-1],
            split
        )
        state_dict = f'{config['model']['smi_model_card'].split('/')[-1]}_{config['model']['prot_model_card'].split('/')[-1]}_{split}.pt'
        state_dict = torch.load(os.path.join(save_path, state_dict), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # generate model representation
        print('generating LORAX representations')
        train_cls, train_prot_smi, train_prot_smi_cls, _ = generate_model_reps(
            model,
            train_dataloader,
            smi_reps,
            prot_reps,
            device
        )
        val_cls, val_prot_smi, val_prot_smi_cls, _ = generate_model_reps(
            model,
            val_dataloader,
            smi_reps,
            prot_reps,
            device
        )
        test_cls, test_prot_smi, test_prot_smi_cls, test_smi_prots = generate_model_reps(
            model,
            test_dataloader,
            smi_reps,
            prot_reps,
            device
        )
        train_val_cls, train_val_prot_smi, train_val_prot_smi_cls, _ = generate_model_reps(
            model,
            list(train_dataloader) + list(val_dataloader),
            smi_reps,
            prot_reps,
            device
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
            fn=partial(hyperparam_objective, train_data=train_cls, val_data=val_cls, device=device),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb] hyperparams')
        best_prot_smi = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi, val_data=val_prot_smi, device=device),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb + cls] hyperparams')
        best_prot_smi_cls = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi_cls, val_data=val_prot_smi_cls, device=device),
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
        r2 = r2_score(ground_truth, preds)
        CI = concordance_index(ground_truth, preds)
        mse = mean_squared_error(ground_truth, preds)
        print(f'{split} | Test MSE: {mse:.4f} | Test R2: {r2:.4f} | Test CI: {CI:.4f}')
        writer.add_scalar("Metrics/test_R2", r2, 0)
        writer.add_scalar("Metrics/test_CI", CI, 0)
        writer.add_scalar("Metrics/test_mse", mse, 0)

        # save predictions
        save_predictions(config, split, preds, test_smi_prots)


def main():
    # load config from pretrained model
    config = yaml.safe_load(open('configs/config.yaml', 'r'))

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
