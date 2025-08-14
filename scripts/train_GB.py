"""
Trains a gradient boosted descision tree on top of lorax
"""

import os
import yaml
from functools import partial
from sklearn.metrics import mean_squared_error

import torch
from transformers import AutoModel
from torch.utils.tensorboard import SummaryWriter

import xgboost as xgb
from hyperopt import fmin, hp, rand

from scripts.train_lorax import get_dataloaders


def hyperparam_objective(param, train_data, val_data):
    """
    objective to optimize hyperparameters for the xgboost model. 
    Adapted from prosmith
    """
    # setting params
    num_round = int(param["num_rounds"])
    param["tree_method"] = "gpu_hist"
    param["sampling_method"] = "gradient_based"
    param['objective'] = 'reg:squarederror' # NOTE: this is only for non-binary tasks
    del param["num_rounds"]
    del param["weight"]

    # training xgboost
    bst = xgb.train(param, train_data, num_round)

    return mean_squared_error(val_data.label, bst.predict(val_data))


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
    cls_reps = []
    smi_reps_dat = []
    prot_reps_dat = []
    ys = []

    # loop through data
    for dat in dataloader:
        smi_token, prot_token, y, smis, prots = dat
        smi_token = {k: v.to(device) for k, v in smi_token.items()}
        prot_token = {k: v.to(device) for k, v in prot_token.items()}

        out = model(smi_token, prot_token)

        cls_reps.append(out[-1])
        [smi_reps_dat.append(smi_reps[i]) for i in smis]
        [prot_reps_dat.append(prot_reps[i]) for i in prots]
        ys.append(y)

    # TODO: turn these into xgb.DMatrix 

    return cls_reps, smi_reps_dat, prot_reps_dat, ys


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
    all_smi = smi_train + smi_val + smi_test
    all_smi_tokens = smi_tokens_train + smi_tokens_val + smi_tokens_test
    for smi, token in zip(all_smi, all_smi_tokens):
        if smi not in smi_reps:
            token = {k: v.to(device) for k, v in token.items()}
            rep = smi_model(token).pooler_output  # embedding
            smi_reps[smi] = rep

    # combine prot tokens
    all_prot = prot_train + prot_val + prot_test
    all_prot_tokens = prot_tokens_train + prot_tokens_val + prot_tokens_test
    for prot, token in zip(all_prot, all_prot_tokens):
        if prot not in prot_reps:
            token = {k: v.to(device) for k, v in token.items()}
            rep = prot_model(token).pooler_output  # embedding
            prot_reps[prot] = rep

    return smi_reps, prot_reps


def train(config):
    """
    training function for the GB descision tree
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    smi_model_card = config['model']['smi_model_card']
    prot_model_card = config['model']['prot_model_card']
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    # loop through data splits
    for i, split in enumerate(os.listdir(config['training']['data_path'])):
        print(f'Training loop for split {split}')

        # create tensorboard log
        log_dir = os.path.join(
            config['training']['log_path'],
            smi_model_card.split('/')[-1],
            'tree',
            split
        )
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
            smi_reps, prot_reps = generate_foundation_reps(  # TODO: use these and feed them into the xgboost model, also figure out what hyperopt + xgboost tricks they use in prosmith...
                smi_model,
                prot_model,
                train_data,
                val_data,
                test_data,
                device
            )

        # load in trained model
        save_path = os.path.join(
            config['training']['results_path'],
            config['model']['smi_model_card'].split('/')[-1],
            split
        )
        model_fp = f'{config['model']['smi_model_card'].split('/')[-1]}_{config['model']['prot_model_card'].split('/')[-1]}_{split}.pt'
        model = torch.load(os.path.join(save_path, model_fp), map_location=device)
        model.eval()

        # generate model representation
        train_cls, train_prot_smi, train_prot_smi_cls, train_y = generate_model_reps(
            model,
            train_dataloader,
            smi_reps,
            prot_reps,
            device
        )
        val_cls, val_prot_smi, val_prot_smi_cls, val_y = generate_model_reps(
            model,
            val_dataloader,
            smi_reps,
            prot_reps,
            device
        )
        test_cls, test_prot_smi, test_prot_smi_cls, test_y = generate_model_reps(
            model,
            test_dataloader,
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
            "weight" : hp.uniform("weight", 0.01,0.99)
        }
        print('optimizing xgboost [cls] hyperparams')
        best_cls = fmin(
            fn=partial(hyperparam_objective, train_data=train_cls, val_data=val_cls),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb] hyperparams')
        best_prot_smi = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi, val_data=val_prot_smi),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        print('optimizing xgboost [prot_emb + smi_emb + cls] hyperparams')
        best_prot_smi_cls = fmin(
            fn=partial(hyperparam_objective, train_data=train_prot_smi_cls, val_data=val_prot_smi_cls),
            space=space_search,
            algo=rand.suggest,
            max_evals=config['train_GB']['max_evals'],
        )

        # plot metrics


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    train(config)