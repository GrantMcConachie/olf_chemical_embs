"""
Trains a gradient boosted descision tree on top of lorax
"""

import yaml

import xgboost as xgb

from scripts.train_lorax import get_dataloaders


def train(config):
    """
    training function for the GB descision tree
    """
    # load in trained model

    # generate representations to put into xgboost

    # train xgboost

    # plot metrics


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    train(config)