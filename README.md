# olf_chemical_embs
This is a repo for **L**oRA-based **O**dorant-**R**eceptor **A**ffinity prediction with **CROSS**-attention (**LORAX**) from [this](https://openreview.net/forum?id=BUUfUcIcfE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) paper.

![alt text](intro-fig.png)

## Setup

To set up the environment run

```
git clone https://github.com/GrantMcConachie/olf_chemical_embs.git
cd olf_chemical_embs
pip install -r requirements.txt
```

The data used in the paper is located [here](https://limewire.com/d/XoELD#31hJF5pEIE). Download and put into a `data/` folder in the parent directory. The `BindingDB` folder in the zenodo is not necessary for this repo.

## Training LORAX

There are two training scripts. `scripts/train_lorax.py` and `scripts/train_GB.py`. `train_lorax.py` will train the low rank adapted multimodal transformer and `train_GB.py` will train the gradient boosted descision tree (XGBoost) ensemble using the saved transformer representation from `train_lorax.py`. The hyperparameters of both the multimodal transformer and the XGBoost ensemble can be changed in the `config/` config files. The config files have a general structure like this

```
model:
    smi_model_card: "DeepChem/ChemBERTa-77M-MTR"
    prot_model_card: "facebook/esm2_t33_650M_UR50D"
    combine:
        mlp_hidden_dim: 512
        full_smiles_sequence: True
        smiles_hidden_dim: 256
        num_heads: 8
        comb_dropout: 0.1
        no_cross_attn: True
        no_prot_model_ft: True
        lin_proj: True

    lora_module:
        inference_mode: False
        r: 8
        lora_alpha: 8
        bias: "none"
        use_rslora: False
        modules_to_save: ["pooler.dense.bias", "pooler.dense.weight"]
        target_modules: ["query", "key", "value"]
        lora_dropout: 0.1

training:
    data_path: "data/CC/rand_splits"
    results_path: "results/CC/lin_proj" # NOTE: change for different save folder
    log_path: "logs/CC/lin_proj" # NOTE: change for different save folder

train_lorax:
    batch_size: 12
    lr: 0.001  # NOTE
    train_epochs: 50

train_GB:
    max_evals: 500
    use_lorax_embs: False
```

Once you have set up your config file, or use one of the default ones in the `config/` directory, you can train the multimodal transformer using

```
python scripts/train_lorax.py --config path/to/config.yaml
```

and the XGBoost ensemble after that using

```
python scripts/train_GB.py --config path/to/config.yaml
```

## Output

`train_lorax.py` will create both a log path and a results path dictated by the config file. The results path will populate with the trained weights of the multimodal transformer and the logs path will populate with a tensorboard file that you can monitor using

```
tensorboard --logdir /path/to/log
```

`train_GB.py` will use these same paths to create logs and save XGBoost model weights and ensemble proportions. The results path also saves the final predictions as a csv file. 

## Huggingface models

To keep all your Huggingface models in one spot, I recommend using

```
export HF_HOME=./saved_models
```

This dictates where the pretrained foundation model weights will be saved. This step is not necessary, but may be helpful.
