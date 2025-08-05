"""
Script to train the lora model

export HF_HOME=/projectnb/depaqlab/Grant/lora/saved_models
"""

import yaml
from tqdm import tqdm
from sklearn.metrics import r2_score

import torch
from transformers import AutoModel
from torch.utils.data import DataLoader

from model.model import ProteinSmilesLoraModel
from data_utils.prot_smi_dataset import ProteinSmilesDataset


def evaluate(model, val_dataloader, device, loss_fn, epoch):
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
            pred = model(smi_token, prot_token).squeeze()
            
            # calculate loss and backpropogate
            loss = loss_fn(pred, y)

            # bookeeping
            preds.append(pred.squeeze())
            ground_truth.append(y)
            running_loss.append(loss.item())
        
        val_r2 = r2_score(torch.concat(ground_truth).cpu().numpy(), torch.concat(preds).cpu().numpy())
        avg_val_loss = sum(running_loss) / len(running_loss)
        print(f'Epoch {epoch} | Avg Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f}')

    return avg_val_loss


def train(
        train_df,
        val_df,
        smi_model_card,
        prot_model_card,
):
    """
    Main training loop for the model
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load config
    config = yaml.safe_load(open('model/config.yaml', 'r'))

    # load models
    smi_model = AutoModel.from_pretrained(smi_model_card).to(device)
    prot_model = AutoModel.from_pretrained(prot_model_card).to(device)

    # dataloader
    train_data = ProteinSmilesDataset(
        train_df,
        smi_model,
        prot_model,
        smi_model_card,
        prot_model_card
    )
    val_data = ProteinSmilesDataset(
        val_df,
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
    tot_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    model.train()
    for epoch in tqdm(range(config['training']['train_epochs']), desc='epoch'):
        running_loss = []

        for i in tqdm(train_dataloader, desc='batch'):
            # zero gradients
            optim.zero_grad()

            # unpack data
            smi_token, prot_token, y = i
            smi_token = {k: v.to(device) for k, v in smi_token.items()}
            prot_token = {k: v.to(device) for k, v in prot_token.items()}
            y = y.to(device)

            # pass through model
            pred = model(smi_token, prot_token).squeeze()
            
            # calculate loss and backpropogate
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()

            # bookeeping
            running_loss.append(loss.item())

        # evaluate
        avg_val_loss = evaluate(model, val_dataloader, device, loss_fn, epoch)
        # TODO: save model when the best validation loss happens
        model.train()

        avg_loss = sum(running_loss) / len(running_loss)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    # init parameters
    train_df= '/projectnb/depaqlab/Grant/lora/data/HC/rand_splits/rand_split_1/train_df.csv'
    val_df = '/projectnb/depaqlab/Grant/lora/data/HC/rand_splits/rand_split_1/val_df.csv'

    # these probably should go in config?
    smi_model_card = 'DeepChem/ChemBERTa-77M-MTR'
    prot_model_card = 'facebook/esm2_t33_650M_UR50D'

    train(
        train_df=train_df,
        val_df=val_df,
        smi_model_card=smi_model_card,
        prot_model_card=prot_model_card
    )
