import os
import pandas as pd

df = pd.read_csv('data/CC/raw/CC_reformat_z.csv')
receptor_names = df['RECEPTOR'].unique()

for name in receptor_names:
    save_path = f'data/CC/LORO/{name}'
    os.makedirs(save_path, exist_ok=True)

    train = df[df['RECEPTOR'] != name]
    val = df[df['RECEPTOR'] == name]

    train.to_csv(os.path.join(save_path, 'train_df.csv'), index=False)
    val.to_csv(os.path.join(save_path, 'val_df.csv'), index=False)
    val.to_csv(os.path.join(save_path, 'test_df.csv'), index=False)