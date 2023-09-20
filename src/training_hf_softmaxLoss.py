from datasets import Dataset
import pandas as pd
import os


# read the train and dev datasets
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
train_data = data[data['kfold'] != 1]
dev_data = data[data['kfold'] == 1]
meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']

