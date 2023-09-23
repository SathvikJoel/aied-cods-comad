from datasets import Dataset
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch

def get_transcript(video_name):
    val = meta_data.get(video_name, default=None)
    if val is None :
        return video_name
    else:
        return val 
    
# read the train and dev datasets
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
train_df = data[data['kfold'] != 1][['pre requisite', 'concept', 'label']]
dev_df = data[data['kfold'] == 1][['pre requisite', 'concept', 'label']]
meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']


train_dataset = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')
dev_dataset = Dataset.from_pandas(dev_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')

train_dataset = train_dataset.map(lambda x: {'pre_requisite' : get_transcript(x['pre_requisite']), 'concept' : get_transcript(x['concept'])})

dev_dataset = dev_dataset.map(lambda x: {'pre_requisite' : get_transcript(x['pre_requisite']), 'concept': get_transcript(x['concept'])})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

full_datasets = [train_dataset, dev_dataset]
transformed_datasets = []

for dataset in full_datasets:
    for part in ['pre_requisite', 'concept']:
        dataset = dataset.map(
            lambda x : tokenizer(
                x[part], max_length = 128, padding = 'max_length',truncation = True
            ), batched=True
        )
        for col in ['input_ids', 'attention_mask']:
            dataset = dataset.rename_column(col, part+ '_' + col)
    transformed_datasets.append(dataset)
        
train_dataset, dev_dataset = set(transformed_datasets)

train_dataset.set_format(type='torch', columns=train_dataset.column_names)
dev_dataset.set_format(type='torch', columns=train_dataset.column_names)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle = True
)

dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size, shuffle = True
)

model = BertModel.from_pretrained('bert-base-uncased')

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

from transformers.optimization import get_linear_schedule_with_warmup

optim = torch.optim.Adam(model.parameters(), lr=2e-5)
# and setup a warmup for the first ~10% steps
total_steps = int(len(train_dataset) / batch_size)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
		optim, num_warmup_steps=warmup_steps,
  	num_training_steps=total_steps - warmup_steps
)

from tqdm.auto import auto
num_epoch = 1
for epoch in range(num_epoch):
    #train
    model.train()
    


    #eval
    model.eval()