from datasets import Dataset
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
from transformers.optimization import get_linear_schedule_with_warmup

print("Starting script training_hf_spftmaxLoss.py")
print("*************** Script Started *****************\n")
def get_transcript(video_name):
    val = meta_data.get(video_name, default=None)
    if val is None :
        return video_name
    else:
        return val 
    
# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# read the train and dev datasets
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
train_df = data[data['kfold'] != 1][['pre requisite', 'concept', 'label']]
dev_df = data[data['kfold'] == 1][['pre requisite', 'concept', 'label']]
meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']


train_dataset = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')
dev_dataset = Dataset.from_pandas(dev_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')

train_dataset = train_dataset.map(lambda x: {'pre_requisite' : get_transcript(x['pre_requisite']), 'concept' : get_transcript(x['concept'])})

dev_dataset = dev_dataset.map(lambda x: {'pre_requisite' : get_transcript(x['pre_requisite']), 'concept': get_transcript(x['concept'])})


# load the tokenizer and model
modelPath = os.path.join('..', 'models', 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(modelPath)
model = BertModel.from_pretrained(modelPath).to(device)

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

ffnn = torch.nn.Linear(768 * 3, 1)
ffnn.to(device)

loss_func = torch.nn.CrossEntropyLoss()


optim = torch.optim.Adam(model.parameters(), lr=2e-5)
# and setup a warmup for the first ~10% steps
total_steps = int(len(train_dataset) / batch_size)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
		optim, num_warmup_steps=warmup_steps,
  	num_training_steps=total_steps - warmup_steps
)

num_epoch = 1
for epoch in range(num_epoch):
    #train
    model.train()
    for batch in train_loader:
        optim.zero_grad()
        input_ids_a = batch['pre_requisite_input_ids'].to(device)
        input_ids_b = batch['concept_input_ids'].to(device)
        attention_a = batch['pre_requisite_attention_mask'].to(device)
        attention_b = batch['concept_attention_mask'].to(device)
        label = batch['label'].to(device)
        embed_a = model(input_ids_a, attention_a)[0]
        embed_b = model(input_ids_b, attention_b)[0]
        u = mean_pool(embed_a, attention_a)
        v = mean_pool(embed_b, attention_b)

        # build the |u-v| tensor
        uv = torch.abs(torch.abs(u - v))
        # concatenate the [u, v, |u-v|] tensors
        x = torch.cat([u, v, uv], -1)
        # pass through the final linear layer
        x = ffnn(x)
        # compute the loss
        loss = loss_func(x, label)
        loss.backward()
        optim.step()
        scheduler.step()
        print(f'epoch : {epoch}, loss : {loss.item():.4f}')
        # make predictions
        preds = torch.argmax(x, dim=1)
        # compute the accuracy
        acc = (preds == label).float().mean()
        print(f'epoch : {epoch}, acc : {acc.item():.4f}')
    #eval
    model.eval()
    for batch in dev_loader:
        input_ids_a = batch['pre_requisite_input_ids'].to(device)
        input_ids_b = batch['concept_input_ids'].to(device)
        attention_a = batch['pre_requisite_attention_mask'].to(device)
        attention_b = batch['concept_attention_mask'].to(device)
        label = batch['label'].to(device)
        embed_a = model(input_ids_a, attention_a)[0]
        embed_b = model(input_ids_b, attention_b)[0]
        u = mean_pool(embed_a, attention_a)
        v = mean_pool(embed_b, attention_b)

        # build the |u-v| tensor
        uv = torch.abs(torch.abs(u - v))
        # concatenate the [u, v, |u-v|] tensors
        x = torch.cat([u, v, uv], -1)
        # pass through the final linear layer
        x = ffnn(x)
        # compute the loss
        loss = loss_func(x, label)
        print(f'epoch : {epoch}, loss : {loss.item():.4f}')
        # make predictions
        preds = torch.argmax(x, dim=1)
        # compute the accuracy
        acc = (preds == label).float().mean()
        print(f'epoch : {epoch}, acc : {acc.item():.4f}')

# save the model usig save_pretrained
model.save_pretrained('bert_ffnn')
