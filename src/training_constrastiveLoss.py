from sentence_transformers import SentenceTransformer, losses, util, evaluation
import pandas as pd
import os
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader


model = SentenceTransformer('stsb-distilbert-base')
num_epochs = 10
train_batch_size = 64
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
output_path = os.path.join('..', 'models')
margin = 0.5

train_samples = []

################## LOAD AND SLICE DATA #####################
data = pd.read_csv(os.path.join('..', 'input', 'train_fold.csv'))
train_data = data[data['kfold'] == 1]
dev_data = data[data['kfold'] != 1]



for row in train_data:
    sample = InputExample(texts = [row['pre requisite'], row['concept']], label=int(row['label']))
    train_samples.append(sample)


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size = train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model = model, distance_metric=distance_metric)

################ EVAL ###################
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
for row in dev_data:
    dev_sentences1.append(row['pre requisite'])
    dev_sentences2.append(row['concept'])
    dev_labels.append(int(row['label']))

binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=binary_acc_evaluator,
    epochs = num_epochs,
    warmup_steps=100,
    output_path=output_path
)

