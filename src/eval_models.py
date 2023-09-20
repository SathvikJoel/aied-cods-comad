# script to evaluate models

from sklearn import metrics
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np


# Load the dev dataset
data = pd.read_csv('..', 'input', 'train_folds.csv')
dev_data = data[data['kfold'] == 1][['label', 'pre requisite', 'concept']]

meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']


def get_transcript(video_name):
    val = meta_data.get(video_name, default=None)
    if val is None :
        return video_name
    else:
        return val 

model_path = os.path.join('..', 'models', 'first_model')

model = SentenceTransformer(model_path)

vf = np.vectorize(lambda x : get_transcript(x))

pre_req_sents = vf(dev_data['pre requisite'].values)

concept_sents = vf(dev_data['pre requsite'].values)

pre_req_embds = model.encode(pre_req_sents, show_progress_bar=True)

concept_embds = model.encode(concept_sents, show_progress_bar=True)

similarity_matrix = util.dot_score(pre_req_embds, concept_embds)

print(similarity_matrix)

print(similarity_matrix.diag())



