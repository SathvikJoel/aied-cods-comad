from sentence_transformers import SentenceTransformer
import os
modelPath = os.path.join('..', 'models', 'stsb-distilbert-base') 

model = SentenceTransformer('stsb-distilbert-base')
model.save(modelPath)
model = SentenceTransformer(modelPath)
print(model)