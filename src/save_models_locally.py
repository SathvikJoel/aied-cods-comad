from sentence_transformers import SentenceTransformer
import os
from transformers import BertTokenizer, BertModel

modelPath = os.path.join('..', 'models', 'bert-base-uncased')

model_name = 'bert-base-uncased'

# Step 2: Instantiate the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Step 3: Save the tokenizer and model to the specified directory
tokenizer.save_pretrained(modelPath)
model.save_pretrained(modelPath)



# modelPath = os.path.join('..', 'models', 'stsb-distilbert-base') 

# model = SentenceTransformer('stsb-distilbert-base')
# model.save(modelPath)
# model = SentenceTransformer(modelPath)
# print(model)