# using bert-model to create static the word-embedding
# https://datascience.stackexchange.com/questions/85566/how-pre-trained-bert-model-generates-word-embeddings-for-out-of-vocabulary-words

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
