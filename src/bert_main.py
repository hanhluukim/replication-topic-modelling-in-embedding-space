# using bert-model to create static the word-embedding
# https://datascience.stackexchange.com/questions/85566/how-pre-trained-bert-model-generates-word-embeddings-for-out-of-vocabulary-words

import torch
from transformers import BertTokenizer, BertModel
from sklearn.datasets import fetch_20newsgroups
import re
from nltk.tokenize import word_tokenize
import nltk
import string
nltk.download('punkt')


from src.bert_preparing import read_raw_documents, simple_preprocess, handle_long_sentences, create_marked_senteces, save_sents_to_txt






tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()


print("reading data:...")
raw_documents, _ = read_raw_documents()
print(len(raw_documents))
print("preprocess data:...")
preprocessed_docs = simple_preprocess(raw_documents)
print(len(preprocessed_docs))
print("transform to sentences:...")
sentences = transform_to_sentences(preprocessed_docs)
print("split sentences to 128 tokens:...")
shorted_sentences =  handle_long_sentences(sentences, 128)
shorted_sentences = create_marked_senteces(shorted_sentences)

