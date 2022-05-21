# using bert-model to create static the word-embedding
# https://datascience.stackexchange.com/questions/85566/how-pre-trained-bert-model-generates-word-embeddings-for-out-of-vocabulary-words

from transformers import BertTokenizer, BertModel
from src.bert_preparing import read_raw_documents, simple_preprocess, handle_long_sentences, create_marked_senteces, save_sents_to_txt
from src.bert_preparing import transform_to_sentences

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()

#-------------------------------Data Preparing-------------------------------------------
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
marked_shorted_sentences = create_marked_senteces(shorted_sentences)
#-------------------------------Get Modell-----------------------------------------------


