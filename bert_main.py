# using bert-model to create static the word-embedding
# https://datascience.stackexchange.com/questions/85566/how-pre-trained-bert-model-generates-word-embeddings-for-out-of-vocabulary-words

from transformers import BertTokenizer, BertModel
from transformers import BertTokenizerFast
import torch
import os
import numpy as np
import random
from tqdm import tqdm

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

#----------------------------implemented bert-embedding-----------------------
from src.bert_preparing import read_raw_documents, simple_preprocess, handle_long_sentences, create_marked_senteces, save_sents_to_txt
from src.bert_preparing import transform_to_sentences
from src.bert_embedding import reform_token_embeddings_of_sentence, get_token_embeddings, get_final_words_embeddings_in_sent
from src.bert_embedding import tokenizerfast_for_a_sent, vocabulary_embeddings_to_text

#----------------------------BERT-Models
tokenizerfast = BertTokenizerFast.from_pretrained('bert-base-uncased')
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
print(f'total sentences: {len(sentences)}')
shorted_sentences =  handle_long_sentences(sentences, 128, 10)
print(f'total shorted sentences: {len(shorted_sentences)}')
marked_shorted_sentences = create_marked_senteces(shorted_sentences)
save_sents_to_txt(marked_shorted_sentences)
#-------------------------------Creating Vocab-Embeddings-----------------------------------------------


vocab = {}

for marked_sent in tqdm(marked_shorted_sentences, desc="creating bert embeddings"):
    #print(marked_sent)
    tokens_tensor, segments_tensors, tokens_ids_with_belonging_information = tokenizerfast_for_a_sent(marked_sent, tokenizerfast)
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        reformed = reform_token_embeddings_of_sentence(outputs)
        sent_tokens_embeddings = get_token_embeddings(reformed)
        #print(f'number of found embeddings: {len(sent_tokens_embeddings)}')
        words_embeddings_in_sent_dict = get_final_words_embeddings_in_sent(marked_sent, tokens_ids_with_belonging_information, sent_tokens_embeddings)
        #save_embeddings_to_text(words_embeddings_in_sent_dict)
        for word, vector in words_embeddings_in_sent_dict.items():
            #print(word)
            if word in vocab.keys():
                #print(vector[:2])
                sum_vector = vocab[word][1] + vector
                #print(sum_vector)
                count = vocab[word][0] + 1
                vocab[word] = (count, sum_vector)
            else:
                #print(vector[:2])
                vocab[word] = (1, vector)

        del tokens_tensor
        del segments_tensors
        del outputs
        del reformed
        del sent_tokens_embeddings
        del words_embeddings_in_sent_dict
    #print("---------------------------------------------------------------------------------------")

#update vocab over all sentences
updated_vocab = {}
for word, (count, sum_vector) in vocab.items():
    updated_vocab[word] = (sum_vector/count)
vocabulary_embeddings_to_text(updated_vocab)

