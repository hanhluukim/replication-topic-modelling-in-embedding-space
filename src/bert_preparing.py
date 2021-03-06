from sklearn.datasets import fetch_20newsgroups
import re
from nltk.tokenize import word_tokenize
import nltk
import string
nltk.download('punkt')


with open('src/stops.txt', 'r') as f:
    stop_words = f.read().split('\n')
    
def read_raw_documents():
    """
    newsgroups_train = fetch_20newsgroups(subset='train')
    raw_documents = []
    raw_labels = []
    for i in range(0,len(newsgroups_train.data)):
        raw_documents.append(newsgroups_train.data[i])
        raw_labels.append(newsgroups_train.target[i])
    """
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    def filter_special_character(docs):
        # remove ., needed for bert
        filter_patter = r'''[\w']+|[!?.,;-~{}`´_<=>:/@*()&'$%#"]'''
        return [re.findall(filter_patter, docs[doc_idx]) for doc_idx in range(len(docs))]
    init_docs_tr = filter_special_character(train_data.data)
    init_docs_ts = filter_special_character(test_data.data)
    complete_documents = init_docs_tr + init_docs_ts
    raw_documents = []
    for doc in complete_documents:
        raw_documents.append(" ".join(doc))
    return raw_documents, None

def simple_preprocess(raw_documents):
    
    def contains_numeric(w):
        return any(char.isdigit() for char in w)
    
    def only_letters(tested_string):
        for letter in tested_string:
            if letter not in "abcdefghijklmnopqrstuvwxyz":
                return False
        return True
    
    def clean_doc_for_bert(doc): 
        #word_list = doc.split(" ") #doc.replace(">","").lower()
        word_list = word_tokenize(doc.lower()) #only using empty space and punctation for tokenization
        cleaned = []
        for w in word_list:
            #if w not in stop_words:
              #if w in string.punctuation or only_letters(w): #using only character from punctation and alpha characters
            if not contains_numeric(w):
                if w in ['.', ','] or only_letters(w):
                    if w in string.punctuation or len( set(w) ) > 1: #punctation with len 1 allowed but alpha word must be longer then 1
                        cleaned.append(w)
        return " ".join(cleaned), cleaned  #save doc in string and in token-list         
       
    cleaned_documents = []
    for doc in raw_documents:
        doc_in_string, doc_in_token_list = clean_doc_for_bert(doc)
        cleaned_documents.append(doc_in_string)
    return cleaned_documents

def simple_preprocess_old(raw_documents):
    
    def contains_numeric(w):
        return any(char.isdigit() for char in w)
    
    def only_letters(tested_string):
        for letter in tested_string:
            if letter not in "abcdefghijklmnopqrstuvwxyz":
                return False
        return True
    
    def clean_doc_for_bert(doc): 
        #word_list = doc.split(" ") #doc.replace(">","").lower()
        word_list = word_tokenize(doc.lower()) #only using empty space and punctation for tokenization
        cleaned = []
        for w in word_list:
            if w not in stop_words:
                #if w in string.punctuation or only_letters(w): #using only character from punctation and alpha characters
                if not contains_numeric(w):
                    if w in string.punctuation or only_letters(w):
                        if w in string.punctuation or len( set(w) ) > 1: #punctation with len 1 allowed but alpha word must be longer then 1
                            cleaned.append( w)
        return " ".join(cleaned), cleaned  #save doc in string and in token-list         
       
    cleaned_documents = []
    for doc in raw_documents:
        doc_in_string, doc_in_token_list = clean_doc_for_bert(doc)
        cleaned_documents.append(doc_in_string)
    return cleaned_documents

def transform_to_sentences_with_labels():
    # we will not use labels
    sentences_with_labels = []
    return sentences_with_labels

def fine_tune_bert():
    # should to be trained?
    # no, because for topic modelling, that is usupervised problem. We just find topics for the documents
    # topic modelling no targets
    return True

def transform_to_sentences(docs): #no labels
    data_as_sentences = []
    for doc in docs:
      for sent in doc.split("."): #make sentences
        #print(f'after split: {sent}')
        updated_sent = " ".join([t for t in sent.strip().split(" ") if len(t) > 1])
        #print(f'update: {updated_sent}')
        if len(updated_sent.split(" ")) > 1:
            data_as_sentences.append(updated_sent)
        else:
            if updated_sent not in data_as_sentences:
                data_as_sentences.append(updated_sent)
    return data_as_sentences

def transform_to_sentences_old(docs): #no labels
    data_as_sentences = []
    for doc in docs:
      for sent in doc.split("."): #make sentences
        updated_sent = " ".join([t for t in sent.strip().split(" ") if len(t) > 1])
        if len(updated_sent.split(" ")) > 1:
            data_as_sentences.append(updated_sent)
        else:
            if updated_sent not in data_as_sentences:
                data_as_sentences.append(updated_sent)
    return data_as_sentences

def split_long_sentence(splitted_sent, given_len, window_size):
    subsents = []
    #for i in range(0,len(splitted_sent), given_len):
    i=0
    while i <= len(splitted_sent): 
        if i == 0:
            sub = " ".join(splitted_sent[i:i+given_len])
            subsents.append(sub)
            i = i + given_len
        else:
            j = i - window_size #windown 5
            if j + given_len <= len(splitted_sent):
                sub = " ".join(splitted_sent[j:j + given_len])
                subsents.append(sub)
            else:
                sub = " ".join(splitted_sent[j:])
                if len(sub)>1:
                    subsents.append(sub)
            i = j + given_len
    return subsents

def split_long_sentence_false(splitted_sent, given_len):
    subsents = []
    #for i in range(0,len(splitted_sent), given_len):
    i=0
    while i <= len(splitted_sent): 
        if i == 0:
            sub = " ".join(splitted_sent[i:i+given_len])
            subsents.append(sub)
            i = i + given_len
        if i!=0:
            j = i + given_len - 10 #windown 5
            if j + given_len <= len(splitted_sent):
                sub = " ".join(splitted_sent[j:j + given_len])
                subsents.append(sub)
            else:
                sub = " ".join(splitted_sent[j:])
                if len(sub)>1:
                    subsents.append(sub)
            i = j + given_len
    return subsents

def handle_long_sentences(sentences, given_len, window_size):
    # overlapped splitting sentence windown 5
    subsents = []
    deleted_long_sents = []
    for sent in sentences:
        splitted_sent = sent.split(" ")
        if len(splitted_sent) > given_len:
          long_sent_subsents = split_long_sentence(splitted_sent, given_len, window_size)
          subsents.extend(long_sent_subsents)
          deleted_long_sents.append(sent)
    # update sentences: remove and add subsents
    for del_sent in deleted_long_sents:
        sentences.remove(del_sent)
    for add_sent in subsents:
        sentences.append(add_sent)
    return sentences

def create_marked_senteces(sentences):
    return ['[CLS] ' + sent + ' [SEP]' for sent in sentences]
def save_sents_to_txt(shorted_sentences):
    with open(r'prepared_data/bert_sentences.txt', 'w') as fp:
      for i, sent in enumerate(shorted_sentences):
          # write each item on a new line
          fp.write(f'{i+1}: \t {sent}\n')
      print('saving sentences from bert-processing')
    return True
