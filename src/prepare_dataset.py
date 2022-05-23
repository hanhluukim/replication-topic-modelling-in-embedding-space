# using dataset 20newsgroups from sklearn
# using other dataset from other source
# texts will be loaded and preprocessed follow the user's options
# the retured data here will be imported to the word-embedding modul to create word-embedding. After that can ETM will be used
# returned data can be used as input for LDA Model

from sklearn.datasets import fetch_20newsgroups
import pathlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
import gensim
import pandas as pd
import re
import string
import random
from scipy.io import savemat
from pathlib import Path
import os
import numpy as np
import random
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

random.seed(42)

with open('src/stops.txt', 'r') as f:
    # list stopwords from the original paper
    stops = f.read().split('\n')
    
class TextDataLoader:
    def __init__(self, source="20newsgroups", train_size=None, test_size=None):
        self.source = source
        self.train_size = None
        self.test_size = None
        # if 20newsgroups, get train-size, test-size directly from fetch_20newsgroups(subset="train/test")
        self.complete_docs = []
        #self.for_lda_model = for_lda_model
        if train_size!=None:
            #self.train_size = int(train_size * len(self.complete_docs))
            #self.test_size = int(test_size * len(self.complete_docs)) 
            self.train_size = train_size
            self.test_size = test_size
        # these values indices will be updated in the split function
        self.train_indices = None
        self.test_indices = None
        self.val_indices = None
    
        self.min_df = 0
        self.n_tokens_train = 0
        self.n_tokens_test = 0
        self.n_tokens_val = 0
        
    def load_tokenize_texts(self, source="20newsgroups"):
        print("loading texts: ...")
        if source == "20newsgroups":
            # download data from package sklearn
            train_data = fetch_20newsgroups(subset='train')#[:100]
            test_data = fetch_20newsgroups(subset='test')#[:50]
            # filter special character from texts
            def filter_special_character(docs):
                filter_patter = r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]'''
                return [re.findall(filter_patter, docs[doc_idx]) for doc_idx in range(len(docs))]
            init_docs_tr = filter_special_character(train_data.data)
            init_docs_ts = filter_special_character(test_data.data)
            #[re.findall(filter_patter, test_data.data[doc]) for doc in range(len(test_data.data[:50]))]
            self.complete_docs = init_docs_tr + init_docs_ts
            self.train_size = len(init_docs_tr) #round(len(init_docs_tr)/len(self.complete_docs),1)
            print(f'train-size after loading: {self.train_size}')
            self.test_size = len(init_docs_ts) #round(len(init_docs_ts)/len(self.complete_docs),1)
            print(f'test-size after loading: {self.test_size}')
            
            del train_data
            del test_data
            del init_docs_tr
            del init_docs_ts
        else: 
            # load data from url or other sources        
            print("load data from ..." + str(source))
            with open(pathlib.Path(source)) as f:
                self.complete_docs = f.readlines()
        print("finished load!")
            
    def preprocess_texts(self, 
                         length_one_remove=True, 
                         punctuation_lower = True, 
                         stopwords_filter = True):
        
        def contains_punctuation(w):
            return any(char in string.punctuation for char in w)
        def contains_numeric(w):
            return any(char.isdigit() for char in w)
        print("start: preprocessing: ...")
        if punctuation_lower:
            self.complete_docs = [[w.lower() for w in self.complete_docs[doc] if not contains_punctuation(w)] for doc in range(len(self.complete_docs))]
            self.complete_docs = [[w for w in self.complete_docs[doc] if not contains_numeric(w)] for doc in range(len(self.complete_docs))]
        if length_one_remove:
            self.complete_docs = [[w for w in self.complete_docs[doc] if len(w)>1] for doc in range(len(self.complete_docs))]
        if stopwords_filter:
            self.complete_docs = [[w for w in self.complete_docs[doc] if w not in stops] for doc in range(len(self.complete_docs))]
        
        self.complete_docs = [" ".join(self.complete_docs[doc]) for doc in range(len(self.complete_docs))]   
        #return True
        print("finised: preprocessing!")
    
    def show_example_raw_texts(self, n_docs=2):
        print("check some sample texts of the dataset after filter punctuation and digits")
        try:
            for i in range(0,n_docs):
                print(self.complete_docs[i])
                print(100*"=")
        except:
            print("complete raw texts was be deleted for memory problem")
            
    def split_and_create_voca_from_trainset(self,max_df=0.7, min_df=2, stopwords_remove_from_voca = True):
        """
        This block is used for creating the init-word2id and init-id2word and the vocabulary.
        The Vocabulary and Dictionaries will be later updated by only train-dataset and stopwords.
        Sort the list of words in the vocabulary by the (word-df)
        """
        # --------------
        self.min_df = min_df
        # --------------
        # filter by max-df and min-df with CountVectorizer
        # CountVectorizer create Vocabulary (Word-Ids). For each doc: Word-Frequency of each word of this document
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None) # stopwords will be used later
        vectorized_documents = vectorizer.fit_transform(self.complete_docs)
        #signed_documents = vectorized_documents.sign()
        # document-frequency for each word in vocabulary
        sum_docs_counts = vectorized_documents.sign().sum(axis=0) 
        print("test-document-frequency: ")
        print(sum_docs_counts)
        
        # init word2id and id2word with vectorizer
        # sort the init-voca-dictionary by documents-frequency
        # saving the frequency of each word in Vocabulary over all documents/ look "sum in the column"
        # number of documents, which containt the word in vectorizer.vocabulary
        self.word2id = {}
        self.id2word = {}
        for w in vectorizer.vocabulary_:
            self.word2id[w] = vectorizer.vocabulary_.get(w)
            self.id2word[vectorizer.vocabulary_.get(w)] = w
        
        # create init-vocabulary from words, that sorted-vocabular ordered by doc-frequencies
        voca_size = sum_docs_counts.shape[1]
        print(f'vocab-size in df: {voca_size}')
        sum_counts_np = np.zeros(voca_size, dtype=int)
        for v in range(voca_size):
            sum_counts_np[v] = sum_docs_counts[0, v]
        idx_sort = np.argsort(sum_counts_np)
        vocabulary = [self.id2word[idx_sort[cc]] for cc in range(voca_size)]
       
        # filter the stopwords from vocabulary and update the dictionary: word2id and id2word 
        if stopwords_remove_from_voca:
            vocabulary = [w for w in vocabulary if w not in stops]
        #self.documents_without_stop_words = [[word for word in document.split() if word not in stops] for document in self.complete_docs]
        self.word2id = {}
        self.id2word = {} 
        for j, w in enumerate(vocabulary): 
            self.word2id[w] = j
            self.id2word[j] = w 
        
        # vocabulary from train-dataset, update word2id and id2word again
        # TODO: make the train_set, test_set, val_set detemetistic
        #train_dataset_size = self.train_size - val_dataset_size
        # get dataset-docs-indices for each set
        val_dataset_size = 100
        docs_tr, docs_ts, docs_tr_indices, docs_ts_indices = train_test_split(self.complete_docs, range(0,len(self.complete_docs)), test_size=self.test_size, random_state=42)
        print(f'validation-size ist: {round(val_dataset_size/len(docs_tr),2)}')
        docs_tr, docs_va, docs_tr_indices, docs_va_indices = train_test_split(docs_tr, docs_tr_indices, test_size=100, random_state=42)
        
        self.train_indices = docs_tr_indices
        self.test_indices = docs_ts_indices
        self.val_indices = docs_va_indices

        del docs_tr
        del docs_ts
        del docs_va
        del docs_tr_indices
        del docs_va_indices
        del docs_ts_indices

        #self.idx_permute = np.random.permutation(self.train_size).astype(int)
        #print(f'permuted indices for the train set: {self.idx_permute[:15]}')

        # only words from train dataset will be maintained in the global vocabulary
        # update word2id and id2word again
        print("start creating vocabulary ...")
        vocabulary = []
        for train_idx in self.train_indices:
            for w in self.complete_docs[train_idx].split():
                if w in self.word2id: #it means that still not-stopwords will be saved, because word2id is before updated by not-stopwords-vocabulary
                    vocabulary.append(w)
                    
        self.vocabulary = list(set(vocabulary))
        #self.vocabulary = list(set(vocabulary))
        print(f'length of the vocabulary: {len(self.vocabulary)}')
        print(f'sample ten words of the vocabulary: {self.vocabulary[:10]}')

        self.word2id = {}
        self.id2word = {} 
        for j, w in enumerate(self.vocabulary): 
            #print(j,w)
            self.word2id[w] = j
            self.id2word[j] = w
        #del vocabulary #delete the old voca
        print(f'length word2id list: {len(self.word2id.keys())}')
        print(f'length id2word list: {len(self.id2word.keys())}')
        print("finished: creating vocabulary")
    
    def get_docs_in_word_ids_for_each_set(self):
        #using the self.train_indices, self.test_indices, self.val_indices to get the documents
        """REMOVE_SAVING
        with open('prepared_data/train_docs_from_complete_docs.txt', 'w') as f:
            #print(self.train_indices)
            sorted_indices = sorted(self.train_indices)
            for train_doc_idx in sorted_indices:
                f.write(f'{train_doc_idx}: {self.complete_docs[train_doc_idx]}')
                f.write("\n")
        """
        docs_tr = [[self.word2id[w] for w in self.complete_docs[train_doc_idx].split() if w in self.word2id] for train_doc_idx in self.train_indices]
        docs_va = [[self.word2id[w] for w in self.complete_docs[val_doc_idx].split() if w in self.word2id] for val_doc_idx in self.val_indices]
        docs_ts = [[self.word2id[w] for w in self.complete_docs[test_doc_idx].split() if w in self.word2id] for test_doc_idx in self.test_indices]
        
        return docs_tr, docs_va, docs_ts
    
    def get_docs_in_words_for_each_set(self):
        # recreate documents from documents in word-ids
        # for embedding processing
        docs_tr, docs_va, docs_ts = self.get_docs_in_word_ids_for_each_set()
        def doc_in_words(doc):
          doc = [self.id2word[wid] for wid in doc]
          return doc
        docs_tr = [doc_in_words(doc) for doc in docs_tr]
        docs_va = [doc_in_words(doc) for doc in docs_va]
        docs_ts = [doc_in_words(doc) for doc in docs_ts]
        # todo: saving the train-documents in words in a file

        # save preprocessed documents to file to use with julia later

        def save_preprocessed_docs(path=None, name=None, docs=None, docs_indices = None):
          docs_df = pd.DataFrame()
          docs_df['index-in-complete-docs'] = docs_indices #indices_in complete-doc
          docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs]
          docs_df.to_csv(f'{path}/{name}.csv',index=False)
          del docs_df
          return True
        """REMOVE_SAVING
        save_path = 'prepared_data/min_df_'+str(self.min_df)
        save_preprocessed_docs(path = save_path, name="preprocessed_docs_train", docs = docs_tr, docs_indices= self.train_indices)
        save_preprocessed_docs(path = save_path, name="preprocessed_docs_test", docs = docs_ts, docs_indices= self.test_indices)
        save_preprocessed_docs(path = save_path, name="preprocessed_docs_val", docs = docs_va, docs_indices= self.val_indices)
        """
        
        return docs_tr, docs_va, docs_ts

    def create_bow_and_savebow_for_each_set(self, for_lda_model = True, normalize = True):
        """
        docs will be saved unter the list of word-ids
        1. from the word2id and vocabulary, documents-set will be recreated
        2. remove the empty documents and documents with only one words
        3. the test-dataset will be splited two halves for some reason in the papers, read again
        """
        
        docs_tr, docs_va, docs_ts = self.get_docs_in_word_ids_for_each_set()
        train_sum_docs_counts = None #using as df to compute the normalized bow
        if normalize:
            """
            docs_tr_in_words = 
            train_vectorizer = CountVectorizer(min_df=None, max_df=None, stop_words=None) # stopwords will be used later
            train_vectorized_documents = train_vectorizer.fit_transform(docs_tr_in_words)
            train_signed_documents = train_vectorized_documents.sign()
            train_sum_docs_counts = train_signed_documents.sum(axis=0)[0]
            """
        
        def remove_empty(in_docs):
            return [doc for doc in in_docs if doc!=[]]
        
        docs_tr = remove_empty(docs_tr)
        docs_ts = remove_empty(docs_ts)
        docs_ts = [doc for doc in docs_ts if len(doc)>1]
        docs_va = remove_empty(docs_va)
        
        
        print(f'train-size-after-all: {len(docs_tr)}')
        print(f'test-size-after-all: {len(docs_ts)}')
        print(f'validation-size-after-all: {len(docs_va)}')
        
        
        # split test-set to two halves
        docs_ts_h1, docs_ts_h2,_,_ = train_test_split(docs_ts, self.test_indices, test_size=0.5, random_state=42)

        def create_list_words(in_docs):
            # just saving all word-ids in a one list
            return [x for y in in_docs for x in y]

        words_tr = create_list_words(docs_tr)
        words_ts = create_list_words(docs_ts)
        words_ts_h1 = create_list_words(docs_ts_h1)
        words_ts_h2 = create_list_words(docs_ts_h2)
        words_va = create_list_words(docs_va)
        self.n_tokens_train = len(words_tr)
        self.n_tokens_test = len(words_ts)
        self.n_tokens_val = len(words_va)
        
        def create_doc_indices(in_docs):
            # repeatly saving the doc-id for each word-id in the doc
            aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
            return [int(x) for y in aux for x in y]

        doc_indices_tr = create_doc_indices(docs_tr)
        doc_indices_ts = create_doc_indices(docs_ts)
        doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
        doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
        doc_indices_va = create_doc_indices(docs_va)
        
        n_docs_tr = len(docs_tr)
        n_docs_ts = len(docs_ts)
        n_docs_ts_h1 = len(docs_ts_h1)
        n_docs_ts_h2 = len(docs_ts_h2)
        n_docs_va = len(docs_va)
        
        del docs_tr
        del docs_ts
        del docs_ts_h1
        del docs_ts_h2
        del docs_va

        def create_bow(normalize, doc_indices, words, n_docs, vocab_size):
            print("start: creating bow representation...")
            """
            t = sorted(list(set(words)))
            print(f'top 10 - word-id of the doc: {t[:10]}')
            #print(doc_indices[:40])
            #print(words[:40])
            print(f'max word-id: {max(t)}')
            print(f'min word-id: {min(t)}')

            print(f'max doc-id: {max(doc_indices)}')
            print(f'min doc-id: {min(doc_indices)}')

            print(f'all docs: {len(doc_indices)}')
            print(f'all words: {len(words)}')

            print(f'docidx unique {len(set(doc_indices))}')
            print(f'words unique: {len(set(words))}')

            print(f'ndocs: {n_docs}')
            print(f'vocab-size: {vocab_size}')
            #print(len([1]*len(doc_indices)))
            """

            bow = sparse.coo_matrix(
              (
                [1]*len(doc_indices),(doc_indices, words)
                ), 
                shape=(n_docs, vocab_size)
                ).tocsr()
            print("finised creating bow input!\n")
            # return normalized bows / tfdif?
            #if normalize:
            #  print("need normalized bows")
            #  bow = bow
            return bow
        
        print(f'length train-documents-indices : {len(doc_indices_tr)}')
        print(f'length of the vocabulary: {len(self.vocabulary)}')
        print("\n")
        bow_tr = create_bow(normalize, doc_indices_tr, words_tr, n_docs_tr, len(self.vocabulary))
        bow_ts = create_bow(normalize, doc_indices_ts, words_ts, n_docs_ts, len(self.vocabulary))
        bow_ts_h1 = create_bow(normalize, doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(self.vocabulary))
        bow_ts_h2 = create_bow(normalize, doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(self.vocabulary))
        bow_va = create_bow(normalize, doc_indices_va, words_va, n_docs_va, len(self.vocabulary))
        
        del words_tr
        del words_ts
        del words_ts_h1
        del words_ts_h2
        del words_va
        del doc_indices_tr
        del doc_indices_ts
        del doc_indices_ts_h1
        del doc_indices_ts_h2
        del doc_indices_va
        #print("bow-train examples")
        #print(pd.DataFrame(bow_tr.toarray()))
        
        def split_bow(bow_in, num_docs):
            indices = [[w for w in bow_in[doc, :].indices] for doc in range(num_docs)]
            counts = [[c for c in bow_in[doc, :].data] for doc in range(num_docs)]
            return indices, counts

        def to_numpy_array(documents):
            return np.array([[np.array(doc) for doc in documents]],
                            dtype=object).squeeze()

        
        if for_lda_model == True:
            print("compact representation for LDA")
            def create_lda_corpus(bow_set):
                df = pd.DataFrame(bow_set.toarray())
                lda_corpus = []
                for i in range(0,df.shape[0]):
                    doc_corpus = [ (j,e) for j, e in enumerate(df.iloc[i])]
                    lda_corpus.append(doc_corpus)
                return lda_corpus
            train_dataset = gensim.matutils.Sparse2Corpus(bow_tr) #create_lda_corpus(bow_tr)
            test_dataset = gensim.matutils.Sparse2Corpus(bow_ts) #create_lda_corpus(bow_ts)
            val_dataset = gensim.matutils.Sparse2Corpus(bow_va) #create_lda_corpus(bow_va)
        
        else: #other models 
            bow_train_tokens, bow_train_counts = split_bow(bow_tr, n_docs_tr)
            bow_test_tokens, bow_test_counts = split_bow(bow_ts, n_docs_ts)
            bow_test_h1_tokens, bow_test_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
            bow_test_h2_tokens, bow_test_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
            bow_val_tokens, bow_val_counts = split_bow(bow_va, n_docs_va)
        
            train_dataset = {
                'tokens': to_numpy_array(bow_train_tokens),
                'counts': to_numpy_array(bow_train_counts),
            }
            
            val_dataset = {
                'tokens': to_numpy_array(bow_val_tokens),
                'counts': to_numpy_array(bow_val_counts),
            }

            test_dataset = {
                'test': {
                    'tokens': to_numpy_array(bow_test_tokens),
                    'counts': to_numpy_array(bow_test_counts),
                },
                'test1': {
                    'tokens': to_numpy_array(bow_test_h1_tokens),
                    'counts': to_numpy_array(bow_test_h1_counts),
                },
                'test2': {
                    'tokens': to_numpy_array(bow_test_h2_tokens),
                    'counts': to_numpy_array(bow_test_h2_counts),
                }
            }
            #---------------------REMOVE_SAVING--------------------
            """
            # saving to the prepared_data folder:
            bow_save_path = 'prepared_data/min_df_'+str(self.min_df)
            Path(bow_save_path).mkdir(parents=True, exist_ok=True)
            savemat(bow_save_path + '/bow_train.mat', {'train': train_dataset}, do_compression=True)
            savemat(bow_save_path + '/bow_test.mat', {'test': test_dataset}, do_compression=True)
            savemat(bow_save_path + '/bow_val.mat', {'validation': test_dataset}, do_compression=True)
            # saving id2word:
            print(f'some id2word befor saving: {list(self.id2word.keys())[:20]}')
            savemat(bow_save_path + '/id2word.mat', {'id': list(self.id2word.keys()), 'word': list(self.id2word.values())}, do_compression=True)
            """
            # saving id2word to csv test
            #---------------------REMOVE_SAVING--------------------

            del bow_train_tokens
            del bow_train_counts
            del bow_test_tokens
            del bow_test_counts
            del bow_val_tokens
            del bow_val_counts
            del bow_test_h1_tokens
            del bow_test_h1_counts
            del bow_test_h2_tokens
            del bow_test_h2_counts
        
        return self.word2id, self.id2word, train_dataset, test_dataset, val_dataset

    def create_train_test_val_data_for_topic_model(self,
                                    for_lda_model = True,  
                                    length_one_word_remove = True, 
                                    punctuation_lower = True, 
                                    stopwords_filter  = True,
                                    max_df = 0.85, 
                                    min_df = 0.01, 
                                    stopwords_remove_from_voca = True):
        self.load_tokenize_texts(self.source)
        self.preprocess_texts(length_one_word_remove, punctuation_lower, stopwords_filter)
        self.split_and_create_voca_from_trainset(max_df, min_df, stopwords_remove_from_voca)
        _,_,train_dataset, test_dataset, val_dataset = self.create_bow_and_savebow_for_each_set()
        return self.vocabulary, self.word2id, self.id2word, train_dataset, test_dataset, val_dataset
    
    def write_info_vocab_to_text(self):
        
        f = open(f'prepared_data/info_vocab_{self.source}.txt', "a")
        f.write(f'min-df: {self.min_df}\t tokens-train: {self.n_tokens_train} \t tokens-valid: {self.n_tokens_val} \t tokens-test: {self.n_tokens_test} \t vocab: {len(self.vocabulary)}')
        f.write("\n")
        f.close()
                
