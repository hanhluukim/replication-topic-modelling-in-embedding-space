# training file for embedding
# options: cbow, skipgram and bert-embedding
# returns: word-embedding for each word in the vocabulary
# inputs: train-documents in words and the vocabulary (?)

import gensim
import pickle
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from numpy import dot, argmax, indices
import pickle

def unit_vec(vector):
      veclen = np.sqrt(np.sum(vector ** 2))
      return vector/veclen
def get_consine_similarity(vector1, vector2):
      #vector unit length so that just use dot product
      vector1 = np.array(vector1)
      vector2 = np.array(vector2)
      return dot(unit_vec(vector1), unit_vec(vector2))

def get_similar_vectors_to_given_vector(topn, vocab, give_vector, all_vectors):
      dists = []
      for vector2 in all_vectors:
            dists.append(get_consine_similarity(give_vector, vector2))
      dists = np.array(dists)
      top_indices = list(dists.argsort()[:topn]) #do not use dist = 0 of same vectors
      top_words = {} #np.array(vocab)[indices]
      for idx in top_indices:
            top_words[vocab[idx]] = dists[idx]
      return top_words
      #all_vectors[argmax([get_consine_similarity(give_vector, vector2) for vector2 in all_vectors])]
      #return 
      
def read_prefitted_embedding_from_npy_and_txt(model_name, etm_vocab, save_path):
      eb_path =  Path.joinpath(save_path, f'{model_name}_embeddings.npy')
      model_vocab_path = Path.joinpath(save_path, f'{model_name}_vocab.txt')
      # loading
      all_embeddings = np.load(eb_path)
      words_in_vocab = []
      with open(model_vocab_path) as f:
            lines = f.readlines()
      for w in lines:
            words_in_vocab.append(w)
            
      embeddings = []
      for w in etm_vocab:
            idx = model_vocab.index(w)
            eb = all_embeddings[idx]
            embeddings.append(eb)
      del all_embeddings
      del model_vocab
      if words_in_vocab == etm_vocab:
            return words_in_vocab, embeddings
      else:
            print("something wrong with reading files")
            return False
      
      
def read_prefitted_embedding_from_npy_pkl(model_name, etm_vocab, save_path):
      eb_path =  Path.joinpath(save_path, f'{model_name}_embeddings.npy')
      model_vocab_path = Path.joinpath(save_path, f'{model_name}_vocab.pkl')

      all_embeddings = np.load(eb_path)
      with open(model_vocab_path, 'rb') as f:
            model_vocab = pickle.load(f)
      words_in_vocab = []
      embeddings = []
      for w in etm_vocab:
            idx = model_vocab.index(w)
            eb = all_embeddings[idx]
            embeddings.append(eb)
      del all_embeddings
      del model_vocab
      if words_in_vocab == etm_vocab:
            return words_in_vocab, embeddings
      else:
            print("something wrong with reading files")
            return False

def read_prefitted_embedding(model_name, vocab, save_path):
      try:
            save_path = Path.joinpath(save_path, f'{model_name}_vocab_embedding.txt')
      except:
            save_path = Path.joinpath(save_path, f'vocab_embedding.txt')

      with open(save_path) as f:
            lines = f.readlines()
      embedding_data = {}
      for t in lines:
            w = t.split("\t")[0]
            v = [float(e) for e in t.split("\t")[1].split(" ")]
            # check again only word in the etm-vocabulary
            if w in vocab:
                  embedding_data[w] = v
      
      # sort embedding_data again by the ordner of the vocabulary from bow
      words_embeddings = np.array(list(embedding_data.values()))
      words = np.array(list(embedding_data.keys()))
      if words == np.array(vocab):
            indices = [vocab.index(words[i]) for i in range(0,len(words))]
            words_in_vocab = [vocab[i] for i in range(0,len(words))]
            words_embeddings = [e for _, e in sorted(zip(indices, words_embeddings))]
            return words_in_vocab, words_embeddings #list(embedding_data.values())
      else:
            print("something wrong at the embedding.py/read_prefitted_embeddings")
            print("use for testing bert")
            return words, words_embeddings #list(embedding_data.values())

class WordEmbeddingCreator:
      def __init__(self, model_name="cbow", documents = None, save_path = ""):
            """
            Input: documents in List of words, train-settings
            Output: word-embedding for the vocabulary

            Args:
                model_name (str, optional): _description_. Defaults to "cbow".
                documents (_type_, optional): _description_. Defaults to None.
                vocab (_type_, optional): _description_. Defaults to None.
                save_path (str, optional): _description_. Defaults to "".
            """
            self.model_name = model_name
            self.save_path = save_path
            self.documents = documents
            self.model = None
            self.all_embeddings = []
            
      def train(self, min_count = 0, embedding_size = 300):
            if self.model_name=="cbow":
                  print("word-embedding train begins")
                  self.model = gensim.models.Word2Vec(self.documents, 
                                                      seed = 42,
                                                      min_count=min_count, 
                                                      sg=0, 
                                                      window=5,
                                                      size=embedding_size,
                                                      iter=5)
            elif self.model_name=="skipgram":
                  print("train begin:word-embedding with skipgram")
                  self.model = gensim.models.Word2Vec(self.documents, 
                                                      seed = 42,
                                                      min_count=min_count, 
                                                      sg=1, 
                                                      window=5,
                                                      size=embedding_size,
                                                      negative=1,
                                                      iter=5)
            else:
                  print("word-embedding with BERT")
                  print("!!!! please run src/bert_main.py to get prepared_data/bert_vocab_embeddings.txt")
            print("word-embedding train finished")
            # todo: save the trained-model
            #return self.model
          
      def create_and_save_vocab_embedding(self, train_vocab = None, embedding_path = None):
            """_summary_

            Args:
                train_vocab (_type_): vocabulary from the prepare_dataset.py
                embedding_path (_type_): path to save the trained-embedding

            Returns:
                _type_: _description_
            """
            model_vocab = []
            if self.model_name=="bert":
                  model_vocab = [] #bert in other processing, ignore here
            else:
                  model_vocab = list(self.model.wv.vocab)
            print(f'length of vocabulary from word-embedding model {len(model_vocab)}')
            print(f'length of the vocabulary of prepraring-dataset-vocabulary: {len(train_vocab)}')
            del self.documents
            
            f = open(Path.joinpath(embedding_path, f'{self.model_name}_vocab_embedding.txt'), 'w') #add to prepared_data
            # sort words in embedding matrix by the ordner from vocabulary
            for v in tqdm(train_vocab): # sort the list embeddings by words in vocabulary
                if v in model_vocab:
                    vec = list(self.model.wv.__getitem__(v))
                    self.all_embeddings.append(vec)
                    f.write(v + '\t')
                    vec_str = ['%.9f' % val for val in vec]
                    vec_str = " ".join(vec_str)
                    f.write(vec_str + '\n')
            f.close()
            self.model.save(str(Path.joinpath(embedding_path, 'word2vec.model')))
            
            # save embeddings as npy and pkl
            model_vocab_path = f'prepared_data/{self.model_name}_vocab.pkl'
            with open(model_vocab_path, 'wb') as f:
                  pickle.dump(model_vocab, f)
            # save embeddings
            np.save(f'prepared_data/{self.model_name}_embeddings.npy',self.all_embeddings)
            return True
      
      def other_save_embeddings(self, train_vocab):
            all_embeddings = []
            model_vocab = list(self.model.wv.vocab)
            for v in tqdm(train_vocab): # sort the list embeddings by words in vocabulary
                 if v in model_vocab:
                       vec = list(self.model.wv.__getitem__(v))
                       all_embeddings.append(vec)
            np.save(f'{self.model_name}_other_embedding.npy', all_embeddings)
            return all_embeddings

      def find_most_similar_words(self, n_neighbor=20, word = None):
            if word!=None:
                  return self.model.wv.most_similar(word, topn=n_neighbor)
            else:
                  print(f'give a word to get the {n_neighbor} neighbor words')

      def find_similar_words_self_implemented(self, topn, train_vocab, word):
            top_words = {}
            model_vocab = list(self.model.wv.vocab)
            #all_embeddings = self.other_save_embeddings(train_vocab)
            if word in train_vocab:
                  if word in model_vocab:
                        considered_vector = list(self.model.wv.__getitem__(word))
                        top_words = get_similar_vectors_to_given_vector(topn, model_vocab, considered_vector, self.all_embeddings)
            return top_words

      def cluster_words(self, embedding_save_path = None, fig_path = None, n_components=3, text = False):
            import umap.umap_ as umap
            import time
            import plotly.express as px
            from sklearn import cluster
            from sklearn import metrics

            # read embedding from file
            with open(Path.joinpath(embedding_save_path, f'{self.model_name}_vocab_embedding.txt')) as f:
              lines = f.readlines()
            embedding_data = []
            words_data = []
            for t in lines:
              w = t.split("\t")[0]
              v = [float(e) for e in t.split("\t")[1].split(" ")]
              words_data.append(w)
              embedding_data.append(v)
            # using kmean to get clusters of words
            kmeans = cluster.KMeans(n_clusters=10)
            kmeans.fit(embedding_data)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            #print("Cluster id labels for inputted data")
            #print(labels)
            #print("Centroids data")
            #print(centroids)
            # dimension reduction with umap
            reducer = umap.UMAP(random_state=42,n_components=n_components)
            embedding = reducer.fit_transform(embedding_data)
            # show samples after dim-reduction in dataframe
            if n_components == 3:
                  wb = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
            else:
                  wb = pd.DataFrame(embedding, columns=['x', 'y'])
            wb['word'] = words_data
            wb['cluster'] = ['cluster ' + str(c) for c in labels]
            # visualization with plotply
            if n_components==3:
                  fig = px.scatter_3d(wb, 
                                    text = wb['word'],
                                    x='x', y='y', z='z',
                                    color = wb['cluster'],
                                    title ="word-embedding-samples")
            else:
                  # n_components = 2
                  if text:
                    fig = px.scatter(wb, text = wb['word'], x='x', y='y', color=wb['cluster'], title='word embedding samples')
                  else:
                    fig = px.scatter(wb, x='x', y='y', color=wb['cluster'], title='word embedding samples')
                    
            fig.write_image(Path.joinpath(fig_path, f'embedding_space_dim_{n_components}.png'))
            fig.write_html(Path.joinpath(fig_path, f'embedding_space_dim_{n_components}.html'))
            fig.show()
            return True

  
class BertEmbedding:
    def __init__(self, saved_embeddings_text_file):
          self.file_path = saved_embeddings_text_file
          self.bert_vocab = None
          self.bert_embeddings = None
          self.bert_norms = None
    def get_bert_embeddings(self, etm_vocab):
          # filtering words by etm_vocab
          words_in_vocab, vocab_embeddings = read_prefitted_embedding("bert", etm_vocab, self.file_path)
          self.bert_embeddings = np.array(vocab_embeddings)
          self.bert_vocab = words_in_vocab
          self.bert_norms = np.array()
          print("bert-embedding ready!")
          return True

    def find_similar_words(self, word, top_neighbors):
          word_idx_in_vocab = self.bert_vocab.index(word)
          considered_vector = self.bert_embeddings[word_idx_in_vocab]
          top_words = get_similar_vectors_to_given_vector(top_neighbors, self.bert_vocab, considered_vector, self.bert_embeddings)
          return top_words

