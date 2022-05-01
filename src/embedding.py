# training file for embedding
# options: cbow, skipgram and bert-embedding
# returns: word-embedding for each word in the vocabulary
# inputs: train-documents in words and the vocabulary (?)

import gensim
import pickle
import os
import numpy as np
from tqdm import tqdm

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
            
      def train(self, min_count, embedding_size):
            if self.model_name=="cbow":
                  print("train begin: word-embedding with cbow")
                  self.model = gensim.models.Word2Vec(self.documents, 
                                                      min_count=min_count, 
                                                      sg=0, 
                                                      window=5,
                                                      vector_size=embedding_size)
            elif self.model_name=="skipgram":
                  print("train begin:word-embedding with skipgram")
                  self.model = gensim.models.Word2Vec(self.documents, 
                                                      min_count=min_count, 
                                                      sg=1, 
                                                      window=5,
                                                      vector_size=embedding_size)
            else:
                  print("word-embedding with BERT")
            print("train finished")
            # todo: save the trained-model
            return self.model
          
      def create_and_save_vocab_embedding(self, train_vocab, embedding_path):
            """_summary_

            Args:
                train_vocab (_type_): vocabulary from the prepare_dataset.py
                embedding_path (_type_): path to save the trained-embedding

            Returns:
                _type_: _description_
            """
            model_vocab = []
            if self.model_name=="bert":
                  model_vocab = []
            else:
                  model_vocab = list(self.model.wv.vocab)
            print(len(model_vocab))
            del self.documents
            f = open(embedding_path, 'w')
            for v in tqdm(model_vocab):
                if v in train_vocab:
                    vec = list(self.model.wv.__getitem__(v))
                    f.write(v + '\t')
                    vec_str = ['%.9f' % val for val in vec]
                    vec_str = " ".join(vec_str)
                    f.write(vec_str + '\n')
            f.close()
            return True

  
