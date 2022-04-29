# building word2vec model for texts
# check similarity to understand for what word-embedding in topic modelling useful is
# just using doc trains to create the embedding-matrix

import gensim
import pickle
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        with open(self.filename) as infile:
            for line in infile:
                yield line.split()

# Gensim code to obtain the embeddings
sentences = MySentences(args.data_file) # a memory-friendly iterator
print('Model training begins')
model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, size=args.dim_rho, 
    iter=args.iters, workers=args.workers, negative=args.negative_samples, window=args.window_size, )
print('Model trained')
vocab = list(json.load(open(args.vocab_file)).keys())
# Write the embeddings to a file
model_vocab = list(model.wv.vocab)
print(len(model_vocab))
del sentences
n = 0
f = open(args.emb_file, 'w')
for v in tqdm(model_vocab):
    if v in vocab:
        vec = list(model.wv.__getitem__(v))
        f.write(v + ' ')
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        f.write(vec_str + '\n')
        n += 1
f.close()
print('DONE! - saved embeddings for ' + str(n) + ' words.')