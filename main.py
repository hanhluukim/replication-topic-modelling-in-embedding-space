from src.prepare_dataset import TextDataLoader
import argparse
from collections import Counter
import pandas as pd
from src.embedding import WordEmbeddingCreator, read_prefitted_embedding
from pathlib import Path
from src.train_etm import DocSet, TrainETM
from src.etm import ETM
import torch
from datetime import datetime
#from src.visualization import show_embedding_with_kmeans_umap
import subprocess
import numpy as np
import random

#--------------------deterministic------------------------------------
import os
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)



#---------------------check cuda-------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f'using cuda: {torch.cuda.is_available()}')

#----------------------argument--------------------------------------
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--model', type=str, default="LDA", help='which topic model should be used')
parser.add_argument('--epochs', type=int, default=150, help='train epochs')
parser.add_argument('--wordvec-model', type=str, default="skipgram", help='method for training word embedding')
parser.add_argument('--loss-name', type=str, default="cross-entropy", help='loss-name')
parser.add_argument('--min-df', type=int, default=10, help='minimal document frequency for vocab size')
parser.add_argument('--num-topics', type=int, default=10, help='number of topics')

args = parser.parse_args()


#------------------------parser--------------------------------------
model_name = args.model
epochs = args.epochs
word2vec_model = args.wordvec_model
min_df = args.min_df
num_topics = args.num_topics


#-----------------------check statistic--------------------------------
"""
for min_df in [2, 5, 10, 30, 100]:
      Path('prepared_data').mkdir(parents=True, exist_ok=True)
      Path(f'prepared_data/min_df_{min_df}').mkdir(parents=True, exist_ok=True)
      
      textsloader = TextDataLoader(source="20newsgroups", train_size=None, test_size=None)
      textsloader.load_tokenize_texts("20newsgroups")
      textsloader.preprocess_texts(length_one_remove=True, punctuation_lower = True, stopwords_filter = True)
      textsloader.split_and_create_voca_from_trainset(max_df=0.7, min_df=min_df, stopwords_remove_from_voca=True)
      for_lda_model = False
      word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, normalize = True)
      textsloader.write_info_vocab_to_text()
      del textsloader 
""" 
#------------------------prepare data---------------------------------

textsloader = TextDataLoader(source="20newsgroups", train_size=None, test_size=None)
print("\n")
textsloader.load_tokenize_texts("20newsgroups")

#-------------------------preprocessing-------------------------------

textsloader.preprocess_texts(length_one_remove=True, punctuation_lower = True, stopwords_filter = True)
textsloader.show_example_raw_texts(n_docs=2)
print("\ntotal documents {}".format(len(textsloader.complete_docs)))

textsloader.split_and_create_voca_from_trainset(max_df=0.7, min_df=min_df, stopwords_remove_from_voca=True)
print("\n")

#------------------------save information about vocabulary-------------------

#-------------------------test data for LDA---------------------------
"""
for_lda_model = True
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)

print("\n")
# BOW-Representation: train['tokens] is the set of unique-word-ids, train['count'] is the token-frequency of each unique-word-id in the document
lda_token_ids = []
lda_counts = []
if for_lda_model == True:
  print("train representation for LDA")
  print(f'id2word for LDA: {id2word}')
  print(f'example bow-representation for lda: ')
  for e in train_set[0]:
    if e[1]!=0:
      lda_token_ids.append(e[0])
      lda_counts.append(e[1])
  print(lda_token_ids)
  print(lda_counts)
"""

#-------------------------------test data for ETM--------------------------------------
for_lda_model = False
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, normalize = True)
print("train-bow-representation for ETM: \n")
print(f'example ids of dict-id2word for ETM: {list(id2word.keys())[:5]}')
print(f'example words of dict-id2word for ETM: {list(id2word.values())[:5]}')
print(100*"=")
textsloader.write_info_vocab_to_text()


"""
print("compare lda and etm representation: \n")
print(lda_token_ids)
print(lda_counts)
print(train_set['tokens'][0])
print(train_set['counts'][0])

print(len(lda_token_ids))
print(len(train_set['tokens'][0]))
"""

#------------------------------short summary information-------------------------------
print(f'Size of the vocabulary after prprocessing ist: {len(textsloader.vocabulary)}')
print(f'Size of train set: {len(train_set["tokens"])}')
print(f'Size of val set: {len(val_set["tokens"])}')
print(f'Size of test set: {len(test_set["test"]["tokens"])}')

del test_set
del val_set

"""
# example word2id
# show for samples: 100 word2id and id2 word
word2id_df_100 = pd.DataFrame()
word2id_df_100['word'] = list(word2id.keys())[:100]
word2id_df_100['id'] = list(word2id.values())[:100]
print(word2id_df_100)
"""

#------------------------get docs in words to use for training embedding---------------
#------------------------doc in words for embedding training---------------------------
#------------------------re-erstellen von Dokumenten nach der Vorverarbeitungen.-------
#------------------------Die Dokumenten sind in Wörtern und werden für Word-Embedding Training benutzt

docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
#train_docs_df = pd.DataFrame()
#train_docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs_tr[:10]]
#print(train_docs_df)

del textsloader

#------------------paths

save_path = Path.joinpath(Path.cwd(), f'prepared_data/min_df_{min_df}')
figures_path = Path.joinpath(Path.cwd(), f'figures/min_df_{min_df}')
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(figures_path).mkdir(parents=True, exist_ok=True)

vocab = list(word2id.keys())
#-------------------------embedding training------------------------------------------
if word2vec_model!="bert":
  wb_creator = WordEmbeddingCreator(model_name=word2vec_model, documents = docs_tr, save_path= save_path)
  wb_creator.train(min_count=0, embedding_size= 300)
  wb_creator.create_and_save_vocab_embedding(vocab, save_path)
  #wb_creator.cluster_words(save_path, figures_path , 2)
  # show embedding of some words
  print("neighbor words of some sample selected words")
  print(f'word: {vocab[0]}')
  print(f'vector: {list(wb_creator.model.wv.__getitem__(vocab[0]))[:5]} ')
  """
  for i in range(0,1):
        print(f'neighbor of word {vocab[i]}')
        print([r[0] for r in wb_creator.find_most_similar_words(n_neighbor=5, word=vocab[i])])
        print([r[1] for r in wb_creator.find_most_similar_words(n_neighbor=5, word=vocab[i])])
  """
else:
      #todo run subprocess
      print("using prepared_data/bert_vocab_embedding.txt")
      """
      subprocess.run(
            ["python", "bert_main.py"])
      """     

#--------------------------topic embedding training-----------------------------------

#embedding word-vectors visualize
#embedding_path = save_path
#fig_path = Path.joinpath(Path.cwd(), "figures")
#wb_creator.cluster_words(embedding_path, fig_path, 2)


# setting parameters for training ETM
class TrainArguments:
      def __init__(self, epochs, batch_size, log_interval):
          self.epochs = epochs
          self.batch_size = batch_size
          self.log_interval = log_interval
class OptimizerArguments:
      def __init__(self, optimizer_name, lr, wdecay):
            self.optimizer = optimizer_name
            self.lr = lr
            self.wdecay = wdecay

train_args = TrainArguments(epochs=epochs, batch_size=1000, log_interval=None)
optimizer_args = OptimizerArguments(optimizer_name="adam", lr=0.002, wdecay=0.0000012)
print(f'using epochs: {train_args.epochs}')
print(f'using optimizer: {optimizer_args.optimizer}')

#--------------------------using Dataset Modul to create DocSet-------------------------

vocab_size = len(list(word2id.keys()))
tr_set = DocSet("train", vocab_size, train_set, normalize_data=True)
print(len(tr_set))
print(f'sum of vector: {sum(tr_set.__getitem__(0))}')
print(f'length of vector: {torch.norm(tr_set.__getitem__(0))}')


#---------------------------reading embedding data from file----------------------------
embedding_vocab, embedding_data = read_prefitted_embedding(word2vec_model, vocab, save_path)
w = vocab[0]
idx = embedding_vocab.index(w)
print(embedding_data[idx][:5])
del embedding_vocab

#---------------------------etm-model setting parameters--------------------------------
#num_topics = 50
t_hidden_size = 800
rho_size = len(embedding_data[0])
emb_size = len(embedding_data[0])
theta_act = "ReLU"

#--------------------------etm model setting-------------------------------------------
etm_model = ETM(
  num_topics, vocab_size, t_hidden_size, rho_size, emb_size, theta_act, 
  embedding_data, enc_drop=0.5).to(device)
print(50*"-" + 'MODEL-SUMMARY' + 50*"-")
print(etm_model)


# train_set must be normalized??
"""
def get_normalized_bows(dataset):
    
    return dataset
    
train_set = get_normalized_bows(train_sprint("end train time: {}".format(datetime.now()-start_time))et)
#
"""

#--------------------------training----------------------------------------------------
print(50*"-" + 'TRAIN' + 50*"-")

start = datetime.now()
train_class = TrainETM().train(
    etm_model,
    args.loss_name,
    vocab_size, 
    train_args, optimizer_args, train_set,
    normalize_data = True,
    figures_path = figures_path)
    #num_topics, t_hidden_size, rho_size, emb_size, theta_act,  embedding_data, 0.5)

#--------------------------------RUN TIME------------------------------------------------
f = open("python_train_runtime.txt", "a")
f.write(f'min_df: {min_df} \t vocab-size {len(vocab)} \t epochs: {epochs} \t run time: {datetime.now()-start}\n')
f.close()

#-------------------show topics
topics = etm_model.show_topics(id2word, 25)
topics = [[e[0] for e in tp] for tp in topics] #get only top words

#------------------save topics and evaluation-----------------------
from src.evaluierung import topicCoherence2, topicDiversity
from tqdm import tqdm

save_topics_path = f'topics/min_df_{min_df}/etm'
topics_f = open(f'{save_topics_path}/{num_topics}_topics.txt', 'w')
for tp in tqdm(topics): 
    topics_f.write(" ".join(tp[:10]) + "\n")
topics_f.close()

# topic coherence and topic diversity and quality
dataset = {'train': None}
for name, bow_documents in dataset.items():
    tc = 0
    td = 0
    if name == 'train':
        tc = topicCoherence2(topics,len(topics),docs_tr,len(docs_tr))
        td = topicDiversity(topics)
    else:
        # test dataset - test_topics
        # continue
        print("no coherrence")
    
    eval_f = open(f'{save_topics_path}/{num_topics}_evaluation.txt', 'a')
    eval_f.write(f'name \t epochs\t topic-coherrence \t topic-diversity \t quality\n')
    eval_f.write(f'{name} \t {epochs} \t {tc} \t {td} \t {tc*td}\n')
    eval_f.close()
del dataset


#get_topic_coherence(list(train_set['tokens']), topics, word2id)

del etm_model
del train_class 

#----------------
"""
show_embedding_with_kmeans_umap(
  id2word, embedding_data, num_topics, etm_model.topic_embeddings_alphas.weight,
  figures_path)
"""