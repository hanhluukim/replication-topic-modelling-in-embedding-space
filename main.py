from src.preprare_dataset import TextDataLoader
import argparse
from collections import Counter
import pandas as pd
from src.embedding import WordEmbeddingCreator
from pathlib import Path
from src.train_etm import DocSet, ETMTrain
from src.etm import ETM

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--model', type=str, default="LDA", help='which topic model should be used')
#parser.add_argument('--n-sub-test', type=int, default=50, help='n-samples of test-set')
args = parser.parse_args()
model_name = args.model

# loading
textsloader = TextDataLoader(source="20newsgroups", train_size=None, test_size=None)
print("\n")
textsloader.load_tokenize_texts("20newsgroups")
print("\n")
textsloader.show_example_raw_texts(n_docs=2)
print("\n")
print("total documents {}".format(len(textsloader.complete_docs)))

# 
textsloader.preprocess_texts(length_one_remove=True, punctuation_lower = True, stopwords_filter = True)
print("\n")
textsloader.split_and_create_voca_from_trainset(max_df=0.7, min_df=10, stopwords_remove_from_voca=True)
print("\n")

for_lda_model = True
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)
#print(train)
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

for_lda_model = False
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)
print("train-bow-representation for ETM: \n")
print(f'id2word for ETM: {id2word}')

print("compare lda and etm representation: \n")
print(lda_token_ids)
print(lda_counts)
print(train_set['tokens'][0])
print(train_set['counts'][0])

print(len(lda_token_ids))
print(len(train_set['tokens'][0]))

print(100*"=")
print(f'Size of the vocabulary after prprocessing ist: {len(textsloader.vocabulary)}')
print(f'Size of train set: {len(train_set["tokens"])}')
print(f'Size of val set: {len(val_set["tokens"])}')
print(f'Size of test set: {len(test_set["test"]["tokens"])}')


# example word2id

# show for samples: 100 word2id and id2 word
word2id_df_100 = pd.DataFrame()
word2id_df_100['word'] = list(word2id.keys())[:100]
word2id_df_100['id'] = list(word2id.values())[:100]
print(word2id_df_100)


# doc in words for embedding training
# re-erstellen von Dokumenten nach der Vorverarbeitungen. Die Dokumenten sind in Wörtern und werden für Word-Embedding Training benutzt
docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
train_docs_df = pd.DataFrame()
train_docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs_tr[:100]]
print(train_docs_df)

# embedding training
save_path = Path.joinpath(Path.cwd(), "vocab_embedding.txt")
wb_creator = WordEmbeddingCreator(model_name="cbow", documents = docs_tr, save_path= save_path)
wb_creator.train(min_count=0, embedding_size= 10)
vocab = list(word2id.keys())
wb_creator.create_and_save_vocab_embedding(vocab, save_path)

# embedding word-vectors visualize
#embedding_path = save_path
#fig_path = Path.joinpath(Path.cwd(), "figures")
#wb_creator.cluster_words(embedding_path, fig_path)

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
            
train_args = TrainArguments(epochs=100, batch_size=100, log_interval=None)
optimizer_args = OptimizerArguments(optimizer_name="Adam", lr=0.005, wdecay=0.1)
print(f'using epochs: {train_args.epochs}')
print(f'using optimizer: {optimizer_args.optimizer}')

# DocSet test
vocab_size = len(list(word2id.keys()))
tr_set = DocSet("train", vocab_size, train_set)
print(len(tr_set))
print(tr_set.__getitem__(0))

# training parameter setting
# reading embedding data from file

with open(save_path) as f:
  lines = f.readlines()
embedding_data = []
for t in lines:
  v = [float(e) for e in t.split("\t")[1].split(" ")]
  embedding_data.append(v)
  
# etm-model setting parameters
num_topics = 5
t_hidden_size = 100
rho_size = len(embedding_data[0])
emb_size = len(embedding_data[0])
theta_act = "relu"

etm_model = ETM(
  num_topics, vocab_size, t_hidden_size, rho_size, emb_size, theta_act, 
  embedding_data, enc_drop=0.5)

# train
train_class = ETMTrain().train(
    etm_model,
    num_topics, 
    vocab_size, 
    train_args, optimizer_args, train_set,
    t_hidden_size, rho_size, emb_size, theta_act,  embedding_data, 0.5)