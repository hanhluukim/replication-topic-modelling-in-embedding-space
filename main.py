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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f'using cuda: {torch.cuda.is_available()}')

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--model', type=str, default="LDA", help='which topic model should be used')
parser.add_argument('--epochs', type=int, default=1000, help='train epochs')
args = parser.parse_args()

model_name = args.model
epochs = args.epochs

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
for_lda_model = False
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, normalize = True)
print("train-bow-representation for ETM: \n")
print(f'id2word for ETM: {id2word}')
"""
print("compare lda and etm representation: \n")
print(lda_token_ids)
print(lda_counts)
print(train_set['tokens'][0])
print(train_set['counts'][0])

print(len(lda_token_ids))
print(len(train_set['tokens'][0]))
"""
print(100*"=")
print(f'Size of the vocabulary after prprocessing ist: {len(textsloader.vocabulary)}')
print(f'Size of train set: {len(train_set["tokens"])}')
print(f'Size of val set: {len(val_set["tokens"])}')
print(f'Size of test set: {len(test_set["test"]["tokens"])}')

"""
# example word2id
# show for samples: 100 word2id and id2 word
word2id_df_100 = pd.DataFrame()
word2id_df_100['word'] = list(word2id.keys())[:100]
word2id_df_100['id'] = list(word2id.values())[:100]
print(word2id_df_100)
"""

# doc in words for embedding training
# re-erstellen von Dokumenten nach der Vorverarbeitungen. Die Dokumenten sind in Wörtern und werden für Word-Embedding Training benutzt
docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
"""
train_docs_df = pd.DataFrame()
train_docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs_tr[:100]]
print(train_docs_df)
"""
# save preprocessed documents to file to use with julia later

def save_preprocessed_docs(name=None, docs=None):
  docs_df = pd.DataFrame()
  docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs]
  docs_df.to_csv(f'prepared_data/{name}.csv',index=False)
  del docs_df
  return True

save_preprocessed_docs(name="preprocessed_docs_train", docs = docs_tr)
save_preprocessed_docs(name="preprocessed_docs_test", docs = docs_t)
save_preprocessed_docs(name="preprocessed_docs_val", docs = docs_v)

# embedding training
save_path = Path.joinpath(Path.cwd(), "prepared_data/vocab_embedding.txt")
wb_creator = WordEmbeddingCreator(model_name="cbow", documents = docs_tr, save_path= save_path)
wb_creator.train(min_count=0, embedding_size= 300)
vocab = list(word2id.keys())
wb_creator.create_and_save_vocab_embedding(vocab, save_path)

# embedding word-vectors visualize
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
            
train_args = TrainArguments(epochs=epochs, batch_size=100, log_interval=None)
optimizer_args = OptimizerArguments(optimizer_name="adam", lr=0.005, wdecay=0.1)
print(f'using epochs: {train_args.epochs}')
print(f'using optimizer: {optimizer_args.optimizer}')

# DocSet test
vocab_size = len(list(word2id.keys()))
tr_set = DocSet("train", vocab_size, train_set)
print(len(tr_set))
print(f'sum of vector: {sum(tr_set.__getitem__(0))}')
print(f'length of vector: {torch.norm(tr_set.__getitem__(0))}')


# reading embedding data from file
embedding_data = read_prefitted_embedding(save_path)
# etm-model setting parameters
num_topics = 10
t_hidden_size = 800
rho_size = len(embedding_data[0])
emb_size = len(embedding_data[0])
theta_act = "tanh"

etm_model = ETM(
  num_topics, vocab_size, t_hidden_size, rho_size, emb_size, theta_act, 
  embedding_data, enc_drop=0.5).to(device)
print(50*"-" + 'MODEL-SUMMARY' + 50*"-")
print(etm_model)

print(50*"-" + 'TRAIN' + 50*"-")
# train_set must be normalized??
"""
def get_normalized_bows(dataset):
    
    return dataset
    
train_set = get_normalized_bows(train_sprint("end train time: {}".format(datetime.now()-start_time))et)
#
"""
start = datetime.now()
train_class = TrainETM().train(
    etm_model,
    vocab_size, 
    train_args, optimizer_args, train_set,
    normalize_data = True)
    #num_topics, t_hidden_size, rho_size, emb_size, theta_act,  embedding_data, 0.5)

f = open("python_runtime.txt", "a")
f.write(f'run time: {datetime.now()-start}')
f.close()
