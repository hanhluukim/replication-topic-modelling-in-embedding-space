import pandas as pd
from pathlib import Path
import time

from src.prepare_dataset import TextDataLoader
# init TextDataLoader für die Datenquelle 20 News Groups
# Daten abrufen vom Sklearn, tokenisieren und besondere Charaktern entfernen
textsloader = TextDataLoader(source="20newsgroups", train_size=None, test_size=None)
textsloader.load_tokenize_texts("20newsgroups")
# Vorverarbeitung von Daten mit folgenden Schritten:
textsloader.preprocess_texts(length_one_remove=True, 
                             punctuation_lower = True, 
                             stopwords_filter = False,
                             use_bert_embedding = True)
# Daten zerlegen für Train, Test und Validation. Erstellen Vocabular aus dem Trainset
min_df= 0
textsloader.split_and_create_voca_from_trainset(max_df=0.8, 
                                                min_df=min_df, 
                                                stopwords_remove_from_voca=False)

# Erstellen BOW-Repräsentation für ETM Modell
for_lda_model = False
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)

# Kontrollieren die Größen von verschiedenen Datensätzen
print(f'Size of the vocabulary after prprocessing ist: {len(textsloader.vocabulary)}')
print(f'Size of train set: {len(train_set["tokens"])}')
print(f'Size of val set: {len(val_set["tokens"])}')
print(f'Size of test set: {len(test_set["test"]["tokens"])}')

# re-erstellen von Dokumenten nach der Vorverarbeitungen. Die Dokumenten sind in Wörtern und werden für Word-Embedding Training benutzt
docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
del docs_t
del docs_v
train_docs_df = pd.DataFrame()
train_docs_df['text-after-preprocessing'] = [' '.join(doc) for doc in docs_tr[:100]]
train_docs_df
del train_docs_df

from src.embedding import WordEmbeddingCreator
vocab = list(word2id.keys()) # vocabulary after preprocessing and creating bow
word2vec_model = 'skipgram'
save_path = Path.joinpath(Path.cwd(), f'prepared_data/min_df_{min_df}')
figures_path = Path.joinpath(Path.cwd(), f'figures/min_df_{min_df}')
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(figures_path).mkdir(parents=True, exist_ok=True)

if word2vec_model!="bert":
    # only for cbow and skipgram model
    wb_creator = WordEmbeddingCreator(model_name=word2vec_model, documents = docs_tr, save_path= save_path)
    wb_creator.train(min_count=0, embedding_size= 300)
    wb_creator.create_and_save_vocab_embedding(vocab, save_path)
del docs_tr
v = vocab[9]
vec = list(wb_creator.model.wv.__getitem__(v))
print(f'word-embedding of the word-- {v}: ')
print(f'dim of vector: {len(vec)}')
# using gensim function
wb_creator.find_most_similar_words(n_neighbor=10, word=v)
# using self-implemented cosine
sw = wb_creator.find_similar_words_self_implemented(10, vocab, v)
df = pd.DataFrame()
df['Ähnliches Wort'] = list(sw.keys())
df['Cosinus-Ähnlichkeit'] = list(sw.values())
print(df.to_latex(index=False))
del sw
del df
#del wb_creator
del textsloader
del word2id
del id2word
del train_set
del test_set
del val_set
# only bert_vocab, not embeddings in this file
with open('prepared_data/bert_vocab.txt') as f:
    lines = f.readlines()
readed_bert_vocab = [e.split("\n")[0] for e in lines]
print(len(readed_bert_vocab))

from src.embedding import BertEmbedding
bert_eb = BertEmbedding('prepared_data') #directory, where the txt.file of bert_vocab_embedding.txt ist
bert_eb.get_bert_embeddings(vocab)
print(bert_eb.bert_embeddings.shape)

print(f'find similar words for {v}: \n')
sw = bert_eb.find_similar_words(v, 10, vocab)
df = pd.DataFrame()
df['Ähnliches Wort'] = list(sw.keys())
df['Cosinus-Ähnlichkeit'] = list(sw.values())
print(df.to_latex(index=False))

# controll the consitence of vocabulary
for w in vocab:
    if w not in bert_eb.bert_vocab:
        print(w)