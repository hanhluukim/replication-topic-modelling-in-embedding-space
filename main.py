from src.preprare_dataset import TextDataLoader
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--model', type=str, default="LDA", help='which topic model should be used')
#parser.add_argument('--n-sub-test', type=int, default=50, help='n-samples of test-set')
args = parser.parse_args()
model_name = args.model
print(model_name)

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
word2id, id2word, train, test, val = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)
#print(train)
print("\n")
# BOW-Representation: train['tokens] is the set of unique-word-ids, train['count'] is the token-frequency of each unique-word-id in the document
lda_token_ids = []
lda_counts = []
if for_lda_model == True:
  print("train representation for LDA")
  print(f'id2word for LDA: {id2word}')
  print(f'example bow-representation for lda: ')
  for e in train[0]:
    if e[1]!=0:
      lda_token_ids.append(e[0])
      lda_counts.append(e[1])
  print(lda_token_ids)
  print(lda_counts)

for_lda_model = False
word2id, id2word, train, test, val = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model)
print("train-bow-representation for ETM: \n")
print(f'id2word for ETM: {id2word}')

print("compare lda and etm representation: \n")
print(lda_token_ids)
print(lda_counts)
print(train['tokens'][0])
print(train['counts'][0])

print(len(lda_token_ids))
print(len(train['tokens'][0]))

print(100*"=")
print(f'Size of the vocabulary after prprocessing ist: {len(textsloader.vocabulary)}')
print(f'Size of train set: {len(train["tokens"])}')
print(f'Size of val set: {len(val["tokens"])}')
print(f'Size of test set: {len(test["test"]["tokens"])}')

# doc in words for embedding training
print(100*"=")
tr, t, v = textsloader.get_docs_in_words_for_each_set()
print(tr[0])


# LDA model



# ETM model