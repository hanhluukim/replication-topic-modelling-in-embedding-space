from src.preprare_dataset import TextDataLoader
import argparse
from collections import Counter

#parser = argparse.ArgumentParser(description='main.py')
#parser.add_argument('--n-sub-train', type=int, default=150, help='n-samples of train-set')
#parser.add_argument('--n-sub-test', type=int, default=50, help='n-samples of test-set')
#args = parser.parse_args()

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
textsloader.split_and_create_voca_from_trainset(max_df=0.85, min_df=0.01, stopwords_remove_from_voca=True)
print("\n")
train, test, val = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=True)
#print(train)
print("\n")
# BOW-Representation: train['tokens] is the set of unique-word-ids, train['count'] is the token-frequency of each unique-word-id in the document
print("example of train-bow-representation: \n")
print(train['tokens'][0])
print(train['counts'][0])

# LDA model


# ETM model