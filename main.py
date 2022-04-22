from src.preprare_dataset import TextDataLoader

# loading
textsloader = TextDataLoader(source="20newsgroups", for_lda_model = False, train_size=None, test_size=None)
textsloader.load_tokenize_texts("20newsgroups")
textsloader.show_example_raw_texts(n_docs=2)
print("total documents {}".format(len(textsloader.complete_docs)))

# 
textsloader.preprocess_texts(length_one_remove=True, punctuation_lower = True, stopwords_filter = True)
textsloader.split_and_create_voca_from_trainset(max_df=0.85, min_df=0.01, stopwords_remove_from_voca=True)
train, test, val = textsloader.create_bow_and_savebow_for_each_set()
print(train)

# LDA model


# ETM model