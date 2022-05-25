from src.prepare_dataset import TextDataLoader
from src.lda import lda
from src.evaluierung import topicCoherence2, topicDiversity

from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from pathlib import Path
from tqdm import tqdm
from gensim.models import LdaModel

import argparse
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--filter-stopwords', type=str, default="true", help='do not filter oder filter stopwords')
args = parser.parse_args()

stopwords_filter = args.filter_stopwords
print(f'filter stopwords: {stopwords_filter}')
if stopwords_filter == "True":
  stopwords_filter = True
else:
  stopwords_filter = False

print(f'filter stopwords: {stopwords_filter}')
num_topics = 20

if stopwords_filter:
  under_dir = "no_stopwords"
else:
  under_dir = "with_stopwords"

f = open(f'prepared_data/info_vocab_20newsgroups.txt', "a")
for min_df in [2, 5, 10, 30, 100]:
    # data 
    textsloader = None
    textsloader = TextDataLoader(source="20newsgroups", 
                                 train_size=None, test_size=None)
    textsloader.load_tokenize_texts("20newsgroups")
    textsloader.preprocess_texts(length_one_remove=True, 
                                 punctuation_lower = True, 
                                 stopwords_filter = stopwords_filter,
                                 use_bert_embedding = False)
    
    textsloader.split_and_create_voca_from_trainset(
        max_df=0.7, min_df=min_df, 
        stopwords_remove_from_voca=stopwords_filter)
    
    for_lda_model = True
    # bow must be first and can get the get_docs_in_words_for_each_set()
    word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, 
                                                                                                     normalize = True)
    textsloader.write_info_vocab_to_text()
    docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
    del test_set
    del val_set
    del docs_t
    del docs_v

    # lda model
    del textsloader
    for num_topics in [20]:
        print('run LDA training...')
        ldamodel = LdaModel(train_set, num_topics= num_topics, id2word = id2word, passes = 50, random_state = 42)
        #lda(train_set, num_topics, id2word)

        lda_topics = ldamodel.show_topics(num_topics= num_topics, num_words=25)
        
        # topics
        topics = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
        for topic in lda_topics:
            topics.append(preprocess_string(topic[1], filters))
        print(f'number of topics: {len(topics)}')    
        # save topics
        save_dir = f'topics/{under_dir}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{save_dir}/min_df_{min_df}').mkdir(parents=True, exist_ok=True)
        Path(f'{save_dir}/min_df_{min_df}/lda').mkdir(parents=True, exist_ok=True)
        
        save_topics_path = f'{save_dir}/min_df_{min_df}/lda'
        topics_f = open(f'{save_topics_path}/{num_topics}_topics.txt', 'w')
        i = 0
        for tp in tqdm(topics): 
            tp_as_str = " ".join(tp[:10])
            row = f'topic: {i+1} - {tp_as_str} \n'
            topics_f.write(row)
            i += 1
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
                print("no coherrence, using perplexity for test")
            
            eval_f = open(f'{save_topics_path}/{num_topics}_evaluation.txt', 'w')
            eval_f.write(f'name \t topic-coherrence \t topic-diversity \t quality\n')
            eval_f.write(f'{name} \t {tc} \t {td} \t {tc*td}\n')
            eval_f.close()
        del dataset
        del ldamodel
    f.write(100*"-" + "\n")
f.close()   