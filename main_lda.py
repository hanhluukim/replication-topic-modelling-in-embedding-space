from src.prepare_dataset import TextDataLoader
from src.lda import lda
from src.evaluation_of_authors import get_topic_diversity_mod_for_lda, get_topic_coherence
from src.evaluierung import topicCoherence2, topicDiversity

from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from pathlib import Path
from tqdm import tqdm
import gensim


# parameters
stopwords_filter = True
num_topics = 50

    
f = open(f'prepared_data/info_vocab_20newsgroups.txt', "a")

for min_df in [2,5,10,30,50,100]:
    # data 
    textsloader = None
    textsloader = TextDataLoader(source="20newsgroups", 
                                 train_size=None, test_size=None)
    textsloader.load_tokenize_texts("20newsgroups")
    textsloader.preprocess_texts(length_one_remove=True, 
                                 punctuation_lower = True, 
                                 stopwords_filter = stopwords_filter)
    textsloader.split_and_create_voca_from_trainset(
        max_df=0.7, min_df=min_df, 
        stopwords_remove_from_voca=stopwords_filter)
    
    for_lda_model = True
    # bow must be first and can get the get_docs_in_words_for_each_set()
    word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, 
                                                                                                     normalize = True)
    textsloader.write_info_vocab_to_text()
    
    # lda model
    #gensim_corpus_train_set = train_set
    #print(f'test: {gensim_corpus_train_set[0]}')
    docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
    
    for num_topics in [10, 50]:
        ldamodel = lda(train_set, num_topics, id2word)
        #lda(train_set, num_topics ,id2word)
        del textsloader
        lda_topics = ldamodel.show_topics(num_topics=50, num_words=25)
        
        # topics
        topics = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
        for topic in lda_topics:
            topics.append(preprocess_string(topic[1], filters))
        print(f'number of topics: {len(topics)}')    
        # save topics
        Path('topics').mkdir(parents=True, exist_ok=True)
        Path(f'topics/min_df_{min_df}').mkdir(parents=True, exist_ok=True)
        Path(f'topics/min_df_{min_df}/lda').mkdir(parents=True, exist_ok=True)
        
        save_topics_path = f'topics/min_df_{min_df}/lda'
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
            
            eval_f = open(f'{save_topics_path}/{num_topics}_evaluation.txt', 'w')
            eval_f.write(f'name \t topic-coherrence \t topic-diversity \t quality\n')
            eval_f.write(f'{name} \t {tc} \t {td} \t {tc*td}\n')
            eval_f.close()
        del dataset
        
    f.write(100*"-" + "\n")
f.close()   