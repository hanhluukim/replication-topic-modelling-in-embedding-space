from src.prepare_dataset import TextDataLoader
from src.lda import lda
from src.evaluierung import topicCoherence2, topicDiversity, topicPerplexityTeil1
from src.evaluierung import topicPerplexityNew

from src.utils_perplexity import get_beta_from_lda, get_theta_from_lda

from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from pathlib import Path
from tqdm import tqdm
from gensim.models import LdaModel
import gensim
import torch
import numpy as np
import math

import argparse
parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('--filter-stopwords', type=str, default="True", help='do not filter oder filter stopwords')
parser.add_argument('--min-df', type=int, default=100, help='minimal document frequency')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--use-tensor', type=bool, default=True, help='using tensors')
parser.add_argument('--batch-test-size', type=int, default=100, help='batch test size')

args = parser.parse_args()
min_df = args.min_df
stopwords_filter = args.filter_stopwords
epochs = args.epochs
using_tensor = args.use_tensor
batch_test_size = args.batch_test_size

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
#---------------------------------------------------------------------------------
min_dfs = [min_df]

for min_df in min_dfs:
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
    word2id, id2word, train_set, test_set, val_set, test_h1_set, test_h2_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, 
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
        ldamodel = LdaModel(train_set, num_topics= num_topics, id2word = id2word, passes = epochs, random_state = 42)
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
        # topic perplexity
        print('calculate perplexity:....')
        vocab_size =  len(list(id2word.values()))
        beta_KV = get_beta_from_lda(ldamodel, num_topics, list(id2word.values()),vocab_size)
        beta_KV = torch.from_numpy(np.array(beta_KV))
        
        theta_test_1_DK = get_theta_from_lda(ldamodel, num_topics, test_h1_set)
        
        n_test_docs_2 = len(test_h2_set)
        test_set_h2_in_bow_sparse_matrix = gensim.matutils.corpus2csc(test_h2_set).transpose()
        
        ppl_over_batches = []
        #batch_test_size = 100
        
        del test_h2_set
        del test_h1_set
        ldamodel.clear()

        i = 0
        j = 0
        while i <= n_test_docs_2:
            if (i+batch_test_size) <= n_test_docs_2:
                print(f'test-docs: from {i} to {i+batch_test_size}')
                if using_tensor == False:
                    theta_test_1_batch = theta_test_1_DK[i:i+batch_test_size]
                    bows_test_2_batch = test_set_h2_in_bow_sparse_matrix[i:i+batch_test_size].toarray().tolist()
                else:
                    #using tensor
                    theta_test_1_batch = torch.tensor(theta_test_1_DK[i:i+batch_test_size])
                    bows_test_2_batch = torch.from_numpy(test_set_h2_in_bow_sparse_matrix[i:i+batch_test_size].toarray())#.tolist()
            else:
                print(f'test-docs: from {i} to {n_test_docs_2}')
                if using_tensor == False:
                    theta_test_1_batch = theta_test_1_DK[i:]
                    bows_test_2_batch = test_set_h2_in_bow_sparse_matrix[i:].toarray().tolist()
                else:
                    theta_test_1_batch = torch.tensor(theta_test_1_DK[i:])
                    bows_test_2_batch = torch.from_numpy(test_set_h2_in_bow_sparse_matrix[i:].toarray())#.tolist()
                    
            if using_tensor == False:
                avg_ppl = topicPerplexityTeil1(theta_test_1_batch, 
                                              bows_test_2_batch, 
                                              vocab_size, 
                                              beta_KV)
            else:
                # covert to tensors
                # perplexity
                print(f'shape of theta-batch: {theta_test_1_batch.shape}')
                print(f'shape of beta-KV-batch: {beta_KV.shape}')
                print(f'shape of bows-test-2-batch: {bows_test_2_batch.shape}') 
                avg_ppl = topicPerplexityNew(theta_test_1_batch, bows_test_2_batch, vocab_size, beta_KV)

            
            print(f'ppl of batch {j+1}: {avg_ppl}')
            ppl_over_batches.append(avg_ppl)
            i = i + batch_test_size
            j += 1
        avg_over_batches = (sum(ppl_over_batches)/len(ppl_over_batches))
        print('avg-ppl-over-batches: {avg_over_batches}')
        ppl_total = round(math.exp(avg_over_batches),1)
        normalized_ppl = ppl_total/vocab_size
        del ppl_over_batches
        del test_set_h2_in_bow_sparse_matrix
        del theta_test_1_DK
        del beta_KV
        
        print(f'end perplexity - show perplexity: ')
        
        print(f'e-normalized-perplexity-lda: {normalized_ppl}')
        print(f'calculate coherence and diversity')
        # topic coherence and topic diversity and quality
        dataset = {'train': None}
        for name, bow_documents in dataset.items():
            tc = 0
            td = 0
            if name == 'train':
                tc = topicCoherence2(topics,len(topics),docs_tr,len(docs_tr))
                td = topicDiversity(topics)
                print(f'topic coherence {tc}')
                print(f'topic diversity {td}')
            else:
                print("no coherrence, using perplexity for test")
            
            eval_f = open(f'{save_topics_path}/{num_topics}_evaluation.txt', 'w')
            eval_f.write(f'name \t topic-coherrence \t topic-diversity \t quality \t perplexity\n')
            eval_f.write(f'{name} \t {tc} \t {td} \t {tc*td} \t {normalized_ppl}\n')
            eval_f.close()
        print(f'ending coherence and diversity')
        del dataset
        del ldamodel
    f.write(100*"-" + "\n")
f.close()   