from src.train_etm import DocSet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
seed=42
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.evaluierung import topicPerplexityTeil1
from src.utils_covert_batch_list import covert_to_list

def get_test_for_completion_task(test_set, 
                                 train_vocab_size,
                                 batch_size=1000, 
                                 normalize_data=True):
    test_loader = DataLoader(
        DocSet("test_set", train_vocab_size, test_set, normalize_data), 
        batch_size, 
        shuffle=True, drop_last = True, 
        num_workers = 0, worker_init_fn = np.random.seed(seed))
    return test_loader

def get_perplexity(etm_model, test_set, vocab_size, test_batch_size):
    print(f'calculate perplexitiy of test dataset: ...')  
    test_set_1 = test_set['test1']
    test_set_2 = test_set['test2']
    test_loader_1 = get_test_for_completion_task(test_set_1, 
                                                 vocab_size,
                                                 batch_size=test_batch_size, 
                                                 normalize_data=True)
    test_loader_2 = get_test_for_completion_task(test_set_2, 
                                                vocab_size,
                                                batch_size=test_batch_size, 
                                                normalize_data=True)
    test_ppl = 0
    other_ppl = 0
    print(f'test-1-loader: {len(test_loader_1)}')
    print(f'test-2-loader: {len(test_loader_2)}')
    
    if len(test_loader_1)==len(test_loader_2):
        etm_model.eval()
        total_perplexity = 0
        total_perplexity_2 = 0
        with torch.no_grad():
            beta = etm_model.get_beta_topic_distribution_over_vocab()
            #for j, batch_doc_as_bows in enumerate(test_loader, 1):
            for i, data in enumerate(zip(test_loader_1, test_loader_2)):
                
                batch_test_1 = data[0]
                batch_test_2 = data[1]
               
                # get theta from the first batch_test_1
                mu_theta, logsigma_theta, kl_theta, _ = etm_model.encode(batch_test_1['normalized'])
                theta_1 = etm_model.get_theta_document_distribution_over_topics(mu_theta, logsigma_theta)
                # using theta of 1 and beta to make prediction for batch_test_2
                pred_batch_test_1 = etm_model.decode(theta_1, beta)
                log_pred_batch_test_1 = torch.log(pred_batch_test_1)
                
                # perplexity of log_pred_batch_1 with batch_test_2
                recon_loss_batch_test_2 = -(log_pred_batch_test_1 * batch_test_2['bow'].to(device)).sum(1) #for each document in batch
                #print(f'loss shape: {recon_loss_batch_test_2.shape}') #cross_entropy for each word in doc, and for each doc in batch # (1000, 3012)
                
                # document-length of each document in batch_test_2 (total words in doc)
                sum_batch_2 = batch_test_2['bow'].sum(1).unsqueeze(1) # [1000,1]
                # number of docs in batch_test_2
                n_words_in_each_doc_in_batch = sum_batch_2.squeeze() #[1000]
                
                # error of each doc in the batch
                loss_new = recon_loss_batch_test_2/n_words_in_each_doc_in_batch
                #print(f'loss shape {loss_new.shape}')
                # mean error over all docs
                loss_new = loss_new.mean().item() #avg over all doc in batches
                total_perplexity += loss_new
                
                # use other perplexity
                """
                theta_test_1_DxK, beta_KxV, count_of_each_word_in_doc_list_test_2 = covert_to_list(theta_1,
                                                                                                   beta, 
                                                                                                   batch_test_2)
                mean_over_batch_ppl = topicPerplexityTeil1(theta_test_1_DxK,
                                                           count_of_each_word_in_doc_list_test_2,
                                                           vocab_size,
                                                           beta_KxV)                   
                total_perplexity_2 += mean_over_batch_ppl
                """
                print(f'batch {i} finished')
                
        avg_loss = total_perplexity/len(test_loader_1)
        #other_ppl = total_perplexity_2/len(test_loader_1)
        test_ppl = round(math.exp(avg_loss),1)
        #print(f'own-ppl: {test_ppl}')
        #print(f'elisabeth-perplexitiy: {avg_loss_2}')
    else:
        print("ERROR: loader works incorrectly")
        
    return test_ppl/vocab_size, test_ppl/vocab_size

#---------------------FOR LDA---------------------------------

def get_beta_from_lda(ldamodel, num_topics, vocab, vocab_size):
    beta = ldamodel.show_topics(num_topics= num_topics, 
                                num_words= vocab_size, 
                                log=False, formatted=False)
    beta_KV = []
    for tp in beta:
        #tp_id = tp[0]
        tp_per_words = dict((word, proba) for word, proba in tp[1])
        proba_over_vocab = [tp_per_words[w] for w in vocab]
        beta_KV.append(proba_over_vocab)
    del vocab
    del vocab_size
    del beta
    return beta_KV
def get_theta_from_lda(ldamodel, num_topics, test_set_h1):
    thetas = []
    for doc in test_set_h1:
        theta_doc = ldamodel.get_document_topics(doc, 
                                                 minimum_probability=0, 
                                                 minimum_phi_value=0, 
                                                 per_word_topics=False)
        theta_doc = dict((topic_id, topic_prob) for topic_id, topic_prob in theta_doc)
        updated_theta_doc = []
        for i in range(0,num_topics):
            updated_theta_doc.append(theta_doc[i])
        thetas.append(updated_theta_doc)
    return thetas

#---------------------------------------------------------------
"""
def get_theta_beta_from_lda(ldamodel, batch_test_1, vocab_size):
    theta_batch = []
    for doc in batch_test_1['bow']:
        theta_doc = ldamodel[doc.tolist()]
        theta_batch.append(theta_doc)
    beta_raw = ldamodel.show_topics(num_words=vocab_size, formatted=False)
    beta = [[e[1] for e in bt] for bt in beta_raw]
    del beta_raw
    return theta_batch, beta
   
def get_perplexity_lda(ldamodel, test_set, vocab_size, test_batch_size = 1000): 
    test_1 = test_set['test1']
    test_2 = test_set['test2']
    
    test_loader_1 = get_test_for_completion_task(test_1, 
                                                vocab_size,
                                                batch_size=test_batch_size, 
                                                normalize_data=True)
    test_loader_2 = get_test_for_completion_task(test_2, 
                                                vocab_size,
                                                batch_size=test_batch_size,
                                                normalize_data=True) 
    total_ppl = 0
    for i, data in enumerate(zip(test_loader_1, test_loader_2)):
        batch_test_1 = data[0]
        batch_test_2 = data[1]   
        theta_batch_1, beta = get_theta_beta_from_lda(ldamodel, batch_test_1, vocab_size)
                                                
        theta_test_1_DxK, beta_KxV, count_of_each_word_in_doc_test_2_list = covert_to_list(theta_batch_1, 
                                                                                        beta, 
                                                                                        batch_test_2)
        
        avg_ppl = topicPerplexityTeil1(theta_test_1_DxK, 
                                    count_of_each_word_in_doc_test_2_list, 
                                    vocab_size, beta_KxV)     
        total_ppl += avg_ppl
    print(f'average perplexity over batches: {total_ppl/len(test_loader_1)}')
    total_ppl = total_ppl/len(test_loader_1)
    print(f'normalized-ppl: {total_ppl/vocab_size}')
    return total_ppl/vocab_size
"""