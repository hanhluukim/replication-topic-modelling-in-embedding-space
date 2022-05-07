# modul for evluating the topic-modells by topic-coherence, topic-diversity and combined-metrics
# topic-model will be also tested in dependency of the vocabulary-size
# how will be tested for stop words?

import numpy as np


# copies from autor code to test model firstly

import torch 
import numpy as np

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))
    return TC

def nearest_neighbors(model, word):
    nearest_neighbors = model.wv.most_similar(word, topn=20)
    nearest_neighbors = [comp[0] for comp in nearest_neighbors]
    return nearest_neighbors







class EvaluationMetrics:
    def __init__(self, saved_model, dataset):
        self.model = saved_model
        self.dataset = dataset
        
    def overall_topic_quality(self):
        topic_coherence = 0
        topic_diversity = 0
        return topic_coherence*topic_diversity
    
    def get_topic_coherence(self):
        return True
    def get_topic_diversity(self,beta, topk=25):
        # paper: percentage of unique words in the top 25 words of all topics
        # beta is the distribution of k-topics over the vocabulary: beta_{K*V}
        num_topics = beta.shape[0]
        list_w = np.zeros((num_topics, topk)) # use to save only the top 25-words in each topic
        for k in range(num_topics):
            idx = beta[k,:].argsort()[-topk:][::-1]
            list_w[k,:] = idx
            
        n_unique = len(np.unique(list_w))
        TD = n_unique / (topk * num_topics)
        return TD
    def get_prediction_performance(self):
        return True
    def topic_quality_in_prediction_power(self):
        return True
    def result_by_voca_size(self):
        return True