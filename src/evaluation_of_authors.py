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
    return TD

def get_topic_diversity_mod_for_lda(topics, n_top_words, word2id):
    num_topics = len(topics)
    list_w = np.empty(shape=[num_topics, n_top_words])
    for k in range(num_topics):
      ids = [word2id[word] for word in topics[k]]
      list_w[k, : ] = ids
    n_unique = len(np.unique(list_w))
    TD = n_unique / (n_top_words * num_topics)
    print('Topic diveristy is: {}'.format(TD))
    return TD

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]#.squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc#.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]#.squeeze(0)
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(data, top10words_topics, word2id):
    #data should be document in word-ids
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(top10words_topics)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_words = top10words_topics[k] #[vocab[a] for a in top_10]
        top_10 = [word2id[idx] for idx in top_words]
        TC_k = 0
        counter = 0
        for i, word_id in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word_id)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word_id, top_10[j])
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