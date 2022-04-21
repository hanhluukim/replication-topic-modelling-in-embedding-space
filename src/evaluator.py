# modul for evluating the topic-modells by topic-coherence, topic-diversity and combined-metrics
# topic-model will be also tested in dependency of the vocabulary-size
# how will be tested for stop words?

import numpy as np


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