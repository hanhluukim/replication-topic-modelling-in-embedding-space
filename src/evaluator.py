class ModellEvaluator:
    def __init__(self, saved_model, dataset):
        self.model = saved_model
        self.dataset = dataset
    def get_topic_coherence(self):
        return True
    def get_topic_diversity(self):
        return True
    def get_prediction_performance(self):
        return True