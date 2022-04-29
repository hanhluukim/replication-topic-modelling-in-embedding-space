# training file for embedding
# options: cbow, skipgram and bert-embedding
# returns: word-embedding for each word in the vocabulary
# inputs: train-documents in words and the vocabulary (?)

class WordEmbeddingCreator(model_name="CBOW"):
    def __init__(self, saved_model, dataset):
      self.model = saved_model
      self.dataset = dataset
  
