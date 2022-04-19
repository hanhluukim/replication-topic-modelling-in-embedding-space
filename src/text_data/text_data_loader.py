class Tokenizer:
    def __init__(self, lang="en", stopwords=None):
        self.stopwords = stopwords
        
    def tokenize(self, lines):
        return True
    
class TextData:
    def __init__(self, source=None):
        self.source = source
        self.docs = None
    def load_raw_texts(self):
        self.docs = []
        return True
    def preprocess_texts(self, tokenization=True, max_df=0.85, min_df=0.05):
        if tokenization:
            print("tokenization")
        if max_df:
            print("filter by max_df")
        if min_df:
            print("filter by min_df")
        return True
    def split_text_data(self, training_size=0.8):
        
        return True
    
    
        
        