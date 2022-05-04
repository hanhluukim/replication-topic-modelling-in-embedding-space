import gensim

def lda(corpus,anzTopics):
lda=LdaModel(corpus,numTopics=anzTopics)
return lda
