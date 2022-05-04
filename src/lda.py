from gensim.models import LdaModel

def lda(corpus,anzTopics,id2word):
	lda=LdaModel(corpus,id2word=id2word,num_topics=anzTopics)
	return lda
