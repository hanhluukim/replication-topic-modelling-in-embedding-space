import math
import torch

def sameDoc(document,w1,w2):
	if w1 in document:
		if w2 in document:
			return 1
		else:
			return 0
	else:
		return 0

def coocurenceP(documents,w1,w2,anzDoc):
	p=0
	for document in documents:
		p=p+(sameDoc(document,w1,w2)/anzDoc)
	return p

def marg(w,documents):
	anzwAll=0
	anzw=0
	for document in documents:
		anzwAll=anzwAll+len(document)
		for word in document:
			if w==word:
				anzw=anzw+1

	return anzw/anzwAll
def marg2(w,documents):
	anzw=0
	for document in documents:
		if w in document:
			anzw=anzw+1

	return anzw/len(documents)

def pointwiseInf(documents,w1,w2,anzDoc,margBerechnung):
	if coocurenceP(documents,w1,w2,anzDoc)==0: 
		return -1
	if margBerechnung==1:
		f= math.log(coocurenceP(documents,w1,w2,anzDoc)/(marg(w1,documents)*marg(w2,documents)))/(-math.log(coocurenceP(documents,w1,w2,anzDoc)))
	else:
		f= math.log(coocurenceP(documents,w1,w2,anzDoc)/(marg2(w1,documents)*marg2(w2,documents)))/(-math.log(coocurenceP(documents,w1,w2,anzDoc)))
	return f

def topicCoherence(topicsWords,anzahlTopics,documents,anzDoc):
	c=0
	for topicWords in topicsWords:
		for i in range(10):
			for j in range(i+1,10):
				c=c+pointwiseInf(documents,topicWords[i],topicWords[j],anzDoc,1)
	c=c/(45*anzahlTopics)
	return c
def topicCoherence2(topicsWords,anzahlTopics,documents,anzDoc):
	c=0
	for topicWords in topicsWords:
		topicWords = topicWords[:10]
		for i in range(10):
			for j in range(i+1,10):
				c=c+pointwiseInf(documents,topicWords[i],topicWords[j],anzDoc,2)
	c=c/(45*anzahlTopics)
	return c

def topicPerplexityNew(theta_test_1, test2_bows, vocab_size, beta_test_1):
    #covert to torch.tensor
    theta_test_1 = torch.tensor(theta_test_1)
    beta_test_1 = torch.tensor(beta_test_1)
    test2_bows = torch.tensor(test2_bows)
    
    print(theta_test_1.shape)
    print(beta_test_1.shape)
    print(test2_bows.shape)
    
    log_pred_test_1 = torch.log(torch.mm(theta_test_1, beta_test_1))
    
    # true bows h= (doc_i, h_over_vocabulary)
    h = -(log_pred_test_1 * test2_bows).sum(1) #sum over the vocabulary
    n_words_in_each_doc = test2_bows.sum(1).unsqueeze(1).squeeze()
    # perplexity for each document: h(doc)/len(doc)
    ppl_each_doc_in_batch = h/n_words_in_each_doc
    print(ppl_each_in_batch.shape)
    ppl_each_doc_in_batch_exp = torch.exp(ppl_each_doc_in_batch)
    return round(ppl_each_doc_in_batch.mean().item(),2)

def topicPerplexityteil2(thetatest1,tests2anzahl_perword,anzahlVocabulary,betatest1):
    erwartung=[]
    h=0
    anzahlwords=0
    for i in range(anzahlVocabulary):
        p=0
        for j in range(len(thetatest1)):
            p=p+thetatest1[j]*betatest1[j][i]
        if p>0:
            erwartung.append(math.log(p))
        else:
            erwartung.append(0)
    for m in range(anzahlVocabulary):
        h=h+tests2anzahl_perword[m]*erwartung[m] #of each word
        anzahlwords=anzahlwords+tests2anzahl_perword[m]
    if anzahlwords==0:
        print(f'error here')
        return 500
    #h is result of a document
    return math.exp(-h/anzahlwords)

def topicPerplexityTeil1(thetastest1,tests2anzahl_perword,anzahlVocabulary,betatest1):
    mean=0
    for t in range(len(thetastest1)):
        # for a document
        mean=mean+topicPerplexityteil2(thetastest1[t],tests2anzahl_perword[t],anzahlVocabulary,betatest1)
    return mean/len(thetastest1) #over all documents

def topicDiversity(topicsTopk):
	unique=[]
	gesamt=len(topicsTopk)*len(topicsTopk[0])
	for topk in topicsTopk:
		topk = topk[:25]
		for word in topk:
			if not word in unique:
				unique.append(word)
	uniqueAnzahl= len(unique)
	return uniqueAnzahl/gesamt