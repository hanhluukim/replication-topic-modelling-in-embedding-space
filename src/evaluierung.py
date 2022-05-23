import math

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
		for i in range(10):
			for j in range(i+1,10):
				c=c+pointwiseInf(documents,topicWords[i],topicWords[j],anzDoc,2)
	c=c/(45*anzahlTopics)
	return c
def topicPerplexityteil2(thetatest1,tests2anzahl_perword,anzahlVocabulary,betatest1):
	erwartung=[]
	h=0
	anzahlwords=0
	for i in range(anzahlVocabulary):
		p=0
		for j in range(len(thetatest1)):
			p=p+thetatest1[j][i]*betatest1[j]
		if p>0:
			erwartung.append(log(p))
		else:
			erwartung.append(0)
	for m in range(anzahlVocabulary):
		h=h+tests2anzahl_perword[m]*erwartung[m]
		anzahlwords=anzahlwords+tests2anzahl_perword[m]
	if anzahlwords==0:
		return 500
	return h/anzahlwords
def topicPerplexityTeil1(thetastest1,tests2anzahl_perword,anzahlVocabulary,betatest1):
	mean=0
	for thetatest1 in thetastest1:
		mean=mean+topicPerplexityteil2(thetatest1,tests2anzahl_perword,anzahlVocabulary,betatest1)
	return	mean/len(thetastest1)
def topicDiversity(topicsTopk):
	unique=[]
	gesamt=len(topicsTopk)*len(topicsTopk[0])
	for topk in topicsTopk:
		for word in topk:
			if not word in unique:
				unique.append(word)
	uniqueAnzahl= len(unique)
	return uniqueAnzahl/gesamt