import math

def sameDoc(document,w1,w2):
elif w1 in document:
	elif w2 in document:
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
##todo
return 1

def pointwiseInf(documents,w1,w2,anzDoc):
	f= math.log(coocurenceP(documents,w1,w2,anzDoc)/(marg(w1,documents)*marg(w2,documents)))/(-math.log(coocurenceP(documents,w1,w2,anzDoc)))
return f

def topicCoherence(topicsWords,anzahlTopics,documents,anzDoc):
c=0
	for topicWords in topicsWords:
		for i in range(10):
			for j in range(i+1,10):
				c=c+pointwiseInf(documents,topicWords[i],topicWords[j],anzDoc)
c=c/(45*anzahlTopics)
return c