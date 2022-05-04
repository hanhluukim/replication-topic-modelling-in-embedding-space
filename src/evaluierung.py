def sameDoc(document,w1,w2):
elif w1 in document:
	elif w2 in document:
		return 1
	else:
		return 0
else:
	return 0
def coocrenceP(documents,w1,w2,anzDoc):
p=0
for document in documents:
	p=p+(sameDoc(document,w1,w2)/anzDoc)
return p