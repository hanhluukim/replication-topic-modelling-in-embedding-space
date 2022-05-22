import numpy as np
import pickle

save_path = "prepared_data/bert_vocab_embedding.txt"
with open(save_path) as f:
    lines = f.readlines()
print(f'total vectors: {len(lines)}')

bert_embeddings = []
bert_vocab = []
for t in lines[:2]:
    line = t.split("\t")
    w = line[0]
    v = [float(e) for e in line[1].split(" ")[:-1]] #remove \n at the end
    bert_embeddings.append(v) 
    bert_vocab.append(w)
    #break

np.save('prepared_data/bert_embeddings.npy', bert_embeddings)
with open('prepared_data/bert_vocab.pkl', 'wb') as f:
    pickle.dump(bert_vocab, f)
    
del bert_vocab
del bert_embeddings
