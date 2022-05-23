from datetime import datetime
import numpy as np
import pickle
from tqdm import tqdm

start = datetime.now()
save_path = "prepared_data/bert_vocab_embedding.txt"
with open(save_path) as f:
    lines = f.readlines()
print(f'total vectors: {len(lines)}')


bert_embeddings = []
bert_vocab = []

f = open('bert_vocab.txt', 'w')
for t in tqdm(lines, desc="covert embeddings"):
    # save word
    line = t.split("\t")
    w = line[0]
    f.write(w + "\n")
    # save vector
    v = [float(e) for e in line[1].split(" ")[:-1]] #remove \n at the end
    bert_embeddings.append(v) 
    bert_vocab.append(w)
f.close()
np.save('prepared_data/bert_embeddings.npy', bert_embeddings)
with open('prepared_data/bert_vocab.pkl', 'wb') as f:
    pickle.dump(bert_vocab, f)
    
del bert_vocab
del bert_embeddings

f = open("python_covert_embeddings_runtime.txt", "a")
f.write(f'run time: {datetime.now()-start}\n')
f.close()