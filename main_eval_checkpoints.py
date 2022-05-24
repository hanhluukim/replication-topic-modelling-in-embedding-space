from src.evaluierung import topicCoherence2, topicDiversity
from src.prepare_dataset import TextDataLoader
from src.etm import ETM

from tqdm import tqdm
from pathlib import Path
import argparse
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f'using cuda: {torch.cuda.is_available()}')

parser = argparse.ArgumentParser(description='main_eval_checkpoints.py')
parser.add_argument('--model-path', type=str, default="LDA", help='which topic model should be used')

parser.add_argument('--min-df', type=int, default=10, help='minimal document frequency for vocab size')
parser.add_argument('--num-topics', type=int, default=10, help='number of topics')
args = parser.parse_args()

model_path = args.model_path
min_df = args.min_df
num_topics = args.num_topics
epochs = 100

#--Data----------------------------------
textsloader = TextDataLoader(source="20newsgroups", train_size=None, test_size=None)
textsloader.load_tokenize_texts("20newsgroups")
textsloader.preprocess_texts(length_one_remove=True, punctuation_lower = True, stopwords_filter = True)
textsloader.show_example_raw_texts(n_docs=2)
textsloader.split_and_create_voca_from_trainset(max_df=0.7, min_df=min_df, stopwords_remove_from_voca=True)
for_lda_model = False
word2id, id2word, train_set, test_set, val_set = textsloader.create_bow_and_savebow_for_each_set(for_lda_model=for_lda_model, normalize = True)
textsloader.write_info_vocab_to_text()
docs_tr, docs_t, docs_v = textsloader.get_docs_in_words_for_each_set()
del textsloader

#-LOADING checkpoint----------------------------------------
th = torch.load(model_path, map_location=device)
vocab_size = len(list(id2word.keys()))
t_hidden_size = 800
rho_size = 300
emb_size = 300
theta_act = "ReLu"
embedding_data = None

etm_model = ETM(
  num_topics, vocab_size, t_hidden_size, rho_size, emb_size, theta_act, 
  embedding_data, enc_drop=0.5).to(device)
etm_model.load_state_dict(th['state_dict'])
etm_model.eval()
topics = etm_model.show_topics(id2word, 25)

#---------------------------------------
save_topics_path = f'topics/min_df_{min_df}/etm'
Path(save_topics_path).mkdir(parents=True, exist_ok=True)

topics_f = open(f'{save_topics_path}/{num_topics}_topics.txt', 'w')
for tp in tqdm(topics): 
    topics_f.write(" ".join(tp[:10]) + "\n")
topics_f.close()

# topic coherence and topic diversity and quality
dataset = {'train': None}
for name, bow_documents in dataset.items():
    tc = 0
    td = 0
    if name == 'train':
        tc = topicCoherence2(topics,len(topics),docs_tr,len(docs_tr))
        td = topicDiversity(topics)
    else:
        # test dataset - test_topics
        # continue
        print("no coherrence")
    
    eval_f = open(f'{save_topics_path}/{num_topics}_evaluation.txt', 'a')
    eval_f.write(f'name \t epochs\t topic-coherrence \t topic-diversity \t quality\n')
    eval_f.write(f'{name} \t {epochs} \t {tc} \t {td} \t {tc*td}\n')
    eval_f.close()