import torch

def tokenizerfast_for_a_sent(sent, tokenizer):
    this_sent_tokenizer = tokenizer(sent)
    # index of token in the vocabulary
    indexed_tokens = this_sent_tokenizer.input_ids
    segments_ids = [1] * len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_ids_with_belonging_information = this_sent_tokenizer.word_ids()
    return tokens_tensor, segments_tensors, tokens_ids_with_belonging_information
  
def reform_token_embeddings_of_sentence(full_outputs):
    hidden_states = full_outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    #print(token_embeddings.shape)
    token_embeddings = torch.squeeze(token_embeddings, dim=1) # size= (n_hidden_layers, n_tokens, 768)
    #print(token_embeddings.shape)
    token_embeddings = token_embeddings.permute(1,0,2) # size= (n_tokens, n_hidden_layers, 768)
    #print(token_embeddings.shape)
    return token_embeddings 

def get_token_embeddings(reformed_token_embeddings):
    # using sum four last layers
    token_vecs_sum = []
    #print(f'get-token-embedding-function: {reformed_token_embeddings.shape}')
    for i, token in enumerate(reformed_token_embeddings): 
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
        #print(f'original {token.shape} and token-emb after sum {sum_vec.shape}')
    return token_vecs_sum # size: n_tokens: 768

def get_subwords_embeddings_of_word(bert_unique_token_id, tokenized_indices, tokens_embeddings):
    belongging_embeddings_of_word = []
    for idx, tokenizer_idx in enumerate(tokenized_indices):
        if tokenizer_idx == bert_unique_token_id:
            belongging_embeddings_of_word.append(tokens_embeddings[idx])
    return torch.stack(belongging_embeddings_of_word, dim=0)

def get_unique_embedding(embeddings=None, methode="mean"):
    #print(embeddings[0].shape)
    if methode == "mean":
        if embeddings.shape[0] == 1:
          return torch.squeeze(embeddings, dim=0)
        else:
          mean_embedding = torch.mean(embeddings, dim=0) #torch.tensor([embeddings])#.mean()
          return mean_embedding

def need_to_update(sent_tokens_ids):
    special_ids = [101, 102] #of CLS and SEP
    for e in sent_tokens_ids:
        if e in special_ids:
            sent_tokens_ids.remove(e)
    return sent_tokens_ids
  
def get_multiple_embeddings_for_words_in_sent(sent_tokens_ids, sent_outputs_tokens_embeddings):
    # a word can be one time oder multiple times in a sentence
    #print(f'tokens-ids in get_multiple_embeddings_: {sent_tokens_ids}')
    sent_tokens_ids = need_to_update(sent_tokens_ids)
    multiple_words_embeddings = []
    unique_words_ids = list(set(sent_tokens_ids))
    for unique_id in unique_words_ids:
        belong_embeddings = get_subwords_embeddings_of_word(unique_id, sent_tokens_ids, sent_outputs_tokens_embeddings)
        #print(f'word-id: {unique_id} - beling-embeddings shape: {belong_embeddings.shape}')
        # mean of belonging_embeddings to get embedding of whole word
        word_embedding = get_unique_embedding(belong_embeddings, "mean")
        #print(f'mean-word-id {unique_id} word-embedding {word_embedding.shape}')
        multiple_words_embeddings.append(word_embedding)
        #print("----------------------------------------------------------")
    return torch.stack(multiple_words_embeddings, dim=0)

def get_indices_of_word_in_original_sent(word, splitted_original_sent):
    indices = []
    for i, e in enumerate(splitted_original_sent):
        if e == word:
            indices.append(i)
    return indices

def get_final_words_embeddings_in_sent(original_sent, sent_tokens_ids, sent_outputs_tokens_embeddings):
    #import numpy as np
    not_unique_words_embeddings = get_multiple_embeddings_for_words_in_sent(sent_tokens_ids, sent_outputs_tokens_embeddings)
    #print(f'total found embeddings in sent: {not_unique_words_embeddings.shape}')
    original_words_list = original_sent.split(" ")
    words_embeddings_in_sent_dict = {}
    for word in set(original_words_list):
        if word not in ['[CLS]', '[SEP]']:
          word_indices = get_indices_of_word_in_original_sent(word, original_words_list)
          #print(f'word---- {word} ---- indices in original sent: {word_indices}')
          # a word can have different-word-embeddings in the sentence, because a word can occur multple times
          # each occurance has a different embedding for this word
          different_occurrences_embeddings_of_word = not_unique_words_embeddings[word_indices]
          #print(f'test: {different_occurrences_embeddings_of_word.shape}')
          mean_unique_word_embedding = get_unique_embedding(torch.tensor(different_occurrences_embeddings_of_word), "mean")
          words_embeddings_in_sent_dict[word] = mean_unique_word_embedding
    return words_embeddings_in_sent_dict

def save_embeddings_in_sent_to_text(sent_id, words_embeddings_in_sent_dict):
    with open(f'./sent_{str(sent_id)}_words_embeddings.txt', 'w') as fp:
        for word, vector in words_embeddings_in_sent_dict.items():
            # write each item on a new line
            fp.write(f'{word}\t')
            for e in vector.tolist():
                fp.write(f'{e} ')
            fp.write("\n")
        #print('saving embeddings')
    return True

def save_embeddings_to_text(words_embeddings_in_sent_dict):
    with open(r'prepared_data/bert_words_embedding.txt', 'a') as fp:
      for word, vector in words_embeddings_in_sent_dict.items():
          # write each item on a new line
          fp.write(f'{word}\t')
          for e in vector.tolist():
            fp.write(f'{e} ')
          fp.write("\n")
      #print('saving embeddings')
    return True

def vocabulary_embeddings_to_text(vocab_embeddings):
    with open(r'prepared_data/bert_vocab_embedding.txt', 'w') as fp:
      for word, vector in vocab_embeddings.items():
          fp.write(f'{word}\t')
          for e in vector.tolist():
            fp.write(f'{e} ')
          fp.write("\n")
      print('saving embeddings')
    return True