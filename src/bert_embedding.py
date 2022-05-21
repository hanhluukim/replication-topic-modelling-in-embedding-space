import torch

def tokenizer_for_a_sent(sent, tokenizer):
    # using for tokenizerFast
    this_sent_tokenizer = tokenizer(sent)
    # index of token in the vocabulary
    indexed_tokens = this_sent_tokenizer.input_ids
    segments_ids = [1] * len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors

def reform_token_embeddings_of_sentence(full_outputs):
    hidden_states = full_outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    print(token_embeddings.shape)
    token_embeddings = torch.squeeze(token_embeddings, dim=1) # size= (n_hidden_layers, n_tokens, 768)
    print(token_embeddings.shape)
    #token_embeddings = token_embeddings.permute(1,0,2) # size= (n_tokens, n_hidden_layers, 768)
    print(token_embeddings.shape)
    return token_embeddings 

def get_token_embeddings(reformed_token_embeddings):
    # using sum four last layers
    token_vecs_sum = []
    print(f'shape befor geting embedding: {reformed_token_embeddings.shape}')
    #i = 0
    for token in reformed_token_embeddings: 
        print(token.shape)
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
        print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
        #i = i + 1
    return token_vecs_sum # size: n_tokens: 768

"""
def find_belonged_embeddings_of_token(token, tokenized_text, embeddings_outputs):
  belonged_embeddings = []
  belonged_indices = []
  for idx, t in enumerate(tokenized_text):
    if t.strip("#") in token:
      belonged_indices.append(i)
  for idx in belonged_indices:
    belonged_embeddings.append(embeddings_outputs[i])
  return belonged_embeddings
"""
def get_belonging_embeddings_of_word(bert_unique_token_id, tokenized_indices, tokens_embeddings):
    belongging_embeddings_of_word = []
    for idx, tokenizer_idx in enumerate(tokenized_indices):
        if tokenizer_idx == bert_unique_token_id:
            belongging_embeddings_of_word.append(tokens_embeddings[idx])
    return belongging_embeddings_of_word

def get_word_embedding(belonging_embeddings=None, methode="mean"):
    if methode == "mean":
        return torch.mean(belonging_embeddings, dim=0)

def need_to_update(sent_tokens_ids):
    special_ids = [101, 102]
    for e in sent_tokens_ids:
        if e in special_ids:
            sent_tokens_ids.remove(e)
    return sent_tokens_ids
  
def get_multiple_embeddings_for_words_in_sent(sent_tokens_ids, sent_outputs_tokens_embeddings):
    # a word can be one time oder multiple times in a sentence
    multiple_words_embeddings = []
    unique_words_ids = list(set(sent_tokens_ids))
    for unique_id in unique_words_ids:
        belong_embeddings = get_belonging_embeddings_of_word(unique_id, sent_tokens_ids, sent_outputs_tokens_embeddings)
        word_embedding = get_word_embedding(belong_embeddings, "mean")
        multiple_words_embeddings.append(word_embedding)
    return multiple_words_embeddings

def get_indices_of_word_in_original_sent(word, splitted_original_sent):
    indices = []
    for i, e in enumerate(splitted_original_sent):
        if e == word:
            indices.append(i)
    return indices

def get_final_words_embeddings_in_sent(original_sent, sent_tokens_ids, sent_outputs_tokens_embeddings):
    not_unique_words_embeddings = get_multiple_embeddings_for_words_in_sent(sent_tokens_ids, sent_outputs_tokens_embeddings)
    original_words_list = original_sent.split(" ")

    for word in original_words_list:
        word_indices = get_indices_of_word_in_original_sent(word, original_words_list)



def get_all_word_embeddings_in_sent(original_sent, sent_tokens_ids, sent_outputs_tokens_embeddings):
    # todo what is one word in multiple position in sentence
    # remove the special token ids from sent_tokens_ids
    def need_to_update(sent_tokens_ids):
        special_ids = [101, 102]
        for e in sent_tokens_ids:
            if e in special_ids:
                sent_tokens_ids.remove(e)
        return sent_tokens_ids
    sent_tokens_id = need_to_update(sent_tokens_ids)
    #----------------------------------------------------------------------------------
    splitted_words_of_sent = original_sent.split(" ")
    # make ids constitent 
    sent_words_ids = { i+1:word for i, word in enumerate(splitted_words_of_sent)}
    sent_words_embeddings = {}
    unique_tokens_ids_list = list(set(sent_tokens_ids)) #neead remove the id of special tokens, which are not in the original sentence
    if len(sent_words_ids.keys()) == len(unique_tokens_ids_list):
      for unique_token_id in unique_tokens_ids_list:
        original_word = sent_words_ids[unique_token_id]
        word_belonging_embeddings = get_belonging_embeddings_of_word(unique_token_id, sent_tokens_ids, sent_outputs_tokens_embeddings)
        sent_words_embeddings[original_word] = get_word_embedding(word_belonging_embeddings)
      return sent_words_embeddings
    else:
      return False
