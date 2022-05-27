
import torch
import torch.nn.functional as F 
from torch import nn
import torchvision.transforms.functional as TF

import os
import numpy as np
import random
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

"""
ETM:
1. Encoder-Teil:
    + input: normalized-bag-of-words w_d
    + encoder-output: mu_theta und logvar_theta und KL_theta
        + wir benutzen reparametrization trick and log-var-trick
2. Embedding-Teil:
    + input: word-embedding-matrix
    + matrix-weights (zum Lernen: alphas): alphas (alpha_k = topic-embedding über L)
    + output: rho * alpha
    + softmax: \beta
3. Decoder:
    + Input: \theta, \beta
    + Output: \theta * \beta (D x V)
4. Lossfunktion:
    + cross-entropy
"""

class ETM(nn.Module):
    def __init__(self, 
                 num_topics = 20, 
                 vocab_size = None, 
                 t_hidden_size = 800, 
                 rho_size = 300, 
                 emsize = 300, 
                 theta_act = "ReLU", 
                 embeddings=None, 
                 enc_drop=0.5):
        
        super(ETM, self).__init__()

        # define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size #= D
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        #self.t_drop = nn.Dropout(enc_drop)
        # define the activation function
        if theta_act == 'tanh':
            self.theta_act = nn.Tanh()
        else: #relu
            self.theta_act = nn.ReLU()
        
        # Read the prefitted-embedding. Weights of self.alphas are itself the representation of topic-embeddings
        #print(f'type embdding: {type(embeddings)}')

        self.vocab_embeddings_rho = torch.from_numpy(np.array(embeddings)).float().to(device)
        # word-embedding-vocabulary will be the input for topic_embedding_alphas
        # weights of topic_embedding_alphas are the representation of each topic
        self.topic_embeddings_alphas = nn.Linear(rho_size, num_topics, bias=False)
        
        # define the encoder-network
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        # using log-var-trick to get latent variable
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        
    def reparameterize(self, mu, logvar):
        # trick to get the sample from Gaussian-Distribuation for update gradient-updating
        # using log-var-trik to allowed positive and negative values
        # see the tutorial: https://www.youtube.com/watch?v=pmvo0S3-G-I
        if self.train():
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu) # e*stad + \mu
        else:
            return mu
    
    def encode(self, normalized_bows):
        # return latent variables
        # get mu and logsigma for the next representation of inputted data in latent space with gaussion distribution
        q_theta = self.q_theta(normalized_bows) #encoder_network get the input data as normalized bows
        mu_theta = self.mu_q_theta(q_theta) #using nn.linear
        logsigma_theta = self.logsigma_q_theta(q_theta) #using nn.linear

        #print(f'kld-size {(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp()).shape}')
        #print(f'kld-size {torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).shape}')
        #print(f'kld-size {torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean().shape}')
        #https://arxiv.org/pdf/1312.6114.pdf -DKL in Gaussian Case. With log-var-trick is little different
        K = self.num_topics
        # this one equal to the equation (11). We use log-var-trick
        kl_theta = -0.5 * torch.sum(
          1 - logsigma_theta.exp()  - mu_theta.pow(2)  + logsigma_theta, 
          dim=-1
          )
        kl_theta_plot = 0.5 * torch.sum(
          1 - logsigma_theta.exp()  - mu_theta.pow(2)  + logsigma_theta, 
          dim=-1
          )
        return mu_theta, logsigma_theta, kl_theta, kl_theta_plot
    
    def get_theta_document_distribution_over_topics(self, mu_theta, logsigma_theta):
        # get the sampling reprensentation of inputted data from latent space
        z = self.reparameterize(mu_theta, logsigma_theta)
        #print(f'shape of z: {z.shape}')
        theta = F.softmax(z, dim=-1) 
        #print(f'shape of theta: {theta.shape}')
        return theta
    
    def get_beta_topic_distribution_over_vocab(self):
        prod = self.topic_embeddings_alphas(self.vocab_embeddings_rho)
        beta = F.softmax(prod, dim=0).transpose(1, 0) ## softmax over vocab dimension
        #print(f'shape of beta: {beta.shape}')
        return beta
    
    def decode(self, theta, beta):
        # make the predictions about the per-doc distribution over vocabulary (or over the words in the documents)
        # try to reconstruct the document-distribution over words (vocabulary)
        # after multiplication maybe zeros. add 1e-6 to get no-zeros matrix
        predictions = torch.mm(theta, beta)
        return predictions
    
    def forward(self,normalized_bows):
        # recieve the input (normalized bows, so that we use softmax at decoder to make sum=1)
        # get sampling for representation (theta)
        mu_theta, logsigma_theta, kl_theta, kl_theta_plot = self.encode(normalized_bows)
        theta = self.get_theta_document_distribution_over_topics(mu_theta, logsigma_theta)
        # get topic-embeddings. they will be used like context-embedding by cbows
        beta = self.get_beta_topic_distribution_over_vocab()
        # make the prediction about the per-document distribuation over words
        preds = self.decode(theta, beta)
        return preds, kl_theta, kl_theta_plot

    def get_all_topics_embeddings(self):
        return self.topic_embeddings_alphas.weights
    
    def show_topics(self, id2word, num_top_words):
        """
        beta is topic-distribution over the vocabulary
        alphas.weights is topic-embedding of each topic
        """

        #print(vocab[:10])
        with torch.no_grad():
          topics = []
          #print("show topics: ")
          #beta: topic-distribution over the vocabulary: beta K*V
          betas = self.get_beta_topic_distribution_over_vocab()
          #print(betas[0])
          #print(torch.sum(betas[0]))
          #print(f'shape of beta: {betas.shape}')
          for beta in betas:
            # get the index of words, which idx haven the most probabilities
            top_indices = list(beta.argsort()[-num_top_words:])
            #print(list(top_indices))
            top_words = {id2word[wid.item()]:beta[wid].item() for wid in top_indices}
            top_words = sorted(top_words.items(), key=lambda x: x[1], reverse=True)
            topics.append(top_words)
          return topics