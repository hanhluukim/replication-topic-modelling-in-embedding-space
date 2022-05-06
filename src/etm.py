import torch
import torch.nn.functional as F 
from torch import nn
import torchvision.transforms.functional as TF
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class ETM(nn.Module):
    def __init__(self, 
                 num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                 theta_act, 
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
        self.t_drop = nn.Dropout(enc_drop)
        # define the activation function
        if theta_act == 'tanh':
            self.theta_act = nn.Tanh()
        else: #relu
            self.theta_act = nn.ReLU()
        
        # Read the prefitted-embedding. Weights of self.alphas are itself the representation of topic-embeddings
        #_, emsize = embeddings.size()
        print(embeddings[0])
        self.vocab_embeddings_rho = torch.from_numpy(np.array(embeddings)).float().to(device)
        self.topic_embeddings_alphas = nn.Linear(rho_size, num_topics, bias=False)
        
        # define the encoder-network
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        
    def reparameterize(self, mu, logvar):
        # trick to get the sample from Gaussian-Distribuation for update gradient-updating
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)
    
    def encode(self, normalized_bows):
        # return latent variables
        # get mu and logsigma for the next representation of inputted data in latent space with gaussion distribution
        q_theta = self.q_theta(normalized_bows) #encoder_network get the input data as normalized bows
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)

        mu_theta = self.mu_q_theta(q_theta) #using nn.linear
        logsigma_theta = self.logsigma_q_theta(q_theta) #using nn.linear

        #print(f'kld-size {(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp()).shape}')
        #print(f'kld-size {torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).shape}')
        #print(f'kld-size {torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean().shape}')
        #https://arxiv.org/pdf/1312.6114.pdf -DKL in Gaussian Case. With log-var-trick is little different
        kl_theta = -0.5 * torch.sum(
          1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), 
          dim=-1
          )#.mean()
        return mu_theta, logsigma_theta, kl_theta
    
    def get_theta_document_distribution_over_topics(self, mu_theta, logsigma_theta):
        # get the sampling reprensentation of inputted data from latent space
        z = self.reparameterize(mu_theta, logsigma_theta)
        #print(f'shape of z: {z.shape}')
        theta = F.softmax(z, dim=-1) 
        #print(f'shape of theta: {theta.shape}')
        return theta
    
    def get_beta_topic_distribution_over_vocab(self):
        try:
            logit = self.topic_embeddings_alphas(self.vocab_embeddings_rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.topic_embeddings_alphas(self.vocab_embeddings_rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        #print(f'shape of beta: {beta.shape}')
        return beta
    
    def decode(self, theta, beta):
        # make the predictions about the per-doc distribution over vocabulary (or over the words in the documents)
        # try to reconstruct the document-distribution over words (vocabulary)
        # after multiplication maybe zeros. add 1e-6 to get no-zeros matrix
        predictions = torch.mm(theta, beta)
        #almost_zeros = torch.full_like(res, 1e-6)
        #results_without_zeros = res.add(almost_zeros)
        #predictions = torch.log(results_without_zeros)
        return predictions
    
    def forward(self,normalized_bows):
        # recieve the input (normalized bows, so that we use softmax at decoder to make sum=1)
        # get sampling for representation (theta)
        mu_theta, logsigma_theta, kl_theta = self.encode(normalized_bows)
        theta = self.get_theta_document_distribution_over_topics(mu_theta, logsigma_theta)
        # get topic-embeddings. they will be used like context-embedding by cbows
        beta = self.get_beta_topic_distribution_over_vocab()
        # make the prediction about the per-document distribuation over words
        preds = self.decode(theta, beta)
        return preds, kl_theta