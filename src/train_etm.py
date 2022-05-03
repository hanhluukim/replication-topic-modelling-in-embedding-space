import torch
from torch import optim
from src.etm import ETM
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import numpy as np
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class DocSet(Dataset):
    def __init__(self, set_name, data):
        self.set_name = set_name
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        with torch.no_grad():
            item = self.data[index]
            # convert item to tensor
            item = TF.to_tensor(item)
            return item

def loss_function(pred_bows, bows, kl_theta):
    # reconstruction-loss between pred-normalized-bows and normalized-bows??
    recon_loss = -(pred_bows * bows).sum(1) #which paper for it?
    return recon_loss, kl_theta

def get_optimizer(model, opt_args):
    if opt_args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.wdecay)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=opt_args.lr)
    return optimizer


class Train():
    def save_checkpoint(self, state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))
        
    def train(self, 
              num_topics, vocab_size, t_hidden_size, rho_size, emb_size, theta_act, 
              train_args, optimizer_args,
              training_set,
              embeddings=None, enc_drop=0.5):
        
        # training setting
        epochs = train_args.epochs
        batch_size = train_args.batch_size
                
        # define etm model
        etm_model = ETM(num_topics, 
                        vocab_size, t_hidden_size, rho_size, emb_size, 
                        theta_act, embeddings, enc_drop).to(device)
                
        #optimizer is class of follow attributs: optimizer.name, optimizer.lr, optimizer.wdecay
        opt = get_optimizer(etm_model, optimizer_args)
        
        #data loading with DataLoader, data must be transform to Dataset Object
        train_loader = DataLoader(
            DocSet("train_set", training_set), 
            batch_size, 
            shuffle=True, drop_last = True, 
            num_workers = 0, worker_init_fn = np.random.seed(seed))

        for epoch in range(0, epochs):
            print('Epoch {}/{}:'.format(epoch, epochs))
            # mode-train
            etm_model.train()
            for j, doc_as_bow in enumerate(train_loader, 1):
                normalized_bows = doc_as_bow
                opt.zero_grad()
                # get the output from net
                pred_bows, kl_theta = etm_model.forward(normalized_bows).to(device)
                # compute the individual losses
                reconstruction_loss, kld_loss = loss_function(pred_bows, normalized_bows, kl_theta)
                total_loss = reconstruction_loss + kld_loss
                # backward and update weights
                total_loss.backward()
                opt.step()


        
            

 
        