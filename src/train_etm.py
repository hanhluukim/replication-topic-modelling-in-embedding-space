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
    def __init__(self, set_name, vocab_size, data):
        self.set_name = set_name
        self.data = data
        self.vocab_size = vocab_size
    def __len__(self):
        return len(self.data['tokens'])
    def __getitem__(self, idx):
        with torch.no_grad():
            # create saving-place for input-vector
            item = np.zeros((self.vocab_size))
            for j, word_id in enumerate(self.data['tokens'][idx]):
                # replace the cell of 0 with the tf-value
                item[word_id] = self.data['counts'][idx][j]
            item = torch.from_numpy(item).float()
            # normalize the item?
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


class ETMTrain():
    def save_checkpoint(self, state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))
        
    def train(self, 
              etm_model,
              vocab_size, 
              train_args, optimizer_args, training_set,
              num_topics, t_hidden_size, rho_size, emb_size, theta_act,
              embeddings=None, enc_drop=0.5):
        
        # training setting
        epochs = train_args.epochs
        batch_size = train_args.batch_size
                
        # define etm model
        etm_model = etm_model.to(device)
                
        #optimizer is class of follow attributs: optimizer.name, optimizer.lr, optimizer.wdecay
        opt = get_optimizer(etm_model, optimizer_args)
        
        #data loading with DataLoader, data must be transform to Dataset Object
        train_loader = DataLoader(
            DocSet("train_set", vocab_size, training_set), 
            batch_size, 
            shuffle=True, drop_last = True, 
            num_workers = 0, worker_init_fn = np.random.seed(seed))

        for epoch in range(0, epochs):
            print('Epoch {}/{}:'.format(epoch, epochs))
            # mode-train
            etm_model.train()
            for j, batch_doc_as_bows in enumerate(train_loader, 1):
                batch_normalized_bows = batch_doc_as_bows
                opt.zero_grad()
                # get the output from net
                pred_bows, kl_theta = etm_model.forward(batch_normalized_bows)
                pred_bows = pred_bows.to(device)
                # compute the individual losses
                reconstruction_loss, kld_loss = loss_function(pred_bows, batch_normalized_bows, kl_theta)
                print(reconstruction_loss.size())
                print(kld_loss.size())
                total_loss = (reconstruction_loss + kld_loss).sum()
                print(f'total loss: {total_loss}')
                # backward and update weights
                total_loss.backward()
                opt.step()


        
            

 
        