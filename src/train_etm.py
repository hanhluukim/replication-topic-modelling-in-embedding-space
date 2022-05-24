import torch
from torch import optim
from src.etm import ETM
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from torch import nn
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

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class DocSet(Dataset):
    def __init__(self, set_name, vocab_size, data, normalize_data = True):
        self.set_name = set_name
        self.data = data
        self.normalize_data = normalize_data
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
            if self.normalize_data == True:
                item = item/sum(item)
            item = torch.from_numpy(item).float()
            # normalize the item?
            return item

def loss_function(loss_name, pred_bows, normalized_bows, kl_theta):
    # reconstruction-loss between pred-normalized-bows and normalized-bows??
    #print(pred_bows.shape)
    #print(normalized_bows.shape)
    #sum over the vocabulary
    #print((pred_bows * normalized_bows).sum(1).shape) #over vocabulary of each document in batch
    #print(kl_theta.shape)
   
    #print(f'sum of pred-vector: {sum(pred_bows[0])}')
    #print(f'length of pred-vector: {torch.norm(pred_bows[0])}')

    #print(f'sum of true-vector: {sum(normalized_bows[0])}')
    #print(f'length of true-vector: {torch.norm(normalized_bows[0])}')

    #sum over the vocabulary and mean of datch. covert to float to use mean()
    #torch.log(res+1e-6)
    # using log(pred)
    #almost_zeros = torch.full_like(pred_bows, 1e-6)
    pred_bows_without_zeros = pred_bows.add(torch.full_like(pred_bows, 1e-6))
        
    if loss_name != "paper-loss":
        #cross_entropy = True
        if loss_name == "cross-entropy":
            mean_recon_loss = -(normalized_bows * torch.log(pred_bows_without_zeros)).sum(1).float().mean()
        else:
        # mean square error
            loss = nn.MSELoss(reduce="mean")
            mean_recon_loss = loss(pred_bows_without_zeros, normalized_bows)
    else:
        mean_recon_loss = -torch.log(pred_bows_without_zeros).sum(1).float().mean()
    return mean_recon_loss, kl_theta.mean()

def get_optimizer(model, opt_args):
    if opt_args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.wdecay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt_args.lr)
    return optimizer


class TrainETM():
    def save_checkpoint(self, state, path=None):
        if path==None:
          #print(state.keys())
          Path('checkpoints').mkdir(parents=True, exist_ok=True)
          cp_by_n_topics = f'checkpoints/num_topics_{state["num_topics"]}'
          Path(cp_by_n_topics ).mkdir(parents=True, exist_ok=True)
          #print(path)
          torch.save(state, f'{cp_by_n_topics}/etm_epoch_{state["epoch"]}.pth.tar')
          print(f'Checkpoint saved at checkpoints/etm_epoch_{state["epoch"]}.pth.tar')
          
    def visualize_losses(self, train_losses, neg_rec_losses, neg_kld_losses, figures_path):
        import matplotlib.pyplot as plt
        #------reconstruction-loss
        plt.figure()
        plt.plot(train_losses, label = 'loss-train')
        plt.plot(neg_rec_losses, label = 'rec-loss')
        #plt.plot(neg_kld_losses, label = 'kld')

        plt.title(f'losses for {len(train_losses)} epochs')
        plt.legend()
        plt.savefig(f'{figures_path}/losses_epoch_{len(train_losses)}.png')
        #plt.show()
        plt.close()
        #------
        plt.figure()
        plt.plot(neg_kld_losses, label = 'kld')
        plt.title(f'kld-losses for {len(train_losses)} epochs')
        plt.legend()
        plt.savefig(f'{figures_path}/kld_epoch_{len(train_losses)}.png')
        #plt.show()
        plt.close()
        return True

    def train(self, 
              etm_model,
              loss_name,
              vocab_size, 
              train_args, optimizer_args, training_set,
              normalize_data = True,
              figures_path = None,
              num_topics = 10
              ):
        
        # training setting
        epochs = train_args.epochs
        batch_size = train_args.batch_size
                
        # define etm model
        etm_model = etm_model.to(device)
                
        # get optimizer is class of follow attributs: optimizer.name, optimizer.lr, optimizer.wdecay
        opt = get_optimizer(etm_model, optimizer_args)
        
        #data loading with DataLoader, data must be transform to Dataset Object
        train_loader = DataLoader(
            DocSet("train_set", vocab_size, training_set, normalize_data), 
            batch_size, 
            shuffle=True, drop_last = True, 
            num_workers = 0, worker_init_fn = np.random.seed(seed))
        print(f'number of batches: {len(train_loader)}')
        val_loader = DataLoader(
            DocSet("train_set", vocab_size, training_set), 
            batch_size, 
            shuffle=True, drop_last = True, 
            num_workers = 0, worker_init_fn = np.random.seed(seed))
        
        # starting training
        epoch_losses = []
        neg_rec_losses = []
        neg_kld_losses = []
        for epoch in range(0, epochs):
            #print('Epoch {}/{}:'.format(epoch, epochs))
            # mode-train
            etm_model.train()
            epoch_loss = 0
            neg_rec = 0
            neg_kld = 0
            for j, batch_doc_as_bows in enumerate(train_loader, 1):
                #print(f'batch-shape: {batch_doc_as_bows.shape}')
                opt.zero_grad()
                # get the output from net
                pred_bows, kl_theta = etm_model.forward(batch_doc_as_bows.to(device))
                pred_bows = pred_bows.to(device)
                # compute the individual losses
                reconstruction_loss, kld_loss = loss_function(loss_name, pred_bows, batch_doc_as_bows.to(device), kl_theta)
                #print(f'reconstruction loss: {reconstruction_loss}')
                #print(f'KL-divergence loss: {kld_loss}')
                avg_batch_loss = reconstruction_loss + kld_loss
                # backward and update weights
                avg_batch_loss.backward()
                opt.step()
                # sum(total_loss_batch)/size(batch)
                epoch_loss += avg_batch_loss
                neg_rec += reconstruction_loss
                neg_kld += kld_loss

            epoch_loss = (epoch_loss/len(train_loader)).item()
            neg_rec = (neg_rec/len(train_loader)).item()
            neg_kld = (neg_kld/len(train_loader)).item()

            print(f'Epoch: {epoch+1}/{epochs}  -  Loss: {round(epoch_loss,5)} \t Rec: {round(neg_rec,5)} \t KL: {round(neg_kld,5)}')
            epoch_losses.append(epoch_loss)
            neg_rec_losses.append(neg_rec)
            neg_kld_losses.append(neg_kld)
        # save checkpoints
        self.save_checkpoint(
              {
                  'epoch': epochs, 
                  'num_topics': num_topics,
                  'params': {
                    'lr': optimizer_args.lr, 
                    'wdecay': optimizer_args.wdecay,
                    },
                  'total_losses': epoch_losses,
                  'rec_losses': neg_rec_losses,
                  'kld_losses': neg_kld_losses,
                  'state_dict': etm_model.state_dict(), 
                  'optimizer' : opt.state_dict()
                  }
              , path = None)
        # visualize the losses during training
        self.visualize_losses(epoch_losses, neg_rec_losses, neg_kld_losses, figures_path)
      

        
            

 
        