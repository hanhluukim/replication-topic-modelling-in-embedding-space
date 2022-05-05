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

def loss_function(pred_bows, normalized_bows, kl_theta):
    # reconstruction-loss between pred-normalized-bows and normalized-bows??
    #print(pred_bows.shape)
    #print(normalized_bows.shape)
    #sum over the vocabulary
    #print((pred_bows * normalized_bows).sum(1).shape) #over vocabulary of each document in batch
    #print(kl_theta.shape)
   
    #print(f'sum of vector: {sum(pred_bows[0])}')
    #print(f'length of vector: {torch.norm(pred_bows[0])}')

    #sum over the vocabulary and mean of datch. covert to float to use mean()
    mean_recon_loss = -(pred_bows * normalized_bows).sum(1).float().mean()
    return mean_recon_loss, kl_theta

def get_optimizer(model, opt_args):
    if opt_args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.wdecay)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=opt_args.lr)
    return optimizer


class TrainETM():
    def save_checkpoint(self, state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))
    def get_normalized_batch(self, batch):
        # if normalize with only in the batch
        # normalize
        #print(f'batch-shape: {batch.shape}')
        #print(batch[0])
        #print(batch[0].shape)
        return batch
    def visualize_losses(self, train_losses, neg_rec_losses, neg_kld_losses):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_losses, label = 'loss-train')
        plt.plot(neg_rec_losses, label = 'rec-loss')
        plt.plot(neg_kld_losses, label = 'kld')

        plt.title(f'losses for {len(train_losses)} epochs')
        plt.legend()
        plt.savefig(f'figures/losses_epoch_{len(train_losses)}.png')
        plt.show()
        plt.close()
    def get_topic_embedding_from_etm(self):
        topic_embeddings = []
        topic_words = []
        return topic_embeddings, topic_words
    def train(self, 
              etm_model,
              vocab_size, 
              train_args, optimizer_args, training_set,
              normalize_data = True
              ):
              #num_topics, t_hidden_size, rho_size, emb_size, theta_act,
              #embeddings=None, enc_drop=0.5
              #):
        
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
                opt.zero_grad()
                # get the output from net
                pred_bows, kl_theta = etm_model.forward(batch_doc_as_bows)
                pred_bows = pred_bows.to(device)
                # compute the individual losses
                reconstruction_loss, kld_loss = loss_function(pred_bows, batch_doc_as_bows, kl_theta)
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

            print(f'Epoch: {epoch}/{epochs}  -  Loss: {epoch_loss} \t Rec: {neg_rec} \t KL: {neg_kld}')
            epoch_losses.append(epoch_loss)
            neg_rec_losses.append(neg_rec)
            neg_kld_losses.append(neg_kld)
            
        # visualize the losses during training
        self.visualize_losses(epoch_losses, neg_rec_losses, neg_kld_losses)
      

        
            

 
        