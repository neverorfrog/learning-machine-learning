from core.utils import HyperParameters
import torch
from core.plotting import *
import numpy as np

class Trainer(HyperParameters):
    """The base class for training models with data"""
    def __init__(self, max_epochs, plot_train_per_epoch = 2, plot_valid_per_epoch = 2):
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def prepare_data(self, data):
        self.batch_size = data.batch_size
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def get_data(self, data, batch):
        X = torch.tensor(*batch[:-1]).flatten(start_dim = 1, end_dim = -1) #one sample on each row -> X.shape = (m, d_in)
        Y = batch[-1].type(torch.float32)# labels -> shape = (m)
        return X,Y
        
    def prepare_model(self, model):
        model.trainer = self
        self.board.xlim = [0, self.max_epochs]
        self.model = model

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data, plot = True, scratch = False):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        accuracy = 0
        early_stopping = False
        for self.epoch in range(self.max_epochs): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            
            #Training
            self.model.train() 
            for batch in self.train_dataloader:
                #Forward propagation
                X,Y = self.get_data(data, batch)
                n = len(X) * self.num_train_batches
                loss = self.model.training_step(X,Y,n,plot) #loss is a scalar
                self.train_batch_idx += 1
                #Backward Propagation
                if not scratch:
                    self.optim.zero_grad()
                    with torch.no_grad():
                        loss.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                        self.optim.step() 
                    
            
            #Validation
            if self.val_dataloader is None:
                return
            self.model.eval()
            stuck_epochs = 0
            for batch in self.val_dataloader:
                X,Y = self.get_data(data, batch)
                old_accuracy = accuracy
                accuracy = self.model.validation_step(X,Y,plot)
                delta = accuracy - old_accuracy
                stuck_epochs = (stuck_epochs+1) if delta < 0.01 else 0
                if stuck_epochs == 10: early_stopping = True
                self.val_batch_idx += 1
                
        # Print accuracy on the test set at the end of all training 
        for batch in torch.utils.data.DataLoader(data.test_data, len(data.test_data)):
            X,Y = self.get_data(data, batch)
        test_accuracy = self.model.accuracy(X,Y)
        print(f"Accuracy: {test_accuracy}")
        
    
    def plot(self, key, value, device, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        if train:
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.epoch + 1
            n = self.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, value.to(device).detach().numpy(),('train_' if train else 'val_') + key, every_n = int(n))