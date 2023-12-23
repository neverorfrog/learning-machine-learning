import os
import numpy as np
import torch
from training_utils import count_elements
from plotting_utils import Parameters, ProgressBoard


class Trainer(Parameters):
    """The base class for training models with data"""
    def __init__(self, max_epochs = 10, batch_size = 64, plot_train_per_epoch = 1, plot_valid_per_epoch = 1):
        self.save_parameters()
        self.board = ProgressBoard()
        
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader(self.batch_size)
        self.test_dataloader = data.test_dataloader(self.batch_size)
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = (len(self.test_dataloader) if self.test_dataloader is not None else 0)

    def get_data(self, data, batch):
        X = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        Y = batch[-1].type(torch.float32)# labels -> shape = (m)        
        return X,Y
        
    def prepare_model(self, model):
        model.trainer = self
        self.board.xlim = [0, self.max_epochs]
        self.model = model

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data, plot = True):
        
        # stuff for the optimizer
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizers()
        
        # stuff for iterations
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        
        #stuff for early stopping
        early_stopping = False
        patience = 5
        worse_epochs = 0
        best_loss = np.inf
        
                
        for self.epoch in range(1, self.max_epochs + 1): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            
            #Training
            self.model.train() 
            for batch in self.train_dataloader:
                #Forward propagation
                X,Y = self.get_data(data, batch)
                # print(count_elements(Y))
                loss = self.model.training_step(X,Y,plot) #loss is a scalar
                self.train_batch_idx += 1
                #Backward Propagation
                self.optim.zero_grad()
                with torch.no_grad():
                    loss.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                    self.optim.step() 
                    
            #Validation
            if self.test_dataloader is None:
                return
            self.model.eval()
            scores = []            
            losses = []
            for batch in self.test_dataloader:
                inputs,labels = self.get_data(data, batch)
                loss, score = self.model.testing_step(inputs, labels, plot)
                losses.append(loss)
                scores.append(score)
                self.val_batch_idx += 1
            mean_loss = np.mean(losses) 
            mean_score = np.mean(scores)
            print(f"EPOCH {self.epoch} SCORE: {mean_score:.3f} LOSS: {mean_loss:.3f}")  
            
            # Early stopping mechanism     
            if mean_loss > best_loss:
                worse_epochs += 1
            else:
                best_loss = mean_loss
                worse_epochs = 0
                model.save()
            if worse_epochs == patience: 
                early_stopping = True
                print(f'Early stopping at epoch {self.epoch} due to no improvement.')
        
    
    def plot(self, key, value, device, train):
            """Plot a point in animation."""
            self.board.xlabel = 'epoch'
            if train:
                x = self.train_batch_idx / self.num_train_batches
                n = self.num_train_batches / self.plot_train_per_epoch
            else:
                x = self.epoch + 1
                n = self.num_test_batches / self.plot_valid_per_epoch
            self.board.draw(x, value.to(device).detach().numpy(),('train_' if train else 'val_') + key, every_n = int(n))
            
