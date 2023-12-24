import numpy as np
import torch
from plotting_utils import Parameters, ProgressBoard
from torch import nn
from training_utils import *

class Trainer(Parameters):
    """The base class for training models with data"""
    
    def __init__(self, model, data, optim_function = torch.optim.Adam, loss_function = nn.CrossEntropyLoss(), score_function = accuracy):
        self.save_parameters()
        
    def compute_loss(self,batch,train=True): #forward propagation
        inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        labels = batch[-1].type(torch.long)# labels -> shape = (m)
        if train is True:
            logits = self.model(inputs)
            loss = self.loss_function(logits, labels)
        else:
            with torch.no_grad():
                logits = self.model(inputs)
                loss = self.loss_function(logits, labels)   
        predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(labels.dtype) # the most probable class is the one with highest probability
        score = self.score_function(predictions,labels,self.data.num_classes) 
        return loss, score  
    

    # That is the effective training cycle in which the epochs pass by
    def fit(self, max_epochs = 10, lr = 0.001, batch_size = 64, patience = 5, plot = False):
        #stuff for dataset
        train_dataloader = self.data.train_dataloader(batch_size)
        test_dataloader = self.data.test_dataloader(batch_size)
        self.num_train_batches = len(train_dataloader)
        self.num_test_batches = (len(test_dataloader) if test_dataloader is not None else 0)
        
        # stuff for iterations
        train_batch_idx = 0
        test_batch_idx = 0
        
        #stuff for early stopping
        early_stopping = False
        worse_epochs = 0
        best_loss = np.inf
        
        optim = self.optim_function(self.model.parameters(), lr=lr)
        
                
        for epoch in range(1, max_epochs + 1): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            
            #Training
            self.model.train() 
            for batch in train_dataloader:
                train_batch_idx += 1
                #Forward propagation
                loss, score = self.compute_loss(batch, train = True)
                if plot:
                    self.plot('loss', loss, self.device, train = True)
                #Backward Propagation
                optim.zero_grad()
                with torch.no_grad():
                    loss.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                    optim.step() 
                    
            #Testing
            self.model.eval()
            scores = []            
            losses = []
            for batch in test_dataloader:
                test_batch_idx += 1
                #Forward propagation
                loss, score = self.compute_loss(batch, train = False)
                if plot:
                    self.plot('loss', loss, self.device, train = False)
                #Logging
                losses.append(loss)
                scores.append(score)
                
            mean_loss = np.mean(losses) 
            mean_score = np.mean(scores)
            print(f"EPOCH {epoch} SCORE: {mean_score:.3f} LOSS: {mean_loss:.3f}")  
            
            # Early stopping mechanism     
            if mean_loss > best_loss:
                worse_epochs += 1
            else:
                best_loss = mean_loss
                worse_epochs = 0
                self.model.save()
            if worse_epochs == patience: 
                early_stopping = True
                print(f'Early stopping at epoch {epoch} due to no improvement.')
        
    
    def plot(self, key, value, device, train, epoch, train_batch_idx):
            """Plot a point in animation."""
            self.board.xlabel = 'epoch'
            if train:
                x = train_batch_idx / self.num_train_batches
                n = self.num_train_batches
            else:
                x = epoch + 1
                n = self.num_test_batches 
            self.board.draw(x, value.to(device).detach().numpy(),('train_' if train else 'val_') + key, every_n = int(n))
            
