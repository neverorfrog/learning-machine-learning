import numpy as np
from sklearn.metrics import classification_report
import torch
from plotting_utils import Parameters, plot_confusion_matrix
from training_utils import *
from config import TRAIN_PARAMS as params

class Trainer(Parameters):
    """The base class for training models with data"""
    
    def __init__(self, model, data):
        self.loss_function = params['loss_function']
        self.score_function = params['score_function']
        self.save_parameters()
        
    def evaluate(self):
        predictions_train = self.model.predict(self.data.X_train)
        predictions_test = self.model.predict(self.data.X_test)
        # evaluation against training set
        print("Train Score: ", self.score_function(predictions_train, self.data.y_train, self.model.num_classes))
        # evaluation against test set
        print("Test Score: ", self.score_function(predictions_test, self.data.y_test, self.model.num_classes))
        print(classification_report(self.data.y_test, predictions_test, digits=3))
        plot_confusion_matrix(self.data.y_test, predictions_test, self.data.classes, normalize=True)
        
    def compute_loss(self,batch,train=True): #forward propagation
        inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        labels = batch[-1].type(torch.long)# labels -> shape = (m)
        if train is True:
            logits = self.model(inputs)
            loss = self.loss_function(logits, labels)
            score = 0
        else:
            with torch.no_grad():
                logits = self.model(inputs)
                loss = self.loss_function(logits, labels)   
                predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(labels.dtype) # the most probable class is the one with highest probability
                score = self.score_function(predictions,labels,self.data.num_classes) 
        return loss, score  
    

    # That is the effective training cycle in which the epochs pass by
    def fit(self, plot = False):
        #parametric stuff
        max_epochs = params['max_epochs']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        patience = params['patience']
        
        #stuff for dataset
        train_dataloader = self.data.train_dataloader(batch_size)
        val_dataloader = self.data.val_dataloader(batch_size)
        self.num_train_batches = len(train_dataloader)
        self.num_val_batches = (len(val_dataloader) if val_dataloader is not None else 0)
        
        # stuff for iterations
        train_batch_idx = 0
        val_batch_idx = 0
        
        #stuff for early stopping
        early_stopping = False
        worse_epochs = 0
        best_loss = np.inf
        
        optim = params['optim_function'](self.model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])
                
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
                    
            #Validation
            self.model.eval()
            scores = []            
            losses = []
            for batch in val_dataloader:
                val_batch_idx += 1
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
                n = self.num_val_batches 
            self.board.draw(x, value.to(device).detach().numpy(),('train_' if train else 'val_') + key, every_n = int(n))
            
