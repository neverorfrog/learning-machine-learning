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
        self.save_parameters()
        
    def evaluate(self):
        predictions_test = self.model.predict(self.data.test_data.samples)
        print(classification_report(self.data.test_data.labels, predictions_test, digits=3))
        plot_confusion_matrix(self.data.test_data.labels, predictions_test, self.data.classes, normalize=True)
        
    def train_step(self,batch): #forward propagation
        inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        labels = batch[-1].type(torch.long)# labels -> shape = (m)
        logits = self.model(inputs)
        # l2_reg = sum(torch.norm(param) for param in self.model.parameters())
        loss = self.loss_function(logits, labels)
        # loss += params['weight_decay'] * l2_reg
        return loss
    
    def eval_step(self,batch):
        with torch.no_grad():
            inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
            labels = batch[-1].type(torch.long)# labels -> shape = (m)
            logits = self.model(inputs)
            loss = self.loss_function(logits, labels)
            predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(torch.long) # the most probable class is the one with highest probability
            report = classification_report(batch[-1],predictions, output_dict=True)
            score = report['weighted avg'][params['metrics']]
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
        best_score = 0
        best_loss = np.inf
        
        optim = params['optim_function'](self.model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])
                
        for epoch in range(1, max_epochs + 1): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            
            #Training
            self.model.train() 
            for batch in train_dataloader:
                train_batch_idx += 1
                #Forward propagation
                loss = self.train_step(batch)
                if plot:
                    self.plot('loss', loss, self.device, train = True)
                #Backward Propagation
                optim.zero_grad()
                with torch.no_grad():
                    loss.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                    optim.step() 
                    
            #Validation
            self.model.eval()
            epoch_loss = 0           
            epoch_score = 0
            for batch in val_dataloader:
                val_batch_idx += 1
                #Forward propagation
                loss, score = self.eval_step(batch)
                if plot:
                    self.plot('loss', loss, self.device, train = False)
                #Logging
                epoch_loss += loss.item()
                epoch_score += score
                
            epoch_loss /= len(val_dataloader) 
            epoch_score /= len(val_dataloader)
            print(f"EPOCH {epoch} SCORE: {epoch_score:.3f} LOSS: {epoch_loss:.3f}")  
            
            # Early stopping mechanism     
            if epoch_score < best_score and epoch_loss > best_loss:
                worse_epochs += 1
            else:
                self.model.save()
                best_score = max(epoch_score, best_score)
                best_loss = min(epoch_loss, best_loss)
                worse_epochs = 0
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
            
