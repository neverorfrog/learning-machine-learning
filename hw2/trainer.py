import numpy as np
from sklearn.metrics import classification_report
import torch
from datamodule import Dataset, ImageDataset, MyDataset
from models import Ensemble
from plotting_utils import plot_confusion_matrix
from training_utils import *
from config import TRAIN_PARAMS as params
from torch.utils.data import random_split

class Trainer():
    """The base class for training models with data"""   
    def evaluate(self,model,data,show=True):
        with torch.no_grad():
            predictions_test = model.predict(data.test_data.samples)
            predictions_train = model.predict(data.train_data.samples)
            predictions_val = model.predict(data.val_data.samples)
        report_test = classification_report(data.test_data.labels, predictions_test, digits=3, output_dict=True)
        report_train = classification_report(data.train_data.labels, predictions_train, digits=3, output_dict=True)
        report_val = classification_report(data.val_data.labels, predictions_val, digits=3, output_dict=True)
        if show: 
            print(report_test)
            plot_confusion_matrix(data.test_data.labels, predictions_test, data.classes, normalize=True)
        model.test_scores.append(report_test['weighted avg'][params['metrics']])
        model.train_scores.append(report_train['weighted avg'][params['metrics']])
        model.val_scores.append(report_val['weighted avg'][params['metrics']])
        
    def train_step(self,model,batch): #forward propagation
        inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        labels = batch[-1].type(torch.long)# labels -> shape = (m)
        logits = model(inputs)
        loss = self.loss_function(logits, labels)
        return loss
    
    def eval_step(self,model,batch):
        with torch.no_grad():
            inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
            labels = batch[-1].type(torch.long)# labels -> shape = (m)
            logits = model(inputs)
            loss = self.loss_function(logits, labels)
            predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(torch.long) # the most probable class is the one with highest probability
            report = classification_report(batch[-1],predictions, output_dict=True)
            score = report['weighted avg'][params['metrics']]
        return loss, score
    
    
    def fit_epoch(self, epoch, model, optim, train_dataloader, val_dataloader):
        #Training
        model.train() 
        for batch in train_dataloader:
            #Forward propagation
            loss = self.train_step(model,batch)
            #Backward Propagation
            optim.zero_grad()
            with torch.no_grad():
                loss.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                optim.step() 
                
        #Validation
        model.eval()
        epoch_loss = 0           
        epoch_score = 0
        for batch in val_dataloader:
            #Forward propagation
            loss, score = self.eval_step(model,batch)
            #Logging
            epoch_loss += loss.item()
            epoch_score += score
            
        epoch_loss /= len(val_dataloader) 
        epoch_score /= len(val_dataloader)
        print(f"EPOCH {epoch} SCORE: {epoch_score:.3f} LOSS: {epoch_loss:.3f}")  
        model.save()
        
        # Early stopping mechanism     
        if epoch_score < self.best_score and epoch_loss > self.best_loss:
            self.worse_epochs += 1
        else:
            self.best_score = max(epoch_score, self.best_score)
            self.best_loss = min(epoch_loss, self.best_loss)
            self.worse_epochs = 0
        if self.worse_epochs == self.patience: 
            print(f'Early stopping at epoch {epoch} due to no improvement.')  
            return True
        
            
    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data):
        #stuff for dataset
        batch_size = params['batch_size']
        train_dataloader = data.train_dataloader(batch_size)
        val_dataloader = data.val_dataloader(batch_size)
        
        #stuff for early stopping
        self.patience = params['patience']
        self.worse_epochs = 0
        self.best_score = 0
        self.best_loss = np.inf
        
        learning_rate = params['learning_rate']
        optim = params['optim_function'](model.parameters(), lr=learning_rate, weight_decay=params['weight_decay'])
        self.loss_function = params['loss_function']
            
        model.test_scores = []
        model.train_scores = []
        max_epochs = params['max_epochs']   
        for epoch in range(1, max_epochs + 1):
            finished = self.fit_epoch(epoch, model, optim, train_dataloader, val_dataloader)  
            self.evaluate(model, data, show=False)
            if finished: break 
        self.evaluate(model, data)
            
class EnsembleTrainer(Trainer):
    def fit(self, models: Ensemble, data: Dataset):
        for i in range(len(models)):
            # Splitting the dataset into a new random subset for each model
            new_train_data, _, new_train_labels, _ = data.split_train(data.train_data, ratio=0.15)
            new_data = MyDataset(samples=new_train_data, labels=new_train_labels)
            new_data.summarize('train')
            print(f"Model {i+1}")
            super().fit(models[i],new_data)
        models.save()