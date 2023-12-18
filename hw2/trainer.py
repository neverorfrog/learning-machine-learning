import numpy as np
import torch
from utils import Parameters, ProgressBoard


class Trainer(Parameters):
    """The base class for training models with data"""
    def __init__(self, max_epochs = 10, plot_train_per_epoch = 1, plot_valid_per_epoch = 1):
        self.save_parameters()
        self.board = ProgressBoard()
        
    def prepare_data(self, data):
        self.batch_size = data.batch_size
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
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
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        old_mean_accuracy = 0
        worse_epochs = 0
        early_stopping = False
        
                
        for self.epoch in range(self.max_epochs): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            
            #Training
            self.model.train() 
            for batch in self.train_dataloader:
                #Forward propagation
                X,Y = self.get_data(data, batch)
                n = len(X) * self.num_train_batches
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
            accuracies = []
            losses = []
            for batch in self.test_dataloader:
                X,Y = self.get_data(data, batch)
                accuracies.append(self.model.score(X,Y))
                losses.append(self.model.testing_step(X,Y, plot = False))
                self.val_batch_idx += 1
            
            mean_accuracy = np.mean(accuracies)
            print(f"ACCURACY: {mean_accuracy:.3f} LOSS: {np.mean(losses):.3f}")       
            if mean_accuracy - old_mean_accuracy < 0: worse_epochs += 1
            if worse_epochs == 10: early_stopping = True
            old_mean_accuracy = mean_accuracy
                
        # Print accuracy on the test set at the end of all training 
        test_accuracy = self.model.score(data.X_test,data.y_test)
        print(f"Accuracy: {test_accuracy}")
        model.save()
        
    
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