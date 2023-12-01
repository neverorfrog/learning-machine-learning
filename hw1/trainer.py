import torch
import numpy as np
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
        X = torch.tensor(*batch[:-1]).flatten(start_dim = 1, end_dim = -1) #one sample on each row -> X.shape = (m, d_in)
        Y = batch[-1].type(torch.float32)# labels -> shape = (m)        
        return X,Y
        
    def prepare_model(self, model):
        model.trainer = self
        self.board.xlim = [0, self.max_epochs]
        self.model = model

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data, plot = False):
        self.prepare_data(data)
        self.prepare_model(model)
        self.epoch = 0
        self.train_batch_idx = 0
        self.test_batch_idx = 0
        # accuracy = 0
        early_stopping = False
        for self.epoch in range(self.max_epochs): # That is the cycle in each epoch where iterations (as many as minibatches) pass by
            if early_stopping == True: break
            #Training
            for batch in self.train_dataloader:
                X = torch.tensor(*batch[:-1]) #shape = (m,d) 
            
                # one-hot encoding                               
                indices = [np.where(data.classes == batch[-1][j].item())[0].item() for j in range(len(batch[-1]))]
                y = torch.zeros(size = (len(batch[-1]), data.num_classes))
                y[range(len(y)),indices] += 1
                                                
                self.model.training_step(X, y, batch, plot) #loss is a scalar
                self.train_batch_idx += 1               
            
            #Validation
            # if self.test_dataloader is None:
            #     return
            # self.model.eval()
            # stuck_epochs = 0
            # for batch in self.test_dataloader:
            #     X,Y = self.get_data(data, batch)
            #     old_accuracy = accuracy
            #     accuracy = self.model.validation_step(X,Y,plot)
            #     delta = accuracy - old_accuracy
            #     stuck_epochs = (stuck_epochs+1) if delta < 0.01 else 0
            #     if stuck_epochs == 10: early_stopping = True
            #     self.test_batch_idx += 1
                
        # Print accuracy on the test set at the end of all training 
        test_accuracy = self.model.accuracy(data.X_test,data.y_test)
        print(f"Accuracy: {test_accuracy}")
        
    
    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        if train:
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.epoch + 1
            n = self.num_test_batches / self.plot_valid_per_epoch
        self.board.draw(x, value.numpy(),('train_' if train else 'val_') + key, every_n = int(n))