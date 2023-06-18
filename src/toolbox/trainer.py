from toolbox.utils import HyperParameters
import torch
from toolbox.plotting import ProgressBoard

class Trainer(HyperParameters):
    """The base class for training models with data"""
    def __init__(self, max_epochs, plot_train_per_epoch = 5, plot_valid_per_epoch = 1):
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def prepare_data(self, data):
        self.batch_size = data.batch_size
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        self.board.xlim = [0, self.max_epochs]
        self.model = model

    # That is the effective training cycle in which the epochs pass by
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    
    # That is the cycle in each epoch where iterations (as many as minibatches) pass by
    def fit_epoch(self):
        
        #entering into training mode
        self.model.train()
        l = 0 
        for batch in self.train_dataloader:
            l = self.model.get_loss(batch)
            # print(f"Loss at epoch {self.epoch + 1},batch {self.train_batch_idx % self.num_train_batches + 1}: {l}\n")
            self.optim.zero_grad()
            with torch.no_grad():
                l.backward() #here we calculate the chained derivatives (every parameters will have .grad changed)
                self.optim.step()
            self.train_batch_idx += 1
        # print(f"Loss at epoch {self.epoch + 1}: {l}\n")
        
        # entering into evaluation mode
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(batch)
            self.val_batch_idx += 1
    
    def plot(self, key, value, device, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        if train:
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.epoch + 1
            n = self.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, value.to(device).detach().numpy(),('train_' if train else 'val_') + key, every_n=int(n))