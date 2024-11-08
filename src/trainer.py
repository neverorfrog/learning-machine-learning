import torch
import numpy as np
from src.utils import Parameters
from sklearn.metrics import classification_report
from src.plotting_utils import plot_confusion_matrix

class Trainer():
    """The base class for training models with data"""   
    def __init__(self, params: dict):
        self.params = params
        
    def fit_epoch(self, epoch, model, optim, train_dataloader, val_dataloader):
        #Training
        model.train() 
        for batch in train_dataloader:
            # Forward propagation
            loss = model.train_step(batch)
            # Backward Propagation
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
        self.batch_size = self.params['batch_size']
        train_dataloader = data.train_dataloader(self.batch_size)
        val_dataloader = data.val_dataloader(self.batch_size)
        
        #stuff for early stopping
        self.patience = self.params['patience']
        self.worse_epochs = 0
        self.best_score = 0
        self.best_loss = np.inf
        
        self.lr = self.params['learning_rate']
        optim = self.params['optim_function'](model.parameters(), lr=self.lr, weight_decay=self.params['weight_decay'])
        # self.loss_function = self.params['loss_function']
        
        model.test_scores = []
        model.train_scores = []
        max_epochs = self.params['max_epochs']   
        for epoch in range(1, max_epochs + 1):
            finished = self.fit_epoch(epoch, model, optim, train_dataloader, val_dataloader)  
            # model.evaluate(data, show=False) # TODO
            if finished: break 
        # self.evaluate(model, data)
    

# class ClassifierTrainer(Trainer):
#     def __init__(self, params: dict):
#         super().__init__(params)
    
#     def eval_step(self,model,batch):
#         with torch.no_grad():
#             inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
#             labels = batch[-1].type(torch.long)# labels -> shape = (m)
#             logits = model(inputs)
#             loss = self.loss_function(logits, labels)
#             predictions = torch.tensor(logits.argmax(axis = 1).squeeze()).type(torch.long) # the most probable class is the one with highest probability
#             report = classification_report(batch[-1],predictions, output_dict=True)
#             score = report['weighted avg'][self.params['metrics']]
#         return loss, score 