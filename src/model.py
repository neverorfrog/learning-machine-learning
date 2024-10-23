from abc import abstractmethod
import os
from matplotlib import pyplot as plt
import torch
from torch import nn
from src.utils import Parameters
from src.trainer import *
from src.utils import *


class Model(nn.Module, Parameters):
    """The base class of models"""
    def __init__(self, name=None):
        super().__init__()
        self.save_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, X):
        pass
    
    def save(self):
        if self.name is None: return # TODO
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        print("MODEL SAVED!")

    def load(self, name):
        if self.name is None: return # TODO
        path = os.path.join("models",name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        self.eval()
        print("MODEL LOADED!")
            
class Classifier(Model):
    """The base class of models. Not instantiable because forward inference has to be defined by subclasses."""
    def __init__(self, name, num_classes, bias=True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_scores = []
        self.train_scores = []
        self.val_scores = []
        self.training_time = 0
        self.save_parameters() #saves as class fields the parameters of the constructor
        
    def train_step(self,batch): #forward propagation
        inputs = torch.tensor(*batch[:-1]) #one sample on each row -> X.shape = (m, d_in)
        labels = batch[-1].type(torch.long)# labels -> shape = (m)
        logits = self(inputs)
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
            score = report['weighted avg'][self.params['metrics']]
        return loss, score
        
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
        # return torch.softmax(self(X), dim=-1).argmax(axis = -1).squeeze() #shape = (m)
    
    def save(self):
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.state_dict(), open(os.path.join(path,"model.pt"), "wb"))
        torch.save(self.test_scores, open(os.path.join(path,"test_scores.pt"), "wb")) 
        torch.save(self.train_scores, open(os.path.join(path,"train_scores.pt"), "wb"))
        torch.save(self.val_scores, open(os.path.join(path,"val_scores.pt"), "wb"))
        torch.save(self.training_time, open(os.path.join(path,"training_time.pt"), "wb"))
        print("MODEL SAVED!")

    def load(self, name):
        path = os.path.join("models",name)
        self.load_state_dict(torch.load(open(os.path.join(path,"model.pt"),"rb")))
        self.test_scores = torch.load(open(os.path.join(path,"test_scores.pt"),"rb"))
        self.train_scores = torch.load(open(os.path.join(path,"train_scores.pt"),"rb"))
        self.val_scores = torch.load(open(os.path.join(path,"val_scores.pt"),"rb"))
        self.training_time = torch.load(open(os.path.join(path,"training_time.pt"),"rb"))
        self.eval()
        print("MODEL LOADED!")
        
    def plot(self, name, complete=False):        
        plt.plot(self.test_scores, label=f'{name} - test scores')
        if complete:
            plt.plot(self.train_scores, label=f'{name} - train scores')
            plt.plot(self.val_scores, label=f'{name} - val scores')
        plt.legend()
        plt.ylabel('score')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self,data,show=True):
        with torch.no_grad():
            predictions_test = self.predict(data.test_data.samples)
            predictions_train = self.predict(data.train_data.samples)
            predictions_val = self.predict(data.val_data.samples)
        report_test = classification_report(data.test_data.labels, predictions_test, digits=3, output_dict=True)
        report_train = classification_report(data.train_data.labels, predictions_train, digits=3, output_dict=True)
        report_val = classification_report(data.val_data.labels, predictions_val, digits=3, output_dict=True)
        if show: 
            print(report_test)
            plot_confusion_matrix(data.test_data.labels, predictions_test, data.classes, normalize=True)
        self.test_scores.append(report_test['weighted avg'][self.params['metrics']])
        self.train_scores.append(report_train['weighted avg'][self.params['metrics']])
        self.val_scores.append(report_val['weighted avg'][self.params['metrics']])