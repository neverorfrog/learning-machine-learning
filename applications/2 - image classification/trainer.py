import sys
sys.path.append('/home/flavio/code/main')

from core.trainer import ClassifierTrainer
from datamodule import Dataset, MyDataset
from models import Ensemble
from training_utils import *
from config import TRAIN_PARAMS as params
from torch.utils.data import random_split
            
class EnsembleTrainer(ClassifierTrainer):
    def fit(self, models: Ensemble, data: Dataset):
        for i in range(len(models)):
            # Splitting the dataset into a new random subset for each model
            new_train_data, _, new_train_labels, _ = data.split_train(data.train_data, ratio=0.15)
            new_data = MyDataset(samples=new_train_data, labels=new_train_labels)
            new_data.summarize('train')
            print(f"Model {i+1}")
            super().fit(models[i],new_data)
        models.save()