import random
import torch
from training_utils import random_color_jitter
from torchvision import transforms
from torch import nn

def random_odd(start, end):
    number = random.randint(start, end)
    return number if number % 2 != 0 else number+1

DOMAIN_PARAMS = {
    'num_classes': 5,
    'input_channels': 3
}

DATA_PARAMS = {
    'use_weighted_sampler': True,
    'train_class_weights': torch.tensor([0.7,0.4,0.4,0.4,1.45], dtype=torch.float32),
    'val_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
    'train_transform': transforms.Compose([
        random_color_jitter(),
        transforms.RandomInvert(),
        transforms.RandomRotation(degrees=15)
    ]),
    'val_split_size': 0.15,
    'min_size_per_class': 300
}

TRAIN_PARAMS = {
    'max_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 64,
    'patience': 5,
    'score_function': "f1-score",
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],16,16,32],
    'kernels': [3,3,3],
    'strides': [2,2,1],
    'pool_kernels': [2,2,2],
    'pool_strides': [2,2,2],
    'fc_dims': [32,16,DOMAIN_PARAMS['num_classes']],
    'dropout': 0.2
}