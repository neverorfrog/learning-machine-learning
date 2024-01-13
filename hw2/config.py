import random
import torch
from training_utils import random_color_jitter
from torchvision import transforms
from torch import nn

DOMAIN_PARAMS = {
    'num_classes': 5,
    'input_channels': 3
}

DATA_PARAMS = {
    'use_weighted_sampler': False,
    'resample': True,
    'train_class_weights': torch.tensor([1.3,1,1,1.3,1], dtype=torch.float32),
    'val_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
    'train_transform': transforms.Compose([
        random_color_jitter(),
        transforms.RandomInvert(),
        transforms.RandomErasing(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip()
    ]),
    'val_split_size': 0.15,
    'min_size_per_class': 300
}

TRAIN_PARAMS = {
    'max_epochs': 70,
    'learning_rate': 0.0001,
    'batch_size': 64,
    'patience': 5,
    'metrics': 'recall',
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],10,20],
    'kernels': [9,5],
    'strides': [3,1],
    'pool_kernels': [2,2],
    'pool_strides': [2,2],
    'fc_dims': [500,DOMAIN_PARAMS['num_classes']],
    'dropout': 0.1
}