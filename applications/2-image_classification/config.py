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
    'class_weights': torch.tensor([0.8,0.6,0.6,0.5,1.5], dtype=torch.float32),
    'train_transform': transforms.Compose([
        random_color_jitter(),
        transforms.RandomInvert(),
        transforms.RandomErasing(),
        transforms.RandomRotation(degrees=5),
    ]),
    'val_split_size': 0.15,
    'min_size_per_class': 300
}

TRAIN_PARAMS = {
    'max_epochs': 20,
    'learning_rate': 0.01,
    'batch_size': 64,
    'patience': 5,
    'metrics': 'precision',
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],5,15,30,30],
    'kernels': [7,5,5,5],
    'strides': [1,1,1,1],
    'pool_kernels': [2,2,2,2],
    'pool_strides': [2,2,2,2],
    'fc_dims': [120,DOMAIN_PARAMS['num_classes']],
    'dropout': 0
}

