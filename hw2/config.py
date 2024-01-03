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
    'train_class_weights': torch.tensor([0.7,0.4,0.4,0.4,1.45], dtype=torch.float32),
    'val_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
    'train_transform': transforms.Compose([
        random_color_jitter(),
        transforms.RandomInvert(),
        transforms.RandomErasing(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip()
    ]),
    'val_split_size': 0.15,
    'min_size_per_class': 1
}

TRAIN_PARAMS = {
    'max_epochs': 70,
    'learning_rate': 0.00005,
    'batch_size': 64,
    'patience': 5,
    'score_function': "f1-score",
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],5,32],
    'kernels': [7,3],
    'strides': [3,2],
    'pool_kernels': [2,2],
    'pool_strides': [2,2],
    'fc_dims': [32*3*3,64,DOMAIN_PARAMS['num_classes']],
    'dropout': 0.2
}