import random
import torch
from training_utils import accuracy, f1_score, random_color_jitter
from torchvision import transforms
from torch import nn

def random_odd(start, end):
    number = random.randint(start, end)
    return number if number % 2 != 0 else number+1

DATA_PARAMS = {
    'use_weighted_sampler': False,
    'train_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
    'val_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
    'train_transform': transforms.Compose([
        transforms.RandomInvert(),
        transforms.RandomAutocontrast(),
        transforms.GaussianBlur(kernel_size=random_odd(3,9)),
        random_color_jitter()
    ]),
    'val_transform': transforms.Compose([
    ]),
    'min_size_per_class': 1700
}

TRAIN_PARAMS = {
    'max_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 64,
    'patience': 5,
    'score_function': accuracy,
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.0001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [4,8,8],
    'kernels': [5,3,3],
    'strides': [3,1,1],
    'pool_kernels': [2,2,2],
    'pool_strides': [2,2,1]
}