import torch
from training_utils import random_color_jitter
from torchvision import transforms
from torch import nn

DOMAIN_PARAMS = {
    'num_classes': 5,
    'input_channels': 3
}

DATA_PARAMS = {
    'use_weighted_sampler': True,
    'resample': False,
    'train_class_weights': torch.tensor([0.7,0.7,0.7,0.5,1.6], dtype=torch.float32),
    'val_class_weights': torch.tensor([1,1,1,1,1], dtype=torch.float32),
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
    'max_epochs': 40,
    'learning_rate': 0.0001,
    'batch_size': 64,
    'patience': 5,
    'metrics': 'f1-score',
    'optim_function': torch.optim.Adam,
    'weight_decay': 0.001,
    'loss_function': nn.CrossEntropyLoss()
}

MODEL_PARAMS = {
    'channels': [DOMAIN_PARAMS['input_channels'],10,16,32],
    'kernels': [5,5,5],
    'strides': [1,1,1],
    'pool_kernels': [2,2,2],
    'pool_strides': [2,2,2],
    'fc_dims': [2048,64,DOMAIN_PARAMS['num_classes']],
    'dropout': 0.2
}

# MODEL_PARAMS = {
#     'channels': [DOMAIN_PARAMS['input_channels'],10,16,32,64,64,64],
#     'kernels': [5,5,5,3,3,3],
#     'strides': [1,1,1,1,1,1],
#     'pool_kernels': [2,2,2,2,2,2],
#     'pool_strides': [2,1,2,1,2,1],
#     'fc_dims': [576,64,DOMAIN_PARAMS['num_classes']],
#     'dropout': 0.2
# }

# MODEL_PARAMS = {
#     'channels': [DOMAIN_PARAMS['input_channels'],5,32],
#     'kernels': [7,3],
#     'strides': [3,2],
#     'pool_kernels': [2,2],
#     'pool_strides': [2,2],
#     'fc_dims': [32*3*3,64,DOMAIN_PARAMS['num_classes']],
#     'dropout': 0.2
# }

