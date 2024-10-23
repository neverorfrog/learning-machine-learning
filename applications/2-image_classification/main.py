import sys
sys.path.append('/home/flavio/code/machine-deep-learning')
import warnings
warnings.filterwarnings("ignore")

import time
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
torch.manual_seed(2000)
np.random.seed(2000)
random.seed(2000)

from models import *
from car_env import CarEnv
from datamodule import MyDataset
from src.trainer import Trainer
from config import DATA_PARAMS, TRAIN_PARAMS
# from core.plotting_utils import *
# from training_utils import *

car_env = CarEnv()
dataset = MyDataset(load=True, params=DATA_PARAMS)
dataset.summarize('train')

names = []
names.append('boh')
# names.append('vanilla')
# names.append('with weighted sampling')
# names.append('smoteenn')
# names.append('no dropout')
# names.append('deeper')

complete_plot = False
train_model = True

for name in names:
    model = CNN(name,num_classes=5)
    trainer = Trainer(params=TRAIN_PARAMS)

    start_time = time.time()
    if not train_model:
        model.load(name)
    else:
        trainer.fit(model,dataset)
    model.training_time = time.time() - start_time

    plt.plot(model.test_scores, label=f'{name} - test scores')
    if complete_plot:
        plt.plot(model.train_scores, label=f'{name} - train scores')
        plt.plot(model.val_scores, label=f'{name} - val scores')
    # car_env.play(model)
    trainer.evaluate(model, dataset)
# plt.legend()
# plt.ylabel('score')
# plt.xlabel('epoch')
# plt.show()