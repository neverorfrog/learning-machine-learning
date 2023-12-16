import warnings
from trainer import *
from models import *
from datamodule import Dataset
warnings.filterwarnings("ignore")
from utils import *
from car_env import *

dataset = Dataset(path = './data')
dataset.summarize()

car_env = CarEnv()
model = Model(5)
trainer = Trainer(max_epochs = 10)
trainer.fit(model, dataset)
evaluate(model, dataset)

car_env.play(model)