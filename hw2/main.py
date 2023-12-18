import warnings
from trainer import *
from models import *
from datamodule import Dataset
warnings.filterwarnings("ignore")
from utils import *
from car_env import *

car_env = CarEnv()
dataset = Dataset(load = False)
dataset.summarize()

model = Model(name="dec-18", num_classes=5, lr=0.00001)
trainer = Trainer(max_epochs = 50)
trainer.fit(model, dataset, plot = False)

# model.load()
evaluate(model, dataset)

car_env.play(model)