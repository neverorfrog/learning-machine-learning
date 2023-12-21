import warnings
from trainer import *
from models import *
from datamodule import Dataset
warnings.filterwarnings("ignore")
from plotting_utils import *
from car_env import *
from training_utils import *

car_env = CarEnv()
dataset = Dataset(load = True)
dataset.summarize()

model = Model(name="new", num_classes = 5, score_function = accuracy, lr=0.001)
trainer = Trainer(max_epochs = 50, batch_size = 16)
trainer.fit(model, dataset, plot = False)

# model.load()
model.evaluate(dataset)

car_env.play(model)