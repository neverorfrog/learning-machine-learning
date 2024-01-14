import warnings
from trainer import *
from models import *
from datamodule import MyDataset
warnings.filterwarnings("ignore")
from plotting_utils import *
from car_env import *
from training_utils import *
import random
torch.manual_seed(2000)
np.random.seed(2000)
random.seed(2000)
import time

car_env = CarEnv()
dataset = MyDataset(load=False)
dataset.summarize('train')

# model = Ensemble("new", num_classes=5)
# trainer = EnsembleTrainer()

model = CNN("new",num_classes=5)
trainer = Trainer()

start_time = time.time()
trainer.fit(model,dataset)
end_time = time.time()
print(f'Training time: {end_time - start_time}')
model.load("new")
EnsembleTrainer().evaluate(model,dataset)
car_env.play(model)