import warnings
from trainer import *
from models import *
from datamodule import Dataset
warnings.filterwarnings("ignore")
from plotting_utils import *
from car_env import *
from training_utils import *
torch.manual_seed(200)

car_env = CarEnv()
dataset = Dataset(load = False)
dataset.summarize(train = True)
# print("\n")
# dataset.summarize(train = False)

cnn = CNN(name="new", num_classes = 5)
trainer = Trainer(model=cnn, data=dataset)

trainer.fit()
cnn.load(name="new")
trained = Trainer(cnn, dataset)
trained.evaluate()
car_env.play(cnn)