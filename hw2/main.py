import warnings
from trainer import *
from models import *
from datamodule import MyDataset
warnings.filterwarnings("ignore")
from plotting_utils import *
from car_env import *
from training_utils import *
torch.manual_seed(0)

car_env = CarEnv()
dataset = MyDataset(load = False)
dataset.summarize('train')

cnn = CNN(name="new", num_classes = 5)
trainer = Trainer(model=cnn, data=dataset)

trainer.fit()
cnn.load(name="new")
trained = Trainer(cnn, dataset)
trained.evaluate()
car_env.play(cnn)