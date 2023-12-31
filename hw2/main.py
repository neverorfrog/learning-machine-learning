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

car_env = CarEnv()
dataset = MyDataset(load = False)
dataset.summarize('train')

# cnn = CNN(name="906", num_classes = 5)
# trainer = Trainer(model=cnn, data=dataset)

# cnn.load("906")
# trainer.fit()
# cnn.load(name="906")
# trained = Trainer(cnn, dataset)
# trained.evaluate()
# car_env.play(cnn)