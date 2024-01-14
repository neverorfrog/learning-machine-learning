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
dataset = MyDataset(load=False)
dataset.summarize('train')

ensemble = Ensemble("new_ens", num_classes=5)
EnsembleTrainer().fit(ensemble,dataset)
# ensemble.load(name="new_ens")
EnsembleTrainer().evaluate(ensemble,dataset)
car_env.play(ensemble)