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

names = []
# names.append('nothing')
# names.append('vanilla')
# names.append('with weighted sampling')
# names.append('smoteenn')
names.append('no dropout')
# names.append('deeper')

complete_plot = False
train_model = False

for name in names:
    model = CNN(name,num_classes=5)
    trainer = Trainer()

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
plt.legend()
plt.ylabel('score')
plt.xlabel('epoch')
plt.show()