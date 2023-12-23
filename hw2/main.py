import warnings
from trainer import *
from models import *
from datamodule import Dataset
warnings.filterwarnings("ignore")
from plotting_utils import *
from car_env import *
from training_utils import *

torch.manual_seed(2000)

car_env = CarEnv()
dataset = Dataset(load = True)
dataset.summarize()
class_weights = dataset.class_weights() # inverse to how frequent the class is
loss_function = nn.CrossEntropyLoss(weight=class_weights)

model = Model(name="new", num_classes = 5, loss_function = loss_function, score_function = accuracy, lr=0.0001)
trainer = Trainer(max_epochs = 10, batch_size = 64)
trainer.fit(model, dataset, plot = False)
# model.load()
model.evaluate(dataset)

car_env.play(model)