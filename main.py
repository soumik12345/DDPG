from train import Trainer
from matplotlib import pyplot as plt

trainer = Trainer(config_file='./configs/BipedalWalker-v2.json')
trainer.train()