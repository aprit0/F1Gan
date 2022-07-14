from math import pi
import time
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import writer, SummaryWriter
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from train_compare import Trainer
from datasets import *
from models.wgangp import Generator, Critic

args = {}
args['dataset'] = 'sines' #sines, arma, diy_sines
args['log_name'] = 'disect'+'_{:.6f}'.format(time.time())
args['epochs'] = 100000 #15000
args['lookback'] = 1
args['batches'] = 16
args['checkpoint_path'] = None
args['store_every'] = 5000
args['print_every'] = 2500
args['sample_count'] = 5

# Create Dataloader
if args['dataset'] == 'sines':
    dataset = Sines(frequency_range=[0, 2 * pi], amplitude_range=[0, 2 * pi], seed=42, n_series=200)
elif args['dataset'] == 'diy':
    dataset = Load('train_sin25_100', 1)
else:
    dataset = ARMA((0.7, ), (0.2, ))
    
# Instantiate Generator and Critic + initialize weights
g = Generator(input_size=dataset.dataset.shape[1], output_size=dataset.dataset.shape[1])
g_opt = torch.optim.RMSprop(g.parameters(), lr=0.00005)
d = Critic(features=dataset.dataset.shape[0])
d_opt = torch.optim.RMSprop(d.parameters(), lr=0.00005)
dataloader = DataLoader(dataset, batch_size=args['batches'])

# Instantiate Trainer
trainer = Trainer(g, d, g_opt, d_opt, dataset, print_every=args['print_every'], sample_count=args['sample_count'])
# Train model
trainer.train(dataloader, epochs=args['epochs'], plot_training_samples=True, checkpoint=args['checkpoint_path'])
# Validate model
trainer.validate(args['epochs'], dataset.dataset.shape)
