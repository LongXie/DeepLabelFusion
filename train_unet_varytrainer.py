#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.model import get_model
from utils.gen_utils import get_logger

from datasets.Prisma3TT1T2Dataset import get_train_loaders
from model.losses import get_loss_criterion
from model.metrics import get_evaluation_metric
from model.trainer import get_trainer
from model.utils import get_number_of_learnable_parameters, get_tensorboard_formatter

# Create main logger
logger = get_logger('VaryTrainer')

# Top level configuration
import yaml
import config.runconfig as runconfig 
c = runconfig.Config_Run()

# Load and log experiment configuration
config = yaml.safe_load(open(c.config_file, 'r'))

# Set up GPU if available
DEFAULT_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
device = config.get('device', DEFAULT_DEVICE)
config['device'] = torch.device(device)
torch.cuda.set_device(1)
print(config['device'])


# In[2]:


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', None)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


# In[3]:


# Create the model
model = get_model(config)
# put the model on GPUs
logger.info(f"Sending the model to '{config['device']}'")
model = model.to(config['device'])

# Log the number of learnable parameters
logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

# Create loss criterion
loss_criterion = get_loss_criterion(config)

# Create evaluation metric
eval_criterion = get_evaluation_metric(config)

# Create the optimizer
optimizer = _create_optimizer(config, model)

# Create learning rate adjustment strategy
lr_scheduler = _create_lr_scheduler(config, optimizer)

# Create data loaders
loaders = get_train_loaders(config, c.train_patch_csv, c.val_patch_csv)

# Create model trainer
trainer = get_trainer(config)

# get tensorboard formatter
trainer_config = config['trainer']
tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))

# Start training
trainer.fit(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device, 
            loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
            logger=logger, checkpoint_dir = c.checkpointDir, tensorboard_formatter=tensorboard_formatter)

