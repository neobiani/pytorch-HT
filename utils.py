
import torch
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import random
import os
import numpy as np

def create_optimizer(config, model): return optim.Adam(model.parameters(), lr=config['init_lr']/config['warmup_factor'])

def create_lr_scheduler(config, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['n_epochs']-config['warmup_epo'])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=config['warmup_factor'], total_epoch=config['warmup_epo'], after_scheduler=scheduler_cosine)
    return scheduler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
 
