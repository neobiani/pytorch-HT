
import torch

def get_config():
    config={

    '_folds':[0,1,2,3,4],

    'image_size' : 256,
    'batch_size' : 6,
    'num_workers' : 4,
    'out_dim' : 1,
    'init_lr' : 3e-3,
    'warmup_factor' : 1,
    'warmup_epo' : 1,
    'n_epochs' : 40,
    'device' : torch.device('cuda:0'),
    'model_type':'slow_r50',
    'SEED' : 2021,
    'Resume':False,
    'kernel_type' : 'slow_r50_256_6bs_40e_0.003_ver2'

    }

    return config