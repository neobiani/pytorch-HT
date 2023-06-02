import random

import torch

from config import get_config
from trainer import create_trainer
from utils import seed_everything

def main():
    # Load and log experiment configuration
    config = get_config()

    seed_everything(config['SEED'])

    # create trainer
    #trainer = create_trainer(config)
    # Start training
    for fold in config['_folds']:
        create_trainer(fold, config)
        


if __name__ == '__main__':
    main()