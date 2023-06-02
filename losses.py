
import torch.nn as nn

def get_loss_criterion(config): return nn.BCEWithLogitsLoss()