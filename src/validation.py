import torch.nn as nn

def criterion(outputs1, outputs2, targets, config):
    return nn.MarginRankingLoss(margin=config['margin'])(outputs1, outputs2, targets)
