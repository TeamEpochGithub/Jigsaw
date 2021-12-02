import random
import string
import os

import numpy as np

import torch

from transformers import AutoTokenizer

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

class Config():

    def __init__(self):

        HASH_NAME = id_generator(size=12)
        
        config = {"seed": 2021,
          "epochs": 3,
          "model_name": "roberta-base",
          "train_batch_size": 32,
          "valid_batch_size": 64,
          "max_length": 128,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "num_classes": 1,
          "margin": 0.5,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "hash_name": HASH_NAME
          }

        config["tokenizer"] = AutoTokenizer.from_pretrained(config['model_name'])
        config['group'] = f'{HASH_NAME}-Baseline'


        set_seed(config['seed'])

        self.config = config



