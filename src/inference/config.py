import random
import os

import numpy as np

from transformers import AutoTokenizer
import torch


def set_seed(seed=42):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Torch backend controls the behavior of various backends that PyTorch supports
    # When running on the CuDNN backend, two further options must be set

    # only use deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True
    # does not try different algorithms to choose the fastest
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


CONFIG = dict(
    seed=42,
    model_name="../input/roberta-base",
    test_batch_size=64,
    max_length=128,
    num_classes=1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

set_seed(CONFIG["seed"])

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG["model_name"])
