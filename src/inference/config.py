from transformers import AutoTokenizer
import torch

CONFIG = dict(
    seed=42,
    model_name="../input/roberta-base",
    test_batch_size=64,
    max_length=128,
    num_classes=1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG["model_name"])
