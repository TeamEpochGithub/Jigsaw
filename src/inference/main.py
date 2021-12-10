import os
import gc

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import JigsawDataset
from model import JigsawModel
import config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_PATHS = [
    "../input/pytorch-w-b-jigsaw-starter/Loss-Fold-0.bin",
    "../input/pytorch-w-b-jigsaw-starter/Loss-Fold-1.bin",
    "../input/pytorch-w-b-jigsaw-starter/Loss-Fold-2.bin",
    "../input/pytorch-w-b-jigsaw-starter/Loss-Fold-3.bin",
    "../input/pytorch-w-b-jigsaw-starter/Loss-Fold-4.bin",
]

DATA_PATH = "../input/jigsaw-toxic-severity-rating/comments_to_score.csv"

CONFIG = config.CONFIG

df = pd.read_csv(DATA_PATH)

test_dataset = JigsawDataset(df, CONFIG["tokenizer"], max_length=CONFIG["max_length"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["test_batch_size"],
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, data in bar:
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy())

    PREDS = np.concatenate(PREDS)
    gc.collect()

    return PREDS


def inference(model_paths, dataloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = JigsawModel(CONFIG["model_name"], CONFIG)
        model.to(CONFIG["device"])
        model.load_state_dict(torch.load(path))

        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


preds = inference(MODEL_PATHS, test_loader, CONFIG["device"])

print(f"Total Predictiions: {preds.shape[0]}")
print(f"Total Unique Predictions: {np.unique(preds).shape[0]}")

# *** Visualize output ***

df["score"] = preds
df.head()

df["score"] = df["score"].rank(method="first")
df.head()

# *** Write submission predictions to file ***

df.drop("text", axis=1, inplace=True)
df.to_csv("submission.csv", index=False)
