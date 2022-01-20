import os
import gc

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import JigsawModel
from data import JigsawDataset
from config import CONFIG
from typing import List

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

COMMON_PATH = '../input/pytorchjigsawstarter/'

# models that will be ensembled
MODEL_PATHS = [
    COMMON_PATH + "Loss-Fold-0.bin",
    COMMON_PATH + "Loss-Fold-1.bin",
    COMMON_PATH + "Loss-Fold-2.bin",
    COMMON_PATH + "Loss-Fold-3.bin",
    COMMON_PATH + "Loss-Fold-4.bin",
]

DATA_PATH = "../input/jigsaw-toxic-severity-rating/comments_to_score.csv"


@torch.no_grad()
def predict(model: torch.nn.Module, dataloader, device):
    """
    predicts for one given model
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    model.eval()
    predictions = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        predictions.append(outputs.view(-1).cpu().detach().numpy())

    predictions = np.concatenate(predictions)
    # collect garbage
    gc.collect()

    return predictions


def inference(model_paths: List[str], dataloader: torch.utils.data.DataLoader, device):
    """
    goes through model checkpoints and evaluates all comments on each
    :param model_paths: list of model paths
    :param dataloader:
    :param device:
    :return: predictions as np array
    """
    final_predictions = []
    for i, path in enumerate(model_paths):
        # initialize model
        model = JigsawModel(CONFIG["model_name"], CONFIG["num_classes"])
        model.to(CONFIG["device"])
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

        print(f"Getting predictions for model {i+1}")
        preds = predict(model, dataloader, device)
        final_predictions.append(preds)

    # final predictions is now of shape [len(model_paths), len(data)]
    # so we need to ensemble the results
    final_predictions = np.array(final_predictions)
    final_predictions = np.mean(final_predictions, axis=0)
    return final_predictions


def main():
    """
    generates submission.csv
    :return:
    """
    # prepare dataloader
    df = pd.read_csv(DATA_PATH)
    test_dataset = JigsawDataset(
        df, CONFIG["tokenizer"], max_length=CONFIG["max_length"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["test_batch_size"],
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    preds = inference(MODEL_PATHS, test_loader, CONFIG["device"])

    print(f"Total Predictions: {preds.shape[0]}")
    print(f"Total Unique Predictions: {np.unique(preds).shape[0]}")

    # *** Visualize output ***
    df["score"] = preds
    df.head()

    # kaggle evaluates on rank, but we predicted toxicity as a score, so we need to convert it
    df["score"] = df["score"].rank(method="first")
    df.head()

    # *** Write submission predictions to file ***
    df.drop("text", axis=1, inplace=True)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
