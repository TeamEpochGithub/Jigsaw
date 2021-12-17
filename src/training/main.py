import gc
import os

from transformers import AdamW

from data_loader import get_df
from config import CONFIG, HASH_NAME
from model import JigsawModel

from train import JigsawTrainer

DATA_PATH = "../input/jigsaw-toxic-severity-rating/validation_data.csv"

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# *** Weights and Biases Setup ***
import wandb

try:
    api_key = os.environ["WANDB_API"]
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print("Unable to load Weight's and Biases")

df = get_df(DATA_PATH, CONFIG)


def main():

    for fold in range(0, CONFIG["n_fold"]):
        print(f"{y_}====== Fold: {fold} ======{sr_}")
        run = wandb.init(
            project="Jigsaw",
            config=CONFIG,
            job_type="Train",
            group=CONFIG["group"],
            tags=["roberta-base", f"{HASH_NAME}", "margin-loss"],
            name=f"{HASH_NAME}-fold-{fold}",
            anonymous="must",
        )

        model = JigsawModel(CONFIG["model_name"], CONFIG["num_classes"])
        model.to(CONFIG["device"])

        # Create trainer instance
        trainer = JigsawTrainer(model, CONFIG, wandb, run, df)

        # Create Dataloaders
        train_loader, valid_loader = trainer.prepare_loaders(fold)

        # Define Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = trainer.fetch_scheduler(optimizer)

        model, history = trainer.run_training(
            optimizer,
            scheduler,
            num_epochs=CONFIG["epochs"],
            fold=fold,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        run.finish()

        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()


if __name__ == "__main__":
    main()
