import gc
import os

from transformers import AdamW

from data_loader import get_df
from model import JigsawModel

from train import JigsawTrainer

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


def main(data_path):
    """
        Setup and execute training of the model
    """

    from config import CONFIG, HASH_NAME

    # DATA_PATH = "../input/jigsaw-toxic-severity-rating/validation_data.csv"
    DATA_PATH = data_path

    CONFIG["dataset_name"] = data_path.split("/")[-1].split(".")[0]
    print(f"### Now running with {CONFIG['dataset_name']} ###")
    print("")

    # Load and split the data
    df = get_df(DATA_PATH, CONFIG)

    # Main trainig loop
    for fold in range(0, CONFIG["n_fold"]):

        print(f"{y_}====== Fold: {fold} ======{sr_}")

        # Inform Weights & Biases that this is one iteration
        run = wandb.init(
            project="Jigsaw",
            entity="teamepoch-team1",
            config=CONFIG,
            job_type="Train",
            group=CONFIG["group"],
            tags=["roberta-base", f"{HASH_NAME}", "margin-loss"],
            name=f"{HASH_NAME}-fold-{fold}",
            anonymous="must",
        )

        # Instantiate the model and transfer it to [DEVICE] (e.g. GPU)
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

        # Run the training
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

        # Force python garbage collection to free unused memory and avoid leaks
        _ = gc.collect()
        print()


if __name__ == "__main__":
    dataset_folder = "../input/jigsaw-eda-1/"
    datasets = [
        "1111-2.csv",
        "1111-4.csv",
        "1111-9.csv",
        "2222-2.csv",
        "2222-4.csv",
        "2222-9.csv",
        "3333-2.csv",
        "3333-4.csv",
        "3333-9.csv",
    ]

    for file in datasets:

        main(dataset_folder + file)
