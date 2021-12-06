import gc
import os

from transformers import AdamW

from data_loader import Data
from config import Config
from model import JigsawModel

from train import run_training, prepare_loaders, fetch_scheduler

DATA_PATH = "./data/jigsaw-toxic-severity-rating/validation_data.csv"
CONFIG = Config().config

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

df = Data(DATA_PATH, CONFIG).df


def main():

    for fold in range(0, CONFIG["n_fold"]):
        print(f"{y_}====== Fold: {fold} ======{sr_}")
        # run = wandb.init(project='Jigsaw',
        #                 config=CONFIG,
        #                 job_type='Train',
        #                 group=CONFIG['group'],
        #                 tags=['roberta-base', f'{HASH_NAME}', 'margin-loss'],
        #                 name=f'{HASH_NAME}-fold-{fold}',
        #                 anonymous='must')

        # Create Dataloaders
        train_loader, valid_loader = prepare_loaders(fold=fold, config=CONFIG, df=df)

        model = JigsawModel(CONFIG["model_name"], CONFIG)
        model.to(CONFIG["device"])

        # Define Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = fetch_scheduler(optimizer, CONFIG)

        model, history = run_training(
            model,
            optimizer,
            scheduler,
            device=CONFIG["device"],
            num_epochs=CONFIG["epochs"],
            fold=fold,
            config=CONFIG,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        # run.finish()

        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()


if __name__ == "__main__":
    main()
