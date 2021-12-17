import gc
import time
import copy
import os

import numpy as np

from collections import defaultdict

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loader import JigsawDataset

# For colored terminal text
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings


class JigsawTrainer:
    def __init__(self, model, config, wandb, run, df):
        self.model = model
        self.config = config
        self.wandb = wandb
        self.device = config["device"]
        self.run = run
        self.df = df

        # For descriptive error messages
        warnings.filterwarnings("ignore")

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    @torch.no_grad()
    def valid_one_epoch(self, dataloader, epoch, optimizer):
        self.model.eval()

        dataset_size = 0
        running_loss = 0.0
        epoch_loss = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, data in bar:
            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)
            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)
            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)
            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)
            targets = data["target"].to(self.device, dtype=torch.long)

            batch_size = more_toxic_ids.size(0)

            more_toxic_outputs = self.model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = self.model(less_toxic_ids, less_toxic_mask)

            loss = self.criterion(
                more_toxic_outputs, less_toxic_outputs, targets, self.config
            )

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )

        gc.collect()

        return epoch_loss

    def train_one_epoch(self, optimizer, scheduler, dataloader, epoch):
        self.model.train()

        dataset_size = 0
        running_loss = 0.0
        epoch_loss = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)
            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)
            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)
            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)
            targets = data["target"].to(self.device, dtype=torch.long)

            batch_size = more_toxic_ids.size(0)

            more_toxic_outputs = self.model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = self.model(less_toxic_ids, less_toxic_mask)

            loss = self.criterion(
                more_toxic_outputs, less_toxic_outputs, targets, self.config
            )
            loss = loss / self.config["n_accumulate"]
            loss.backward()

            if (step + 1) % self.config["n_accumulate"] == 0:
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )
        gc.collect()

        return epoch_loss

    def run_training(
        self, optimizer, scheduler, num_epochs, fold, train_loader, valid_loader
    ):
        # To automatically log gradients
        self.wandb.watch(self.model, log_freq=100)

        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch_loss = np.inf
        history = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            gc.collect()
            train_epoch_loss = self.train_one_epoch(
                optimizer, scheduler, dataloader=train_loader, epoch=epoch
            )

            val_epoch_loss = self.valid_one_epoch(
                valid_loader, epoch=epoch, optimizer=optimizer
            )

            history["Train Loss"].append(train_epoch_loss)
            history["Valid Loss"].append(val_epoch_loss)

            # Log the metrics
            self.wandb.log({"Train Loss": train_epoch_loss})
            self.wandb.log({"Valid Loss": val_epoch_loss})

            # deep copy the model
            if val_epoch_loss <= best_epoch_loss:
                print(
                    f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
                )
                best_epoch_loss = val_epoch_loss
                self.run.summary["Best Loss"] = best_epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                PATH = f"Loss-Fold-{fold}.bin"
                torch.save(self.model.state_dict(), PATH)
                # Save a model file from the current directory
                print(f"Model Saved{sr_}")

            print()

        end = time.time()
        time_elapsed = end - start
        print(
            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600,
                (time_elapsed % 3600) // 60,
                (time_elapsed % 3600) % 60,
            )
        )
        print("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        return self.model, history

    def prepare_loaders(self, fold):
        df_train = self.df[self.df.kfold != fold].reset_index(drop=True)
        df_valid = self.df[self.df.kfold == fold].reset_index(drop=True)

        train_dataset = JigsawDataset(
            df_train,
            tokenizer=self.config["tokenizer"],
            max_length=self.config["max_length"],
        )
        valid_dataset = JigsawDataset(
            df_valid,
            tokenizer=self.config["tokenizer"],
            max_length=self.config["max_length"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["train_batch_size"],
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config["valid_batch_size"],
            num_workers=2,
            shuffle=False,
            pin_memory=True,
        )

        return train_loader, valid_loader

    def fetch_scheduler(self, optimizer):
        if self.config["scheduler"] == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["T_max"], eta_min=self.config["min_lr"]
            )
        elif self.config["scheduler"] == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config["T_0"], eta_min=self.config["min_lr"]
            )
        else:
            return None

        return scheduler

    def criterion(self, out1, out2, targets):
        return nn.MarginRankingLoss(margin=self.config["margin"])(out1, out2, targets)
