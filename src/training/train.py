import gc
import time
import copy
import os

import numpy as np

from collections import defaultdict

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

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
    """
        A class that holds training-specific data (model, config, wandb, dataset)
        to streamline trainig.
    """

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
        """
            Evaluate the current model + epoch
            :param dataloader:
            :param epoch: The current epoch
            :param optimizer:
        """
        # Set model to eval mode
        self.model.eval()

        dataset_size = 0
        running_loss = 0.0
        epoch_loss = 0

        # Main eval loop
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, data in bar:
            # Send relevant data to processing device
            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)
            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)
            more_toxic_punctuation = data["more_toxic_wierd_punctuation"].to(
                self.device, dtype=torch.long
            )
            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)
            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)
            less_toxic_punctuation = data["less_toxic_wierd_punctuation"].to(
                self.device, dtype=torch.long
            )
            targets = data["target"].to(self.device, dtype=torch.long)

            batch_size = more_toxic_ids.size(0)

            # Get result of more/less toxic input
            more_toxic_outputs = self.model(
                more_toxic_ids, more_toxic_mask, more_toxic_punctuation
            )
            less_toxic_outputs = self.model(
                less_toxic_ids, less_toxic_mask, less_toxic_punctuation
            )

            # Get loss from output + targets
            loss = self.criterion(more_toxic_outputs, less_toxic_outputs, targets)

            # Add loss to total loss, weighted by batch size
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )

        # Force garbage collection
        gc.collect()

        return epoch_loss

    def train_one_epoch(self, optimizer, scheduler, dataloader, epoch):

        # Set model to training mode
        self.model.train()

        dataset_size = 0
        running_loss = 0.0
        epoch_loss = 0

        # Main training loop
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            # Send relevant data to processing device
            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)
            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)
            more_toxic_punctuation = data["more_toxic_wierd_punctuation"].to(
                self.device, dtype=torch.long
            )
            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)
            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)
            less_toxic_punctuation = data["less_toxic_wierd_punctuation"].to(
                self.device, dtype=torch.long
            )
            targets = data["target"].to(self.device, dtype=torch.long)

            batch_size = more_toxic_ids.size(0)

            # Get result of more/less toxic input
            more_toxic_outputs = self.model(
                more_toxic_ids, more_toxic_mask, more_toxic_punctuation
            )
            less_toxic_outputs = self.model(
                less_toxic_ids, less_toxic_mask, less_toxic_punctuation
            )

            # Get loss from output + targets
            loss = self.criterion(more_toxic_outputs, less_toxic_outputs, targets)

            # Use loss for backpropagation
            loss = loss / self.config["n_accumulate"]
            loss.backward()

            # Only update the gradients every *n_accumulate* training steps
            # Otherwise, just store the gradients without updating the weights
            if (step + 1) % self.config["n_accumulate"] == 0:
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            # Add loss to total loss, weighted by batch size
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )

        # Force garbage collection
        gc.collect()

        return epoch_loss

    def run_training(
        self, optimizer, scheduler, num_epochs, fold, train_loader, valid_loader
    ):
        """
            This function is responsible for the entire training process.
            After setup, it repeatedly trains and evaluates the model, for each epoch.
            :param optimizer:
            :param scheduler:
            :param num_epochs: The amount of training + validation epochs
            :param fold: The data fold to use
            :param train_loader: The class that loads training data
            :param train_loader: The class that loads validation data
        """
        # To automatically log gradients
        self.wandb.watch(self.model, log_freq=100)

        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Setup
        start = time.time()

        # Save the current state of the model, that can be reused
        # for future training loops after the result of this loop
        # is saved
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch_loss = np.inf
        history = defaultdict(list)

        # Main train + eval loop
        for epoch in range(1, num_epochs + 1):
            # Force garbage collection
            gc.collect()

            # Run a training round
            train_epoch_loss = self.train_one_epoch(
                optimizer, scheduler, dataloader=train_loader, epoch=epoch
            )

            # Run a validation round
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

        # Load best model weights so that future calls of
        # this function don't accidentally reuse the
        # result of this invocation
        self.model.load_state_dict(best_model_wts)

        return self.model, history

    def prepare_loaders(self, fold):
        """
            Creates data loaders for training and validation, based upon
            the current fold. This ensures that only data corresponding
            to the fold is provided.
            :param fold: The fold
        """

        # Drop data (not) part of the fold
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
        """
            Create the correct scheduler for the given optimizer
            Returns None if the config does not specify the scheduler to be used
            :param optimizer:
        """
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
        """
            Evaluate the loss of the given outputs + targets
            :param out1: The first set of outputs
            :param out2: The second set of outputs
            :param targets: The ground truth
        """
        return nn.MarginRankingLoss(margin=self.config["margin"])(out1, out2, targets)
