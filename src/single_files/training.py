#!/usr/bin/env python
import contextlib as __stickytape_contextlib


@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil

    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)


with __stickytape_temporary_dir() as __stickytape_working_dir:

    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "wb") as f:
                        f.write(b"\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "wb") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys

    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module(
        "data_loader.py",
        b'import pandas as pd\n\nimport torch\nfrom torch.utils.data import Dataset\nfrom sklearn.model_selection import StratifiedKFold\n\n\ndef get_df(path, config):\n    """\n        Get the data from a given path\n    """\n    # Read data from file\n    df = pd.read_csv(path)\n    df.head()\n\n    # Create multiple data folds based on config\n    skf = StratifiedKFold(\n        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]\n    )\n\n    # Add the fold id to each row\n    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.worker)):\n        df.loc[val_, "kfold"] = int(fold)\n\n    df["kfold"] = df["kfold"].astype(int)\n    df.head()\n\n    return df\n\n\nclass JigsawDataset(Dataset):\n    """\n        A wrapper around the pandas dataframe containing the data\n    """\n\n    def __init__(self, df, tokenizer, max_length):\n        self.df = df\n        self.max_len = max_length\n        self.tokenizer = tokenizer\n        self.more_toxic = df["more_toxic"].values\n        self.less_toxic = df["less_toxic"].values\n\n    def __len__(self):\n        return len(self.df)\n\n    def __getitem__(self, index):\n        more_toxic = self.more_toxic[index]\n        less_toxic = self.less_toxic[index]\n        inputs_more_toxic = self.tokenizer.encode_plus(\n            more_toxic,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n        inputs_less_toxic = self.tokenizer.encode_plus(\n            less_toxic,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n        target = 1\n\n        more_toxic_ids = inputs_more_toxic["input_ids"]\n        more_toxic_mask = inputs_more_toxic["attention_mask"]\n\n        less_toxic_ids = inputs_less_toxic["input_ids"]\n        less_toxic_mask = inputs_less_toxic["attention_mask"]\n\n        return {\n            "more_toxic_ids": torch.tensor(more_toxic_ids, dtype=torch.long),\n            "more_toxic_mask": torch.tensor(more_toxic_mask, dtype=torch.long),\n            "less_toxic_ids": torch.tensor(less_toxic_ids, dtype=torch.long),\n            "less_toxic_mask": torch.tensor(less_toxic_mask, dtype=torch.long),\n            "target": torch.tensor(target, dtype=torch.long),\n        }\n',
    )
    __stickytape_write_module(
        "model.py",
        b'import torch.nn as nn\nfrom transformers import AutoModel\n\n\nclass JigsawModel(nn.Module):\n    """\n        A wrapper around the model being used\n    """\n\n    def __init__(self, model_name, num_classes):\n\n        # Create the model\n        super(JigsawModel, self).__init__()\n        self.model = AutoModel.from_pretrained(model_name)\n\n        # Add a dropout layer\n        self.drop = nn.Dropout(p=0.2)\n\n        # Add a linear output layer\n        self.fc = nn.Linear(768, num_classes)\n\n    def forward(self, ids, mask):\n        """\n            Perform a forward feed\n            :param ids: The input\n            :param mask: The attention mask\n        """\n        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)\n        out = self.drop(out[1])\n        outputs = self.fc(out)\n        return outputs\n',
    )
    __stickytape_write_module(
        "train.py",
        b'import gc\nimport time\nimport copy\nimport os\n\nimport numpy as np\n\nfrom collections import defaultdict\n\nimport torch\nfrom torch.optim import lr_scheduler\nfrom torch.utils.data import DataLoader\nimport torch.nn as nn\n\nfrom tqdm import tqdm\n\nfrom data_loader import JigsawDataset\n\n# For colored terminal text\nfrom colorama import Fore, Back, Style\n\nb_ = Fore.BLUE\ny_ = Fore.YELLOW\nsr_ = Style.RESET_ALL\n\n# Suppress warnings\nimport warnings\n\n\nclass JigsawTrainer:\n    """\n        A class that holds training-specific data (model, config, wandb, dataset)\n        to streamline trainig.\n    """\n\n    def __init__(self, model, config, wandb, run, df):\n        self.model = model\n        self.config = config\n        self.wandb = wandb\n        self.device = config["device"]\n        self.run = run\n        self.df = df\n\n        # For descriptive error messages\n        warnings.filterwarnings("ignore")\n\n        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"\n\n    @torch.no_grad()\n    def valid_one_epoch(self, dataloader, epoch, optimizer):\n        """\n            Evaluate the current model + epoch\n            :param dataloader:\n            :param epoch: The current epoch\n            :param optimizer:\n        """\n        # Set model to eval mode\n        self.model.eval()\n\n        dataset_size = 0\n        running_loss = 0.0\n        epoch_loss = 0\n\n        # Main eval loop\n        bar = tqdm(enumerate(dataloader), total=len(dataloader))\n        for _, data in bar:\n            # Send relevant data to processing device\n            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)\n            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)\n            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)\n            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)\n            targets = data["target"].to(self.device, dtype=torch.long)\n\n            batch_size = more_toxic_ids.size(0)\n\n            # Get result of more/less toxic input\n            more_toxic_outputs = self.model(more_toxic_ids, more_toxic_mask)\n            less_toxic_outputs = self.model(less_toxic_ids, less_toxic_mask)\n\n            # Get loss from output + targets\n            loss = self.criterion(\n                more_toxic_outputs, less_toxic_outputs, targets\n            )\n\n            # Add loss to total loss, weighted by batch size\n            running_loss += loss.item() * batch_size\n            dataset_size += batch_size\n\n            epoch_loss = running_loss / dataset_size\n\n            bar.set_postfix(\n                Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]\n            )\n\n        # Force garbage collection\n        gc.collect()\n\n        return epoch_loss\n\n    def train_one_epoch(self, optimizer, scheduler, dataloader, epoch):\n\n        # Set model to training mode\n        self.model.train()\n\n        dataset_size = 0\n        running_loss = 0.0\n        epoch_loss = 0\n\n        # Main training loop\n        bar = tqdm(enumerate(dataloader), total=len(dataloader))\n        for step, data in bar:\n            # Send relevant data to processing device\n            more_toxic_ids = data["more_toxic_ids"].to(self.device, dtype=torch.long)\n            more_toxic_mask = data["more_toxic_mask"].to(self.device, dtype=torch.long)\n            less_toxic_ids = data["less_toxic_ids"].to(self.device, dtype=torch.long)\n            less_toxic_mask = data["less_toxic_mask"].to(self.device, dtype=torch.long)\n            targets = data["target"].to(self.device, dtype=torch.long)\n\n            batch_size = more_toxic_ids.size(0)\n\n            # Get result of more/less toxic input\n            more_toxic_outputs = self.model(more_toxic_ids, more_toxic_mask)\n            less_toxic_outputs = self.model(less_toxic_ids, less_toxic_mask)\n\n            # Get loss from output + targets\n            loss = self.criterion(more_toxic_outputs, less_toxic_outputs, targets)\n\n            # Use loss for backpropagation\n            loss = loss / self.config["n_accumulate"]\n            loss.backward()\n\n            # Only update the gradients every *n_accumulate* training steps\n            # Otherwise, just store the gradients without updating the weights\n            if (step + 1) % self.config["n_accumulate"] == 0:\n                optimizer.step()\n\n                # zero the parameter gradients\n                optimizer.zero_grad()\n\n                if scheduler is not None:\n                    scheduler.step()\n\n            # Add loss to total loss, weighted by batch size\n            running_loss += loss.item() * batch_size\n            dataset_size += batch_size\n\n            epoch_loss = running_loss / dataset_size\n\n            bar.set_postfix(\n                Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]\n            )\n\n        # Force garbage collection\n        gc.collect()\n\n        return epoch_loss\n\n    def run_training(\n        self, optimizer, scheduler, num_epochs, fold, train_loader, valid_loader\n    ):\n        """\n            This function is responsible for the entire training process.\n            After setup, it repeatedly trains and evaluates the model, for each epoch.\n            :param optimizer:\n            :param scheduler:\n            :param num_epochs: The amount of training + validation epochs\n            :param fold: The data fold to use\n            :param train_loader: The class that loads training data\n            :param train_loader: The class that loads validation data\n        """\n        # To automatically log gradients\n        self.wandb.watch(self.model, log_freq=100)\n\n        if torch.cuda.is_available():\n            print("[INFO] Using GPU: {}\\n".format(torch.cuda.get_device_name()))\n\n        # Setup\n        start = time.time()\n\n        # Save the current state of the model, that can be reused\n        # for future training loops after the result of this loop\n        # is saved\n        best_model_wts = copy.deepcopy(self.model.state_dict())\n        best_epoch_loss = np.inf\n        history = defaultdict(list)\n\n        # Main train + eval loop\n        for epoch in range(1, num_epochs + 1):\n            # Force garbage collection\n            gc.collect()\n\n            # Run a training round\n            train_epoch_loss = self.train_one_epoch(\n                optimizer, scheduler, dataloader=train_loader, epoch=epoch\n            )\n\n            # Run a validation round\n            val_epoch_loss = self.valid_one_epoch(\n                valid_loader, epoch=epoch, optimizer=optimizer\n            )\n\n            history["Train Loss"].append(train_epoch_loss)\n            history["Valid Loss"].append(val_epoch_loss)\n\n            # Log the metrics\n            self.wandb.log({"Train Loss": train_epoch_loss})\n            self.wandb.log({"Valid Loss": val_epoch_loss})\n\n            # deep copy the model\n            if val_epoch_loss <= best_epoch_loss:\n                print(\n                    f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"\n                )\n                best_epoch_loss = val_epoch_loss\n                self.run.summary["Best Loss"] = best_epoch_loss\n                best_model_wts = copy.deepcopy(self.model.state_dict())\n\n                if self.config["dataset_name"]:\n                    dataset_nm = self.config["dataset_name"]\n                    PATH = f"Loss-Fold-{fold}-{dataset_nm}.bin"\n                else:\n                    PATH = f"Loss-Fold-{fold}.bin"\n                torch.save(self.model.state_dict(), PATH)\n                # Save a model file from the current directory\n                print(f"Model Saved{sr_}")\n\n            print()\n\n        end = time.time()\n        time_elapsed = end - start\n        print(\n            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(\n                time_elapsed // 3600,\n                (time_elapsed % 3600) // 60,\n                (time_elapsed % 3600) % 60,\n            )\n        )\n        print("Best Loss: {:.4f}".format(best_epoch_loss))\n\n        # Load best model weights so that future calls of\n        # this function don\'t accidentally reuse the\n        # result of this invocation\n        self.model.load_state_dict(best_model_wts)\n\n        return self.model, history\n\n    def prepare_loaders(self, fold):\n        """\n            Creates data loaders for training and validation, based upon\n            the current fold. This ensures that only data corresponding\n            to the fold is provided.\n            :param fold: The fold\n        """\n\n        # Drop data (not) part of the fold\n        df_train = self.df[self.df.kfold != fold].reset_index(drop=True)\n        df_valid = self.df[self.df.kfold == fold].reset_index(drop=True)\n\n        train_dataset = JigsawDataset(\n            df_train,\n            tokenizer=self.config["tokenizer"],\n            max_length=self.config["max_length"],\n        )\n        valid_dataset = JigsawDataset(\n            df_valid,\n            tokenizer=self.config["tokenizer"],\n            max_length=self.config["max_length"],\n        )\n\n        train_loader = DataLoader(\n            train_dataset,\n            batch_size=self.config["train_batch_size"],\n            num_workers=2,\n            shuffle=True,\n            pin_memory=True,\n            drop_last=True,\n        )\n        valid_loader = DataLoader(\n            valid_dataset,\n            batch_size=self.config["valid_batch_size"],\n            num_workers=2,\n            shuffle=False,\n            pin_memory=True,\n        )\n\n        return train_loader, valid_loader\n\n    def fetch_scheduler(self, optimizer):\n        """\n            Create the correct scheduler for the given optimizer\n            Returns None if the config does not specify the scheduler to be used\n            :param optimizer:\n        """\n        if self.config["scheduler"] == "CosineAnnealingLR":\n            scheduler = lr_scheduler.CosineAnnealingLR(\n                optimizer, T_max=self.config["T_max"], eta_min=self.config["min_lr"]\n            )\n        elif self.config["scheduler"] == "CosineAnnealingWarmRestarts":\n            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(\n                optimizer, T_0=self.config["T_0"], eta_min=self.config["min_lr"]\n            )\n        else:\n            return None\n\n        return scheduler\n\n    def criterion(self, out1, out2, targets):\n        """\n            Evaluate the loss of the given outputs + targets\n            :param out1: The first set of outputs\n            :param out2: The second set of outputs\n            :param targets: The ground truth\n        """\n        return nn.MarginRankingLoss(margin=self.config["margin"])(out1, out2, targets)\n',
    )
    __stickytape_write_module(
        "config.py",
        b'import random\nimport string\nimport os\n\nimport numpy as np\n\nimport torch\n\nfrom transformers import AutoTokenizer\n\n\ndef id_generator(size=12, chars=string.ascii_lowercase + string.digits):\n    return "".join(random.SystemRandom().choice(chars) for _ in range(size))\n\n\ndef set_seed(seed=42):\n    """\n        Sets the seed of the entire notebook so results are the same every time we run.\n        This is for REPRODUCIBILITY.\n    """\n    np.random.seed(seed)\n    random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed(seed)\n    # Torch backend controls the behavior of various backends that PyTorch supports\n    # When running on the CuDNN backend, two further options must be set\n\n    # only use deterministic convolution algorithms\n    torch.backends.cudnn.deterministic = True\n    # does not try different algorithms to choose the fastest\n    torch.backends.cudnn.benchmark = False\n    # Set a fixed value for the hash seed\n    os.environ["PYTHONHASHSEED"] = str(seed)\n\n\nHASH_NAME = id_generator(size=12)\n\nCONFIG = {\n    "seed": 2021,\n    "epochs": 3,\n    "model_name": "roberta-base",\n    "train_batch_size": 32,\n    "valid_batch_size": 64,\n    "max_length": 128,\n    "learning_rate": 1e-4,\n    "scheduler": "CosineAnnealingLR",\n    "min_lr": 1e-6,\n    "T_max": 500,\n    "weight_decay": 1e-6,\n    "n_fold": 5,\n    "n_accumulate": 1,\n    "num_classes": 1,\n    "margin": 0.5,\n    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),\n    "hash_name": HASH_NAME,\n}\n\nCONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG["model_name"])\nCONFIG["group"] = f"{HASH_NAME}-Baseline"\n\nset_seed(CONFIG["seed"])\n',
    )
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
