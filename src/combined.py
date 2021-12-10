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
        b'import pandas as pd\n\nimport torch\nfrom torch.utils.data import Dataset\nfrom sklearn.model_selection import StratifiedKFold\n\n\nclass Data:\n    def __init__(self, path, config):\n        df = pd.read_csv(path)\n        df.head()\n\n        skf = StratifiedKFold(\n            n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]\n        )\n\n        for fold, (_, val_) in enumerate(skf.split(X=df, y=df.worker)):\n            df.loc[val_, "kfold"] = int(fold)\n\n        df["kfold"] = df["kfold"].astype(int)\n        df.head()\n\n        self.df = df\n\n\nclass JigsawDataset(Dataset):\n    def __init__(self, df, tokenizer, max_length, config):\n        self.df = df\n        self.max_len = max_length\n        self.tokenizer = tokenizer\n        self.more_toxic = df["more_toxic"].values\n        self.less_toxic = df["less_toxic"].values\n\n    def __len__(self):\n        return len(self.df)\n\n    def __getitem__(self, index):\n        more_toxic = self.more_toxic[index]\n        less_toxic = self.less_toxic[index]\n        inputs_more_toxic = self.tokenizer.encode_plus(\n            more_toxic,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n        inputs_less_toxic = self.tokenizer.encode_plus(\n            less_toxic,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n        target = 1\n\n        more_toxic_ids = inputs_more_toxic["input_ids"]\n        more_toxic_mask = inputs_more_toxic["attention_mask"]\n\n        less_toxic_ids = inputs_less_toxic["input_ids"]\n        less_toxic_mask = inputs_less_toxic["attention_mask"]\n\n        return {\n            "more_toxic_ids": torch.tensor(more_toxic_ids, dtype=torch.long),\n            "more_toxic_mask": torch.tensor(more_toxic_mask, dtype=torch.long),\n            "less_toxic_ids": torch.tensor(less_toxic_ids, dtype=torch.long),\n            "less_toxic_mask": torch.tensor(less_toxic_mask, dtype=torch.long),\n            "target": torch.tensor(target, dtype=torch.long),\n        }\n',
    )
    __stickytape_write_module(
        "config.py",
        b'import random\nimport string\nimport os\n\nimport numpy as np\n\nimport torch\n\nfrom transformers import AutoTokenizer\n\n\ndef id_generator(size=12, chars=string.ascii_lowercase + string.digits):\n    return "".join(random.SystemRandom().choice(chars) for _ in range(size))\n\n\ndef set_seed(seed=42):\n    """Sets the seed of the entire notebook so results are the same every time we run.\n    This is for REPRODUCIBILITY."""\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed(seed)\n    # When running on the CuDNN backend, two further options must be set\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False\n    # Set a fixed value for the hash seed\n    os.environ["PYTHONHASHSEED"] = str(seed)\n\n\nclass Config:\n    def __init__(self):\n\n        HASH_NAME = id_generator(size=12)\n\n        config = {\n            "seed": 2021,\n            "epochs": 3,\n            "model_name": "roberta-base",\n            "train_batch_size": 32,\n            "valid_batch_size": 64,\n            "max_length": 128,\n            "learning_rate": 1e-4,\n            "scheduler": "CosineAnnealingLR",\n            "min_lr": 1e-6,\n            "T_max": 500,\n            "weight_decay": 1e-6,\n            "n_fold": 5,\n            "n_accumulate": 1,\n            "num_classes": 1,\n            "margin": 0.5,\n            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),\n            "hash_name": HASH_NAME,\n        }\n\n        config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_name"])\n        config["group"] = f"{HASH_NAME}-Baseline"\n\n        set_seed(config["seed"])\n\n        self.config = config\n',
    )
    __stickytape_write_module(
        "model.py",
        b'import torch.nn as nn\nfrom transformers import AutoModel\n\n\nclass JigsawModel(nn.Module):\n    def __init__(self, model_name, config):\n        super(JigsawModel, self).__init__()\n        self.model = AutoModel.from_pretrained(model_name)\n        self.drop = nn.Dropout(p=0.2)\n        self.fc = nn.Linear(768, config["num_classes"])\n\n    def forward(self, ids, mask):\n        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)\n        out = self.drop(out[1])\n        outputs = self.fc(out)\n        return outputs\n',
    )
    __stickytape_write_module(
        "train.py",
        b'import gc\nimport time\nimport copy\nimport os\n\nimport numpy as np\n\nfrom collections import defaultdict\n\nimport torch\nfrom torch.optim import lr_scheduler\nfrom torch.utils.data import DataLoader\n\nfrom tqdm import tqdm\n\nfrom validation import criterion\nfrom data_loader import JigsawDataset\n\n# For colored terminal text\nfrom colorama import Fore, Back, Style\n\nb_ = Fore.BLUE\ny_ = Fore.YELLOW\nsr_ = Style.RESET_ALL\n\n# Suppress warnings\nimport warnings\n\nwarnings.filterwarnings("ignore")\n\n# For descriptive error messages\nos.environ["CUDA_LAUNCH_BLOCKING"] = "1"\n\n@torch.no_grad()\ndef valid_one_epoch(model, dataloader, device, epoch, config, optimizer):\n    model.eval()\n\n    dataset_size = 0\n    running_loss = 0.0\n    epoch_loss = 0\n\n    bar = tqdm(enumerate(dataloader), total=len(dataloader))\n    for _, data in bar:\n        more_toxic_ids = data["more_toxic_ids"].to(device, dtype=torch.long)\n        more_toxic_mask = data["more_toxic_mask"].to(device, dtype=torch.long)\n        less_toxic_ids = data["less_toxic_ids"].to(device, dtype=torch.long)\n        less_toxic_mask = data["less_toxic_mask"].to(device, dtype=torch.long)\n        targets = data["target"].to(device, dtype=torch.long)\n\n        batch_size = more_toxic_ids.size(0)\n\n        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)\n        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)\n\n        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets, config)\n\n        running_loss += loss.item() * batch_size\n        dataset_size += batch_size\n\n        epoch_loss = running_loss / dataset_size\n\n        bar.set_postfix(\n            Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]\n        )\n\n    gc.collect()\n\n    return epoch_loss\n\n\ndef train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, config):\n    model.train()\n\n    dataset_size = 0\n    running_loss = 0.0\n    epoch_loss = 0\n\n    bar = tqdm(enumerate(dataloader), total=len(dataloader))\n    for step, data in bar:\n        more_toxic_ids = data["more_toxic_ids"].to(device, dtype=torch.long)\n        more_toxic_mask = data["more_toxic_mask"].to(device, dtype=torch.long)\n        less_toxic_ids = data["less_toxic_ids"].to(device, dtype=torch.long)\n        less_toxic_mask = data["less_toxic_mask"].to(device, dtype=torch.long)\n        targets = data["target"].to(device, dtype=torch.long)\n\n        batch_size = more_toxic_ids.size(0)\n\n        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)\n        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)\n\n        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets, config)\n        loss = loss / config["n_accumulate"]\n        loss.backward()\n\n        if (step + 1) % config["n_accumulate"] == 0:\n            optimizer.step()\n\n            # zero the parameter gradients\n            optimizer.zero_grad()\n\n            if scheduler is not None:\n                scheduler.step()\n\n        running_loss += loss.item() * batch_size\n        dataset_size += batch_size\n\n        epoch_loss = running_loss / dataset_size\n\n        bar.set_postfix(\n            Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]\n        )\n    gc.collect()\n\n    return epoch_loss\n\n\ndef run_training(\n    model,\n    optimizer,\n    scheduler,\n    device,\n    num_epochs,\n    fold,\n    config,\n    train_loader,\n    valid_loader,\n):\n    # To automatically log gradients\n    # wandb.watch(model, log_freq=100)\n\n    if torch.cuda.is_available():\n        print("[INFO] Using GPU: {}\\n".format(torch.cuda.get_device_name()))\n\n    start = time.time()\n    best_model_wts = copy.deepcopy(model.state_dict())\n    best_epoch_loss = np.inf\n    history = defaultdict(list)\n\n    for epoch in range(1, num_epochs + 1):\n        gc.collect()\n        train_epoch_loss = train_one_epoch(\n            model,\n            optimizer,\n            scheduler,\n            dataloader=train_loader,\n            device=config["device"],\n            epoch=epoch,\n            config=config,\n        )\n\n        val_epoch_loss = valid_one_epoch(\n            model, valid_loader, device=config["device"], epoch=epoch, config=config, optimizer=optimizer\n        )\n\n        history["Train Loss"].append(train_epoch_loss)\n        history["Valid Loss"].append(val_epoch_loss)\n\n        # Log the metrics\n        # wandb.log({"Train Loss": train_epoch_loss})\n        # wandb.log({"Valid Loss": val_epoch_loss})\n\n        # deep copy the model\n        if val_epoch_loss <= best_epoch_loss:\n            print(\n                f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"\n            )\n            best_epoch_loss = val_epoch_loss\n            # run.summary["Best Loss"] = best_epoch_loss\n            best_model_wts = copy.deepcopy(model.state_dict())\n            PATH = f"Loss-Fold-{fold}.bin"\n            torch.save(model.state_dict(), PATH)\n            # Save a model file from the current directory\n            print(f"Model Saved{sr_}")\n\n        print()\n\n    end = time.time()\n    time_elapsed = end - start\n    print(\n        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(\n            time_elapsed // 3600,\n            (time_elapsed % 3600) // 60,\n            (time_elapsed % 3600) % 60,\n        )\n    )\n    print("Best Loss: {:.4f}".format(best_epoch_loss))\n\n    # load best model weights\n    model.load_state_dict(best_model_wts)\n\n    return model, history\n\n\ndef prepare_loaders(fold, config, df):\n    df_train = df[df.kfold != fold].reset_index(drop=True)\n    df_valid = df[df.kfold == fold].reset_index(drop=True)\n\n    train_dataset = JigsawDataset(\n        df_train,\n        tokenizer=config["tokenizer"],\n        max_length=config["max_length"],\n        config=config,\n    )\n    valid_dataset = JigsawDataset(\n        df_valid,\n        tokenizer=config["tokenizer"],\n        max_length=config["max_length"],\n        config=config,\n    )\n\n    train_loader = DataLoader(\n        train_dataset,\n        batch_size=config["train_batch_size"],\n        num_workers=2,\n        shuffle=True,\n        pin_memory=True,\n        drop_last=True,\n    )\n    valid_loader = DataLoader(\n        valid_dataset,\n        batch_size=config["valid_batch_size"],\n        num_workers=2,\n        shuffle=False,\n        pin_memory=True,\n    )\n\n    return train_loader, valid_loader\n\n\ndef fetch_scheduler(optimizer, config):\n    if config["scheduler"] == "CosineAnnealingLR":\n        scheduler = lr_scheduler.CosineAnnealingLR(\n            optimizer, T_max=config["T_max"], eta_min=config["min_lr"]\n        )\n    elif config["scheduler"] == "CosineAnnealingWarmRestarts":\n        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(\n            optimizer, T_0=config["T_0"], eta_min=config["min_lr"]\n        )\n    elif config["scheduler"] == None:\n        return None\n\n    return scheduler\n',
    )
    __stickytape_write_module(
        "validation.py",
        b'import torch.nn as nn\n\n\ndef criterion(outputs1, outputs2, targets, config):\n    return nn.MarginRankingLoss(margin=config["margin"])(outputs1, outputs2, targets)\n',
    )
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
            train_loader, valid_loader = prepare_loaders(
                fold=fold, config=CONFIG, df=df
            )

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
