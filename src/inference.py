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
        "data.py",
        b'import torch\nfrom torch.utils.data import Dataset\n\n\nclass JigsawDataset(Dataset):\n    def __init__(self, df, tokenizer, max_length):\n        self.df = df\n        self.max_len = max_length\n        self.tokenizer = tokenizer\n        self.text = df["text"].values\n\n    def __len__(self):\n        return len(self.df)\n\n    def __getitem__(self, index):\n        text = self.text[index]\n        inputs = self.tokenizer.encode_plus(\n            text,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n\n        ids = inputs["input_ids"]\n        mask = inputs["attention_mask"]\n\n        return {\n            "ids": torch.tensor(ids, dtype=torch.long),\n            "mask": torch.tensor(mask, dtype=torch.long),\n        }\n',
    )
    __stickytape_write_module(
        "model.py",
        b'import torch.nn as nn\n\nfrom transformers import AutoModel\n\n\nclass JigsawModel(nn.Module):\n    def __init__(self, model_name, config):\n        super(JigsawModel, self).__init__()\n        self.model = AutoModel.from_pretrained(model_name)\n        self.drop = nn.Dropout(p=0.2)\n        self.fc = nn.Linear(768, config["num_classes"])\n\n    def forward(self, ids, mask):\n        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)\n        out = self.drop(out[1])\n        outputs = self.fc(out)\n        return outputs\n',
    )
    __stickytape_write_module(
        "config.py",
        b"import random\nimport os\n\nimport numpy as np\n\nfrom transformers import AutoTokenizer\nimport torch\n\ndef set_seed(seed = 42):\n    '''Sets the seed of the entire notebook so results are the same every time we run.\n    This is for REPRODUCIBILITY.'''\n    np.random.seed(seed)\n    random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed(seed)\n    # When running on the CuDNN backend, two further options must be set\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False\n    # Set a fixed value for the hash seed\n    os.environ['PYTHONHASHSEED'] = str(seed)\n    \nCONFIG = dict(\n    seed=42,\n    model_name=\"../input/roberta-base\",\n    test_batch_size=64,\n    max_length=128,\n    num_classes=1,\n    device=torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\"),\n)\n\nset_seed(CONFIG['seed'])\n\nCONFIG[\"tokenizer\"] = AutoTokenizer.from_pretrained(CONFIG[\"model_name\"])\n",
    )
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
        "../input/pytorch-jigsaw-starter/Loss-Fold-0.bin",
        "../input/pytorch-jigsaw-starter/Loss-Fold-1.bin",
        "../input/pytorch-jigsaw-starter/Loss-Fold-2.bin",
        "../input/pytorch-jigsaw-starter/Loss-Fold-3.bin",
        "../input/pytorch-jigsaw-starter/Loss-Fold-4.bin",
    ]

    DATA_PATH = "../input/jigsaw-toxic-severity-rating/comments_to_score.csv"

    CONFIG = config.CONFIG

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
