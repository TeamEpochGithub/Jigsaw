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
        "model.py",
        b'import torch.nn as nn\nfrom transformers import AutoModel\n\n\nclass JigsawModel(nn.Module):\n    """\n        A wrapper around the model being used\n    """\n\n    def __init__(self, model_name, num_classes):\n\n        # Create the model\n        super(JigsawModel, self).__init__()\n        self.model = AutoModel.from_pretrained(model_name)\n\n        # Add a dropout layer\n        self.drop = nn.Dropout(p=0.2)\n\n        # Add a linear output layer\n        self.fc = nn.Linear(768, num_classes)\n\n        # self.relu = nn.ReLU()\n        # self.largeFC = nn.Linear(768, 768)\n\n    def forward(self, ids, mask):\n        """\n            Perform a forward feed\n            :param ids: The input\n            :param mask: The attention mask\n        """\n        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)\n        out = self.drop(out[1])\n        # out = self.largeFC(out)\n        # out = self.relu(out)\n        outputs = self.fc(out)\n        return outputs\n',
    )
    __stickytape_write_module(
        "data.py",
        b'import re\nfrom bs4 import BeautifulSoup\nimport torch\nfrom torch.utils.data import Dataset\n\n\ndef text_cleaning(text):\n    \'\'\'\n    Cleans text into a basic form for NLP. Operations include the following:-\n    1. Remove special charecters like &, #, etc\n    2. Removes extra spaces\n    3. Removes embedded URL links\n    4. Removes HTML tags\n    5. Removes emojis\n\n    text - Text piece to be cleaned.\n    \'\'\'\n    template = re.compile(r\'https?://\\S+|www\\.\\S+\')  # Removes website links\n    text = template.sub(r\'\', text)\n\n    soup = BeautifulSoup(text, \'lxml\')  # Removes HTML tags\n    only_text = soup.get_text()\n    text = only_text\n\n    emoji_pattern = re.compile("["\n                               u"\\U0001F600-\\U0001F64F"  # emoticons\n                               u"\\U0001F300-\\U0001F5FF"  # symbols & pictographs\n                               u"\\U0001F680-\\U0001F6FF"  # transport & map symbols\n                               u"\\U0001F1E0-\\U0001F1FF"  # flags (iOS)\n                               u"\\U00002702-\\U000027B0"\n                               u"\\U000024C2-\\U0001F251"\n                               "]+", flags=re.UNICODE)\n    text = emoji_pattern.sub(r\'\', text)\n\n    text = re.sub(r"[^a-zA-Z\\d]", " ", text)  # Remove special Charecters\n    text = re.sub(\' +\', \' \', text)  # Remove Extra Spaces\n    text = text.strip()  # remove spaces at the beginning and at the end of string\n\n    return text\n\n\nclass JigsawDataset(Dataset):\n    """\n    A class to be passed to a dataloader.\n    """\n\n    def __init__(self, df, tokenizer, max_length):\n        self.df = df\n        self.max_len = max_length\n        self.tokenizer = tokenizer\n        self.text = df["text"].values\n\n    def __len__(self):\n        return len(self.df)\n\n    def __getitem__(self, index):\n        text = text_cleaning(self.text[index])\n        inputs = self.tokenizer.encode_plus(\n            text,\n            truncation=True,\n            add_special_tokens=True,\n            max_length=self.max_len,\n            padding="max_length",\n        )\n\n        # tokens, extracted from sentences\n        ids = inputs["input_ids"]\n        # same length as tokens, 1 if the token is actual, 0 if it is a placeholder to fill max_length\n        mask = inputs["attention_mask"]\n\n        return {\n            "ids": torch.tensor(ids, dtype=torch.long),\n            "mask": torch.tensor(mask, dtype=torch.long),\n        }\n',
    )
    __stickytape_write_module(
        "config.py",
        b'import random\nimport os\n\nimport numpy as np\n\nfrom transformers import AutoTokenizer\nimport torch\n\n\ndef set_seed(seed=42):\n    """\n    Sets the seed of the entire notebook so results are the same every time we run.\n    This is for REPRODUCIBILITY.\n    """\n    np.random.seed(seed)\n    random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed(seed)\n    # Torch backend controls the behavior of various backends that PyTorch supports\n    # When running on the CuDNN backend, two further options must be set\n\n    # only use deterministic convolution algorithms\n    torch.backends.cudnn.deterministic = True\n    # does not try different algorithms to choose the fastest\n    torch.backends.cudnn.benchmark = False\n    # Set a fixed value for the hash seed\n    os.environ["PYTHONHASHSEED"] = str(seed)\n\n\nCONFIG = dict(\n    seed=42,\n    model_name="../input/roberta-base",\n    test_batch_size=64,\n    max_length=128,\n    num_classes=1,\n    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),\n)\n\nset_seed(CONFIG["seed"])\n\nCONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG["model_name"])\n',
    )
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

    COMMON_PATH = "../input/pytorchjigsawstarter/"

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
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            predictions.append(outputs.view(-1).cpu().detach().numpy())

        predictions = np.concatenate(predictions)
        # collect garbage
        gc.collect()

        return predictions

    def inference(
        model_paths: List[str], dataloader: torch.utils.data.DataLoader, device
    ):
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
                model.load_state_dict(
                    torch.load(path, map_location=torch.device("cpu"))
                )

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
