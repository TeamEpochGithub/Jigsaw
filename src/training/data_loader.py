import re
from bs4 import BeautifulSoup
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


def get_df(path, config):
    """
        Get the data from a given path
    """
    # Read data from file
    df = pd.read_csv(path)
    df.head()

    # Create multiple data folds based on config
    skf = StratifiedKFold(
        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]
    )

    # Add the fold id to each row
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.worker)):
        df.loc[val_, "kfold"] = int(fold)

    df["kfold"] = df["kfold"].astype(int)
    df.head()

    return df


def text_cleaning(text):
    """
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    """
    template = re.compile(r"https?://\S+|www\.\S+")  # Removes website links
    text = template.sub(r"", text)

    soup = BeautifulSoup(text, "lxml")  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = text.strip()  # remove spaces at the beginning and at the end of string

    return text


def has_wierd_punctuation(text):
    if "!!" in text or "??" in text or "?!" in text or "!?" in text:
        return 1
    return 0


class JigsawDataset(Dataset):
    """
        A wrapper around the pandas dataframe containing the data
    """

    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df["more_toxic"].values
        self.less_toxic = df["less_toxic"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        more_toxic = text_cleaning(self.more_toxic[index])
        less_toxic = text_cleaning(self.less_toxic[index])
        more_toxic_punct = has_wierd_punctuation(more_toxic)
        less_toxic_punct = has_wierd_punctuation(less_toxic)
        inputs_more_toxic = self.tokenizer.encode_plus(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )
        inputs_less_toxic = self.tokenizer.encode_plus(
            less_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )
        target = 1

        more_toxic_ids = inputs_more_toxic["input_ids"]
        more_toxic_mask = inputs_more_toxic["attention_mask"]

        less_toxic_ids = inputs_less_toxic["input_ids"]
        less_toxic_mask = inputs_less_toxic["attention_mask"]

        return {
            "more_toxic_ids": torch.tensor(more_toxic_ids, dtype=torch.long),
            "more_toxic_mask": torch.tensor(more_toxic_mask, dtype=torch.long),
            "less_toxic_ids": torch.tensor(less_toxic_ids, dtype=torch.long),
            "less_toxic_mask": torch.tensor(less_toxic_mask, dtype=torch.long),
            "more_toxic_wierd_punctuation": torch.tensor(
                more_toxic_punct, dtype=torch.long
            ),
            "less_toxic_wierd_punctuation": torch.tensor(
                less_toxic_punct, dtype=torch.long
            ),
            "target": torch.tensor(target, dtype=torch.long),
        }
