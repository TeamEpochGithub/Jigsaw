import torch
from torch.utils.data import Dataset


class JigsawDataset(Dataset):
    """
    A class to be passed to a dataloader.
    """

    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df["text"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )

        # tokens, extracted from sentences
        ids = inputs["input_ids"]
        # same length as tokens, 1 if the token is actual, 0 if it is a placeholder to fill max_length
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }
