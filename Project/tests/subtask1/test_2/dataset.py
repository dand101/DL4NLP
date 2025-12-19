import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="text", label_col=None, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        item = {k: torch.tensor(v) for k, v in enc.items()}

        if self.label_col is not None and self.label_col in self.df.columns:
            item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return item
