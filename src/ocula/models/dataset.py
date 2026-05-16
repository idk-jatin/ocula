import torch
from torch.utils.data import Dataset
import pandas as pd


class OculaDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=128,
        text_col="text",
        label_col="label_int",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[text_col, label_col])

        self.texts = df[text_col].to_list()
        self.labels = df[label_col].astype(int).to_list()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
    
    def get_label_counts(self):
        from collections import Counter
        return dict(Counter(self.labels))
