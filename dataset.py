import torch
from torch.utils.data import Dataset

import json

class NihDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __getitem__(self, item):
        record = self.df[["Path", "Label"]].loc[item]

        path = record["Path"]

        label = record["Label"]
        label = json.loads(label)
        label = torch.tensor(label)

        return self.transforms(path), label

    def __len__(self):
        return len(self.df)