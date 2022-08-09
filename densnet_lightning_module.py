import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

import pytorch_lightning as pl
from torch.optim import lr_scheduler

from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121
from pytorch_lightning.metrics import Accuracy

from dataset import NihDataset


np.random.seed(0)


class Net(pl.LightningModule):
    num_class = 14

    def __init__(self, train, val, batch_size=1024):
        super(Net, self).__init__()

        self.batch_size = batch_size

        # data
        self.X_train, self.y_train = train
        self.X_val, self.y_val = val

        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = {"accuracy": Accuracy()}

        self.train_df = pd.read_csv("metadata/train_df.csv")
        self.test_df = pd.read_csv("metadata/test_df.csv")

        self.model = densenet121(spatial_dims=2, in_channels=1, out_channels=self.num_class)

    def prepare_data(self):
        train_transforms = Compose([
            LoadPNG(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor()
        ])

        val_transforms = Compose([
            LoadPNG(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            ToTensor()
        ])

        self.trainset = NihDataset(self.train_df, train_transforms)

        self.valset = NihDataset(self.test_df, val_transforms)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.criterion(logits, y)

        metrics_dict = {f"val_{name}": metric(logits, y) for name, metric in self.metrics.items()}

        return {**{"val_loss": loss}, **metrics_dict}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {name: torch.stack([x[f"val_{name}"] for x in outputs]).mean()
                            for name, metric in self.metrics.items()}

        tensorboard_logs["val_loss"] = avg_loss

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}