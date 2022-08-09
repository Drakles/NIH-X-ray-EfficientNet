import torch
from torch.utils.data import DataLoader
import pandas as pd
from ranger import Ranger

import pytorch_lightning as pl
from monai.metrics import compute_roc_auc

from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, RepeatChannel

from dataset import NihDataset

from efficientnet_pytorch import EfficientNet

pl.seed_everything(42)

class EfficennetLightning(pl.LightningModule):
    num_class = 14

    def __init__(self, train, val, num_channels=3):
        super(EfficennetLightning, self).__init__()

        self.batch_size = 1
        self.num_channels = num_channels

        # data
        self.X_train, self.y_train = train
        self.X_val, self.y_val = val

        #uses sigmoid
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_df = pd.read_csv("metadata/train_df.csv")
        self.test_df = pd.read_csv("metadata/test_df.csv")

        self.model = EfficientNet.from_pretrained('efficientnet-b1',
                                                  num_classes=self.num_class,
                                                  in_channels=self.num_channels)

    def prepare_data(self):
        train_transforms = Compose([
            LoadPNG(image_only=True),
            AddChannel(),
            RepeatChannel(self.num_channels), ## so we can use RGB trained model
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor()
        ])

        val_transforms = Compose([
            LoadPNG(image_only=True),
            AddChannel(),
            RepeatChannel(self.num_channels),  ## so we can use RGB trained model
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
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        #
        # return [optimizer], [scheduler]
        optimizer = Ranger(self.parameters())

        return optimizer

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
        auc = compute_roc_auc(logits, y, average='micro')

        return {"val_loss": loss, "val_auc": auc}

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs["val_loss"] = avg_loss

        avg_val_auc = torch.stack([x[f"val_auc"] for x in outputs]).mean()
        tensorboard_logs["AUC"] = avg_val_auc

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}