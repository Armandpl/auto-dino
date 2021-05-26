import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dino_dataset import DinoDataset


class ActionClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


def main():
    config = dict(
        train_pct = 0.8,
        learning_rate=1e-5,
        batch_size=64
    )

    with wandb.init(project="auto-dino", config=config, job_type="train") as run:
        config = run.config

        artifact = run.use_artifact("imitation_dataset:latest")
        artifact_dir = artifact.download()

        dataset = DinoDataset(artifact_dir)

        train_len = int(len(dataset)*config.train_pct)
        val_len = len(dataset)-train_len

        train, val = random_split(dataset, [train_len, val_len])

        action_classifier = ActionClassifier(config)

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(logger=wandb_logger, gpus=1)
        trainer.fit(
            action_classifier,
            DataLoader(train, batch_size=config.batch_size),
            DataLoader(val, batch_size=config.batch_size),
        )

if __name__ == "__main__":
    main()
