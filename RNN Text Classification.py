import numpy as np
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchtext.data import Field
from torchtext.datasets import IMDB
from torchtext.vocab import FastText
from torchtext.data import BucketIterator

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.text_field = Field(sequential=True, fix_length=200)
            self.label_field = Field(sequential=False)

            train, test = IMDB.splits(self.text_field, self.label_field, root='IMDB', train='train', test='test')
            self.text_field.build_vocab(train, test, vectors=FastText('simple'))
            self.label_field.build_vocab(train)

            self.train_iter, self.test_iter = BucketIterator.splits((train, test), batch_size=self.batch_size, device='cuda')

    def train_dataloader(self):
        return self.train_iter

    def test_dataloader(self):
        return self.test_iter


class GRU(pl.LightningModule):
    def __init__(self, embedding, num_layers=1, batch_size=32, input_size=300, hidden_size=100, out_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.GRU = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.cls = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding[x].to(self.device)
        x = self.GRU(x)[1][-1]
        x = self.cls(x)
        return x

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch.text.T, batch.label
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return loss


if __name__ == '__main__':
    imdb = IMDBDataModule(32)
    imdb.setup()

    gru = GRU(imdb.text_field.vocab.vectors, num_layers=2)
    trainer = pl.Trainer(gpus=1, max_epochs=3)
    trainer.fit(gru, datamodule=imdb)

