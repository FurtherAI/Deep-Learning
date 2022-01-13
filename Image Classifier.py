import numpy as np
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization as norm

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# He, K., Zhang, X., Ren, S., & Sun, J. (2016).
# Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).


def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride = 2 if self.in_channels != self.out_channels else 1

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, stride=self.stride),
            nn.ReLU(),
            conv_bn(self.out_channels, self.out_channels, stride=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = nn.ReLU()(x)
        return x

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()

        self.blocks = nn.Sequential(
            ResBlock(in_channels, out_channels),
            *[ResBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNet(pl.LightningModule):
    def __init__(self, num_classes, num_blocks=[2, 2, 2], layer_channels=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.gate = nn.Sequential(
            nn.Conv2d(3, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(layer_channels[0]),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            *[ResNetLayer(in_channels, out_channels, blocks) for in_channels, out_channels, blocks in zip(layer_channels[:-1], layer_channels[1:], num_blocks)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(layer_channels[-1], num_classes),
            nn.LogSoftmax()
        )


    def configure_optimizers(self):
        return th.optim.Adam(self.parameters())

    def forward(self, x):
        x = self.gate(x)
        x = self.layers(x)
        x = self.decoder(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        log = self.forward(x)
        loss = F.nll_loss(log, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log = self.forward(x)
        loss = F.nll_loss(log, y)
        return loss

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28),
            transforms.ColorJitter(.3, .3, .3),
            transforms.ToTensor(),
            transforms.Normalize(norm().mean, norm().std)
        ])
        cifar10 = CIFAR10('cifar_10', train=True, download=False, transform=transform)
        return DataLoader(cifar10, batch_size=32)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm().mean, norm().std)])
        cifar10 = CIFAR10('cifar_10', train=False, download=False, transform=transform)
        return DataLoader(cifar10, batch_size=32)


if __name__ == '__main__':
    resnet = ResNet(10, num_blocks=[1, 1, 1])
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(resnet)
    th.save(resnet, 'ResNet_v0.00')
