import numpy as np
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.linalg import norm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
import matplotlib.pyplot as plt


MEAN = np.array([.5, .5, .5])
STD = np.array([.5, .5, .5])

# Arjovsky, M., Chintala, S., & Bottou, L. (2017, July).
# Wasserstein generative adversarial networks. In International conference on machine learning (pp. 214-223). PMLR.

# Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
# Improved training of wasserstein gans. arXiv preprint arXiv:1704.00028.


def constraint(module):
    if type(module) == nn.Conv2d:
        module.weight.data = module.weight.data.clamp(-.01, .01)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.G = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.G(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.D = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LayerNorm([64, 32, 32]),
            nn.LeakyReLU(.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LayerNorm([256, 8, 8]),
            nn.LeakyReLU(.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LayerNorm([512, 4, 4]),
            nn.LeakyReLU(.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.D(x)


class WGAN(pl.LightningModule):
    def __init__(self, batch_size=32, n_critic=5, gpc=10):
        super().__init__()

        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gpc = gpc

        self.real_label = th.ones((self.batch_size,), dtype=th.float, device='cuda')
        self.fake_label = th.zeros((self.batch_size,), dtype=th.float, device='cuda')

        self.G = Generator()
        self.D = Discriminator()
        self.loss = nn.BCELoss()

    def configure_optimizers(self):
        gen_opt = th.optim.Adam(self.G.parameters(), betas=(0, .9), lr=1e-4)
        dis_opt = th.optim.Adam(self.D.parameters(), betas=(0, .9), lr=1e-4)  # betas=(0, .9), lr = 1e-4 (i think)
        # gen_opt = th.optim.RMSprop(self.G.parameters(), lr=5e-5)
        # dis_opt = th.optim.RMSprop(self.D.parameters(), lr=5e-5)
        return (
            {'optimizer': dis_opt, 'frequency': self.n_critic},
            {'optimizer': gen_opt, 'frequency': 1}
        )

    def forward(self, z):
        return self.G(z)

    # def on_before_zero_grad(self, *args, **kwargs):
    #     self.D.apply(constraint)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        z = th.randn((self.batch_size, 100, 1, 1), device='cuda')
        if optimizer_idx == 0:

            eps = th.rand((self.batch_size, 1, 1, 1), device='cuda')

            x_bar = self.G(z).detach()
            x_hat = eps * x + (1 - eps) * x_bar  # may not need to detach here
            x_hat.requires_grad = True

            x_hat.grad = None
            out_hat = self.D(x_hat).mean()
            out_hat.backward()  # does out_hat accumulate gradients unwanted?
            grad = x_hat.grad.view(self.batch_size, -1)
            gp = (norm(grad, dim=1, keepdim=True) - 1) ** 2
            penalty = self.gpc * gp

            d_loss = self.D(x_bar) - self.D(x) + penalty

            # f_out = self.D(x_bar).view(-1)
            # r_out = self.D(x).view(-1)
            # f_loss = self.loss(f_out, self.fake_label)
            # r_loss = self.loss(r_out, self.real_label)
            # print(round((r_loss + f_loss).item(), 4))

            print(d_loss.mean().item() - penalty.mean().item(), penalty.mean().item())
            return d_loss.mean()

        if optimizer_idx == 1:

            g_loss = -1 * self.D(self.G(z))
            print(g_loss.mean().item())
            # d_out = self.D(self.G(z)).view(-1)
            # g_loss = self.loss(d_out, self.real_label)
            # print(round(g_loss.item(), 4))
            return g_loss.mean()


    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        celeba = ImageFolder('celeba', transform=transform)
        return DataLoader(celeba, batch_size=self.batch_size, drop_last=True)


if __name__ == '__main__':
    # wgan = WGAN(batch_size=64, n_critic=5, gpc=10)
    #
    # trainer = pl.Trainer(gpus=1, max_epochs=3)
    # trainer.fit(wgan)
    # th.save(wgan, 'WGAN-GP_v0.00')

    wgan = th.load('WGAN-GP_v0.00')

    z = th.randn((4, 100, 1, 1))

    with th.no_grad():
        pred = wgan(z)
        dis = wgan.D(pred)

    img = make_grid(pred).permute(1, 2, 0).numpy() * STD + MEAN
    print(dis)

    plt.imshow(img)
    plt.show()
