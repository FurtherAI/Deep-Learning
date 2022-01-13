import torch as th
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow, figure
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from pl_bolts.transforms.dataset_normalizations import cifar10_normalization as norm

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

# Kingma, D. P., & Welling, M. (2013).
# Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

# He, K., Zhang, X., Ren, S., & Sun, J. (2016).
# Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=512, input_height=32):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(32, 8, 3, padding=1)
        self.enc_conv5 = nn.Conv2d(8, 4, 3, padding=1)
        self.enc_fc = nn.Linear(4096, 512)
        self.flatten = nn.Flatten()

        self.dec_conv1 = nn.ConvTranspose2d(4, 8, 3, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(8, 32, 3, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 64, 3, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(64, 16, 3, padding=1)
        self.dec_conv5 = nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.dec_fc = nn.Linear(512, 4096)

        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        self.log_scale = nn.Parameter(th.Tensor([0.0]))

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = F.relu(self.enc_conv5(x))

        x = self.flatten(x)
        x = self.enc_fc(x)
        return x

    def decode(self, x):
        x = self.dec_fc(x)
        x = x.view(-1, 4, 32, 32)

        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        x = th.tanh(self.dec_conv5(x))
        return x

    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encode(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = th.exp(log_var / 2)
        q = th.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decode(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        return recon_loss, kl


    def configure_optimizers(self, ):
        return th.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = th.exp(logscale)
        mean = x_hat
        dist = th.distributions.Normal(mean, scale)

        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        p = th.distributions.Normal(th.zeros_like(mu), th.ones_like(std))
        q = th.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # elbo

        losses = self.forward(x)

        elbo = (losses[1] - losses[0])
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': losses[1].mean(),
            'recon_loss': losses[0].mean(),
            'reconstruction': losses[0].mean()
        })

        return elbo

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        cifar_10 = CIFAR10('cifar_10', train=True, download=True, transform=transform)
        return DataLoader(cifar_10, batch_size=32)

if __name__ == '__main__':
    # vae = VAE()
    # trainer = pl.Trainer(gpus=1, max_epochs=50)
    # trainer.fit(vae)
    # th.save(vae, 'VAE_v0.00')
    vae = th.load('VAE_v0.00')

    figure(figsize=(20, 20))

    p = th.distributions.Normal(th.zeros((1, 512)), th.ones((1, 512)))
    z = p.rsample((4,))

    with th.no_grad():
        pred = vae.decode(z.to(vae.device)).cpu()

    mean = np.array(norm().mean)
    std = np.array(norm().std)

    img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

    imshow(img)
    plt.show()

