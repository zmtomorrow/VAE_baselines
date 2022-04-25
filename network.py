import torch
import torch.nn as nn
import torch.nn.functional as F



class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class fc_encoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64):
        super(fc_encoder, self).__init__()
        self.latent_channels=latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_channels*2, 1)
        )

    def forward(self, x):
        z=self.encoder(x)
        return z[:,:self.latent_channels,:,:].view(x.size(0),-1),F.softplus(z[:,self.latent_channels:,:,:].view(x.size(0),-1))


class fc_decoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64, out_channels=100):
        super(fc_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d( latent_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, z):
#         print('here',z.size(0))
        return  self.decoder(z.view(z.size(0),-1,8,8))
