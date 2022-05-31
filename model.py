import torch
import torch.nn as nn
from network import *
from distributions import *
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import torch.nn.functional as F
from utils import batch_KL_diag_gaussian_std
eps = 1e-7


class VAE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.z_dim = opt['z_channels'] * 64
        self.device = opt['device']
        if opt['data_set'] == 'MNIST' or opt['data_set'] == 'BinaryMNIST':
            input_channels = 1
        else:
            input_channels = 3
        self.encoder = fc_encoder(input_channels=input_channels, latent_channels=opt['z_channels'])
        if opt['x_dis'] == 'MixLogistic':
            self.decoder = fc_decoder(latent_channels=opt['z_channels'], out_channels=100)
            self.criterion = lambda data, params: discretized_mix_logistic_uniform(data, params)
            self.sample_op = lambda params: discretized_mix_logistic_sample(params)
        elif opt['x_dis'] == 'Logistic':
            self.decoder = fc_decoder(latent_channels=opt['z_channels'], out_channels=9)
            self.criterion = lambda data, params: discretized_logistic(data, params)
            self.sample_op = lambda params: discretized_logistic_sample(params)

        self.prior_mu = torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std = torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())

    def posterior_forward(self, params, x):
        z_mu, z_std = params[0], F.softplus(params[1])
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu, z_std, self.prior_mu.to(self.device), self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo) / np.log(2.)

    def forward(self, x):
        z_mu, z_std = self.encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu, z_std, self.prior_mu.to(self.device), self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo) / np.log(2.)

    def sample(self, num=100):
        with torch.no_grad():
            eps = torch.randn(num, self.z_dim).to(self.device)
            pxz_params = self.decoder(eps)
            return self.sample_op(pxz_params)


# The following is the autoregressive model for learning p_eta(z)

class PixelCNN(nn.Module):
    def __init__(self, device, mixture_num, cat_dim, input_size, output_size, c_hidden, layer_num):
        super().__init__()
        self.mixture_num = mixture_num
        self.cat_dim = cat_dim
        self.input_size = input_size
        self.output_size = output_size
        self.c_in = input_size[1]
        self.c_out = (mixture_num * 3) * self.c_in  # pi, mean, std
        self.c_hidden = c_hidden
        self.device = device
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(self.c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(self.c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList()
        for i in range(layer_num):
            self.conv_layers.append(GatedMaskedConv(c_hidden))
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, self.c_out, kernel_size=(1, 1), padding=0)  # mu and std

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        # x = (x.float() / 255.0) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Channels, Height, Width]
        mean = out[:, :int(self.c_in) * self.mixture_num, :, :]
        std = F.softplus(out[:, int(self.c_in) * self.mixture_num: int(self.c_in) * self.mixture_num * 2, :, :])
        pi = out[:, int(self.c_in) * self.mixture_num * 2:, :, :]
        return mean, std, pi

    @torch.no_grad()
    def sample(self):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        # img_mean = 0. * torch.ones(self.input_size, dtype=torch.float).to(self.device)
        # img_std = 0. * torch.ones(self.input_size, dtype=torch.float).to(self.device)
        img = 0. * torch.ones(self.input_size, dtype=torch.float).to(self.device)
        # Generation loop
        for h in range(self.input_size[2]):
            for w in range(self.input_size[3]):
                pred_mean, pred_std, pred_pi = self.forward(img)
                if self.mixture_num == 1:
                    pred = pred_mean + pred_std * torch.randn_like(pred_std)
                else:
                    mean_list = torch.chunk(pred_mean, self.mixture_num, dim=self.cat_dim)
                    std_list = torch.chunk(pred_std, self.mixture_num, dim=self.cat_dim)
                    pi_list = torch.chunk(pred_pi, self.mixture_num, dim=self.cat_dim)
                    mean = torch.stack(mean_list, dim=-1)
                    std = torch.stack(std_list, dim=-1)
                    gaussian_sample = mean + std * torch.randn_like(std)
                    pi = torch.stack(pi_list, dim=-1) + eps
                    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=pi) # The normalization is automatic
                    categorical_sample = m.sample()
                    pred = (gaussian_sample * categorical_sample).sum(-1)
                img[:, :, h, w] = pred[:, :, h, w]
                # img_mean[:, :, h, w] = pred_mean[:, :, h, w]
                # img_std[:, :, h, w] = pred_std[:, :, h, w]
        return img, (None, None)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.c_in = input_size[2]
        self.c_out = output_size[2]
        self.lstm = nn.LSTM(self.c_in, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.c_out)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out)  # out: tensor of shape (batch_size, seq_length, output_size)
        return out[:, :, :1], F.softplus(out[:, :, 1:]), None

    @torch.no_grad()
    def sample(self):
        # Create empty image
        img_mean = 0. * torch.ones(self.input_size, dtype=torch.float)
        img_std = 0. * torch.ones(self.input_size, dtype=torch.float)
        img = 0. * torch.ones(self.input_size, dtype=torch.float).to(self.device)
        for h in range(self.input_size[-2]):
            pred_mean, pred_std, _ = self.forward(img)
            pred = pred_mean + pred_std * torch.randn_like(pred_std)
            img[:, h, :] = pred[:, h, :]
            img_mean[:, h, :] = pred_mean[:, h, :]
            img_std[:, h, :] = pred_std[:, h, :]
        return img, (img_mean, img_std)


class linear_VAE(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_size):
        super(linear_VAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device

        self.encoder = nn.Sequential(nn.Linear(self.input_size[1], self.hidden_size * 2))
        self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.output_size[1] * 2))

        self.prior_mu = torch.zeros([self.output_size[0], self.hidden_size], requires_grad=False)
        self.prior_std = torch.ones([self.output_size[0], self.hidden_size], requires_grad=False)

    def forward(self, x):
        s = self.encoder(x)
        s_mean, s_std = s[:, :self.hidden_size], F.softplus(s[:, self.hidden_size:])
        s_samples = s_mean + s_std * torch.randn_like(s_std)
        x_hat = self.decoder(s_samples)
        x_hat_mean, x_hat_std = x_hat[:, :self.output_size[1]], F.softplus(x_hat[:, self.output_size[1]:])
        return x_hat_mean, x_hat_std, [s_mean, s_std]

    @torch.no_grad()
    def sample(self):
        s = self.prior_mu + self.prior_std * torch.randn_like(self.prior_std)
        img = self.decoder(s.to(self.device))
        img_mean, img_std = img[:, :self.output_size[1]], F.softplus(img[:, self.output_size[1]:])
        img = img_mean + img_std * torch.randn_like(img_std)
        return img, (img_mean, img_std)


