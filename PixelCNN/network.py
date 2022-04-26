import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .module import GatedMaskedConv, VerticalStackConvolution, HorizontalStackConvolution


class PixelCNN(nn.Module):

    def __init__(self, c_in, c_hidden, layer_num):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList()
        for i in range(layer_num):
            self.conv_layers.append(GatedMaskedConv(c_hidden))
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in, kernel_size=(1, 1), padding=0)

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
        return out[:, :int(self.c_in / 2), :, :], F.softplus(out[:, int(self.c_in / 2):, :, :])

    @torch.no_grad()
    def sample(self, img_shape, device, img=None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = 0. * torch.ones(img_shape, dtype=torch.float).to(device)
        # Generation loop
        for h in range(img_shape[2]):
            for w in range(img_shape[3]):
                pred_mean, pred_std = self.forward(img)
                pred = torch.cat((pred_mean, pred_std), dim=1)
                img[:, :, h, w] = pred[:, :, h, w]
        gen_z_mean = img[:, :int(self.c_in / 2), :, :]
        gen_z_std = img[:, int(self.c_in / 2):, :, :]
        epsilon = torch.randn_like(gen_z_mean)
        gen_z = epsilon * gen_z_std + gen_z_mean
        return gen_z
