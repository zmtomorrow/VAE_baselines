import torch
import torch.nn as nn
import torch.nn.functional as F


class dc_encoder(nn.Module):
    def __init__(self, h_size, z_dim, i_channel):
        super().__init__()
        self.z_dim = z_dim
        self.h_size = h_size
        self.i_channel = i_channel
        self.encoder = nn.Sequential(
            nn.Conv2d(self.i_channel, 32, 6, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5),
            nn.LeakyReLU(0.2),
        )

        self.en_mean = nn.Linear(self.h_size, self.z_dim)
        self.en_logstd = nn.Linear(self.h_size, self.z_dim)

    def forward(self, x, y=None):
        h = self.encoder(x)
        if y is not None:
            h = torch.flatten(h, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            h = torch.cat([h, y], dim=1)
        mu = self.en_mean(h)
        std = F.softplus(self.en_logstd(h))
        return mu, std


class dc_decoder(nn.Module):
    def __init__(self, z_dim, i_channel, i_channel_multiply):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = 256
        self.i_channel_multiply = i_channel_multiply
        self.i_channel = i_channel
        self.fc = nn.Linear(self.z_dim, self.h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.i_channel * self.i_channel_multiply, 4, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 1, 1)
        x = self.decoder(h)
        return x
    
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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class dense_encoder(nn.Module):
    def __init__(self, i_dim, z_dim, if_bn=True):
        super().__init__()
        self.h_dim = 500
        self.z_dim = z_dim
        self.i_dim = i_dim

        self.fc1 = nn.Linear(i_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc31 = nn.Linear(self.h_dim, self.z_dim)
        self.fc32 = nn.Linear(self.h_dim, self.z_dim)

        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
            self.bn2 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()

    def forward(self, x, y=None):
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            x = torch.cat([x, y], dim=1)
        x = x.view(-1, self.i_dim)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        mu = self.fc31(h)
        std = F.softplus(self.fc32(h))
        return mu, std


class dense_decoder(nn.Module):
    def __init__(self, i_dim, z_dim, i_shape, i_channel_multiply, if_bn=True):
        super().__init__()
        self.h_dim = 500
        self.z_dim = z_dim
        self.i_dim = i_dim
        self.i_shape = i_shape
        self.i_channel_multiply = i_channel_multiply

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, self.i_dim * self.i_channel_multiply)
        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
            self.bn2 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()

    def forward(self, z):
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h.view([-1] + [self.i_shape[0] * self.i_channel_multiply] + self.i_shape[1:])


class conv_encoder(nn.Module):
    def __init__(self, input_channels, channels, z_channels):
        super(conv_encoder, self).__init__()
        self.z_channels = z_channels
        self.z_dim = self.z_channels * 8 * 8
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(True),
#             nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(channels),
#             Residual(channels),
#             Residual(channels),
#             nn.Conv2d(channels, latent_channels * 2, 1)
#         )
        self.conv1 = nn.Conv2d(input_channels, channels, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(channels, z_channels, 1)
        
    def forward(self, x, y=None):
        x = self.conv1(x)
        x = F.relu(nn.BatchNorm2d(channels)(x))
        x = self.conv2(x)
        x = nn.BatchNorm2d(channels)(x)
        x = Residual(channels)(x)
        x = Residual(channels)(x)
        x = self.conv3(x)
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            FC = nn.Linear(x.shape[1] + y.shape[1], self.z_dim * 2)
            z = FC(x)
        else:
            x = torch.flatten(x, start_dim=1)
            FC = nn.Linear(x.shape[1], self.z_dim * 2)
            z = FC(x)
        return z[:, :self.z_dim], F.softplus(z[:, self.z_dim:])


class conv_decoder(nn.Module):
    def __init__(self, channels, latent_channels, out_channels):
        super(conv_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, bias=False),
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
        #         print('here', z.shape)
        return self.decoder(z.view(z.size(0), -1, 8, 8))


class MaskedConvolution(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1:, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2 * c_in, 2 * c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class mnist_classifier(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.ntwk = nn.Sequential(nn.Linear(z_dim, 200), 
                                  nn.ReLU(True), 
                                  nn.BatchNorm1d(200), 
                                  nn.Linear(200, 10), 
                                  nn.Softmax())

    def forward(self, z):
        return self.ntwk(z.view(-1, self.z_dim))


class colored_mnist_classifier(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.ntwk_1 = nn.Sequential(nn.Linear(z_dim, 200), 
                                  nn.ReLU(True), 
                                  nn.BatchNorm1d(200), 
                                  nn.Linear(200, 10), 
                                  nn.Softmax())
        self.ntwk_2 = nn.Sequential(nn.Linear(z_dim, 200), 
                                  nn.ReLU(True), 
                                  nn.BatchNorm1d(200), 
                                  nn.Linear(200, 2), 
                                  nn.Softmax())

    def forward(self, z):
        return torch.cat((self.ntwk_1(z.view(-1, self.z_dim)), 
                          self.ntwk_2(z.view(-1, self.z_dim))), dim=1)
