import torchvision
from torch.utils import data
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pickle
import argparse
import os
from typing import Tuple, List, Dict

eps = 1e-7


def forward_kl(shape, z_mean, z_std, z_hat_mean, z_hat_std):
    z_mean = z_mean.view([-1, np.prod(shape[1:])])
    z_std = z_std.view([-1, np.prod(shape[1:])])
    z_hat_mean = z_hat_mean.view([-1, np.prod(shape[1:])])
    z_hat_std = z_hat_std.view([-1, np.prod(shape[1:])])

    kl = (torch.log(z_hat_std + eps) - torch.log(z_std + eps) +
          (z_std ** 2 + (z_mean - z_hat_mean) ** 2) / (2 * z_hat_std ** 2) - 0.5).sum(1).mean(0)  # Forward KL
    return kl


def reverse_kl(shape, z_mean, z_std, z_hat_mean, z_hat_std):
    z_mean = z_mean.view([-1, np.prod(shape[1:])])
    z_std = z_std.view([-1, np.prod(shape[1:])])
    z_hat_mean = z_hat_mean.view([-1, np.prod(shape[1:])])
    z_hat_std = z_hat_std.view([-1, np.prod(shape[1:])])

    kl = (torch.log(z_std + eps) - torch.log(z_hat_std + eps) +
          (z_hat_std ** 2 + (z_hat_mean - z_mean) ** 2) / (2 * z_std ** 2) - 0.5).sum(1).mean(0)  # Reverse KL
    return kl


def mse(shape, z_mean, z_std, z_hat_mean, z_hat_std):
    z_mean = z_mean.view([-1, np.prod(shape[1:])])
    z_std = z_std.view([-1, np.prod(shape[1:])])
    z_hat_mean = z_hat_mean.view([-1, np.prod(shape[1:])])
    z_hat_std = z_hat_std.view([-1, np.prod(shape[1:])])

    d = ((z_hat_mean - z_mean) ** 2 + (z_hat_std - 1.) ** 2).sum(1).mean(0)  # Mean squared error
    return d


def elbo(shape, z_mean, z_std, z_hat_mean, z_hat_std, s_mean, s_std):
    z_mean = z_mean.view([-1, np.prod(shape[1:])])
    z_std = z_std.view([-1, np.prod(shape[1:])])
    z_hat_mean = z_hat_mean.view([-1, np.prod(shape[1:])])
    z_hat_std = z_hat_std.view([-1, np.prod(shape[1:])])
    llk = - (torch.log(z_hat_std + eps) +
           (z_std ** 2 + (z_mean - z_hat_mean) ** 2) / (2 * z_hat_std ** 2)).sum(1).mean(0)
    kl = (torch.log(s_std + eps) +
          (1 ** 2 + (0 - s_mean) ** 2) / (2 * s_std ** 2)).sum(1).mean(0)
    return -(llk - kl)
