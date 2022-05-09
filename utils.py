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

rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5


def batch_KL_diag_gaussian_std(mu_1, std_1, mu_2, std_2):
    diag_1 = std_1 ** 2
    diag_2 = std_2 ** 2
    ratio = diag_1 / diag_2
    return 0.5 * (
            torch.sum((mu_1 - mu_2) ** 2 / diag_2, dim=-1)
            + torch.sum(ratio, dim=-1)
            - torch.sum(torch.log(ratio), dim=-1)
            - mu_1.size(1)
    )


def to_one_hot(tensor, n, fill_with=1.):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data = torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,
                                               transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,
                                              transform=torchvision.transforms.ToTensor())

    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug'] == True:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()])
        else:
            transform = torchvision.transforms.ToTensor()
        train_data = torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False, transform=transform)
        test_data = torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,
                                                 transform=torchvision.transforms.ToTensor())

    elif opt['data_set'] == 'MNIST':
        train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,
                                                transform=torchvision.transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,
                                               transform=torchvision.transforms.ToTensor())

    elif opt['data_set'] == 'BinaryMNIST':
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: torch.round(x),
        ])
        train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False, transform=trans)
        test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False, transform=trans)

    elif opt["data_set"] == "latent":
        train_data = LatentBlockDataset(opt['dataset_path'], train=True, transform=None)
        test_data = LatentBlockDataset(opt['dataset_path'], train=False, transform=None)
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader, test_data_loader, train_data_evaluation


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """
    def __init__(self, file_path, train=True, transform=None):
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
        print('Done loading pretrained latent block data')

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        mean = self.data['mean'][index]
        std = self.data['std'][index]
        return mean, std

    def __len__(self):
        return len(self.data['mean'])


def process_args(args: argparse.Namespace) -> Dict:
    kwargs = vars(args)
    # save_path = args.save_path.rstrip()
    # save_path = (
    #     f"{save_path}/{args.dataset}/dataset_{args.dataset}__method_{args.method}__train_size_{args.train_size}"
    #     f"__DA_{args.aug}__reg_{args.reg}__seed_{args.seed}__"
    # )
    # i = 1
    # while os.path.exists(f"{save_path}{i}") or os.path.exists(
    #         f"{save_path}{i}__complete"
    # ):
    #     i += 1
    # save_path = f"{save_path}{i}"
    # kwargs["save_path"] = save_path
    # if args.save:
    #     os.mkdir(save_path)
    return kwargs