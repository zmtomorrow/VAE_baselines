import torchvision
from torch.utils import data
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt


rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5


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
    if tensor.is_cuda : one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())
        
    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug']==True:
            transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor()])
        else:
            transform=torchvision.transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=transform)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader,test_data_loader,train_data_evaluation
