import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
from torch.utils import data
import torch.nn.functional as F


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


def one_hot_labels(y, dataset_name):
    if dataset_name == 'BinaryMNIST' or dataset_name == 'MNIST':
        return F.one_hot(y, 10)
    elif dataset_name == 'ColoredMNIST':
        y1 = F.one_hot(y[0], 10)  # Digit Label
        y2 = F.one_hot(y[1].to(torch.int64), 2)  # Color Label
        return torch.cat((y1, y2), dim=1)
    else:
        raise NotImplementedError(dataset_name)


def LoadData(opt, latent=False):
    if latent is True:
        train_data = LatentBlockDataset(opt['dataset_path'] + 'latent.pickle', train=True, transform=None)
        test_data = LatentBlockDataset(opt['dataset_path'] + 'latent.pickle', train=False, transform=None)
    else:
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
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(32),
                 torchvision.transforms.ToTensor()]
            )
            train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True,
                                                    transform=transform)
            test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True,
                                                   transform=transform)

        elif opt['data_set'] == 'BinaryMNIST':
            trans = torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                lambda x: torch.round(x),
            ])
            train_data = torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=True, transform=trans)
            test_data = torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=True, transform=trans)

        elif opt['data_set'] == 'ColoredMNIST':
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(32),
                 torchvision.transforms.ToTensor()]
            )
            train_data = ColoredMNIST(opt['dataset_path'], env='all_train', transform=transform)
            test_data = ColoredMNIST(opt['dataset_path'], env='test', transform=transform)
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


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Assign a binary label y to the image based on the digit: y = 0 for digits 0-4 and y = 1 for digits 5-9.
    Flip the label with 50% probability.
    Color the image either red or green according to its (possibly flipped) label.
    Flip the color with a probability e that depends on the environment: 50% in the first training environment,
    50% in the second training environment, and 50% in the test environment.

    Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
    """

    def __init__(self, root='../data', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 20000:
                # 20% in the first training environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 20000:
                train1_set.append((Image.fromarray(colored_arr), [label, color_red]))
            elif idx < 40000:
                train2_set.append((Image.fromarray(colored_arr), [label, color_red]))
            else:
                test_set.append((Image.fromarray(colored_arr), [label, color_red]))

        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


if __name__ == '__main__':
    def plot_dataset_digits(dataset):
        fig = plt.figure(figsize=(13, 8))
        columns = 6
        rows = 3
        # ax enables access to manipulate each of subplots
        ax = []

        for i in range(columns * rows):
            img, label = dataset[i]
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title("Label: " + str(label))  # set title
            plt.imshow(img)

        plt.show()  # finally, render the plot


    train1_set = ColoredMNIST(root='../data', env='train1')
    plot_dataset_digits(train1_set)
