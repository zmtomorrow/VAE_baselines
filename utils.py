import torchvision

import torch

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
    kwargs['save_path'] = kwargs['save_path'] + kwargs['data_set'] + '/'
    if not os.path.exists(kwargs['save_path']):
        os.mkdir(kwargs['save_path'])
    return kwargs