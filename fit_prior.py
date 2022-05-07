import torch
import torch.nn.functional as F
from torch import optim
from utils import *
from model import *
from PixelCNN.network import *
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import utils

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/PixelCNN')
os.chdir('/import/home/xzhoubi/hudson/VAE_baselines/')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=str, default='pixelcnn')
parser.add_argument('--loss', type=str)
parser.add_argument('--z_channels', type=int, default=2)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()
opt = utils.process_args(args)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt["device"] = torch.device("cuda:0")
    opt["if_cuda"] = True
else:
    opt["device"] = torch.device("cpu")
    opt["if_cuda"] = False

opt['data_set'] = 'latent'
opt['x_dis'] = 'Logistic'  ## or MixLogistic
# opt['z_channels'] = 2  ## 2*64
opt['epochs'] = 100
opt['dataset_path'] = f"./save/latent_CIFAR_{opt['z_channels']}.pickle"
opt['save_path'] = './save/'
opt['result_path'] = './result/'
opt['batch_size'] = 100
opt['test_batch_size'] = 200
opt['if_regularizer'] = False
opt['load_model'] = False
# opt['lr'] = 1e-3
opt['data_aug'] = False
opt["seed"] = 0
opt['if_save_model'] = True
opt['save_epoch'] = 50
opt['additional_epochs'] = 100
opt['sample_size'] = 100
opt['if_save_latent'] = True
opt['c_hidden'] = 128
opt['layer_num'] = 15
opt['visualize'] = True
opt['gen_samples'] = 100
np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])
eps = 1e-7

train_data, test_data, train_data_evaluation = LoadData(opt)

if args.architecture == "pixelcnn":
    latent_data_shape = [-1, opt['z_channels'], 8, 8]
    gen_img_shape = [opt['gen_samples']] + [latent_data_shape[1] * 2] + latent_data_shape[2:]
    cat_dim = 1
    model = PixelCNN(device=opt['device'], c_in=latent_data_shape[1] * 2, c_hidden=opt['c_hidden'],
                     layer_num=opt['layer_num']).to(opt["device"])
elif args.architecture == 'rnn':
    latent_data_shape = [-1, opt['z_channels'] * 8 * 8, 1]
    gen_img_shape = [opt['gen_samples']] + [latent_data_shape[1]] + [latent_data_shape[2] * 2]
    cat_dim = 2
    model = RNN(device=opt['device'], input_size=latent_data_shape[2] * 2, hidden_size=opt['c_hidden'],
                num_layers=2, output_size=latent_data_shape[2] * 2).to(opt["device"])
else:
    raise NotImplementedError(args.architecture)

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

pretrained_VAE = VAE(opt).to(opt['device'])
pretrained_VAE.load_state_dict(torch.load(f"{opt['save_path']}model_CIFAR_{str(opt['z_channels'])}.pth"))

model.train()

KL_list = []
for epoch in range(1, opt['epochs'] + 1):
    model.train()
    for z_mean, z_std in tqdm(train_data):
        optimizer.zero_grad()
        z_mean = z_mean.reshape(latent_data_shape).to(opt["device"])
        z_std = z_std.reshape(latent_data_shape).to(opt["device"])
        z = torch.cat((z_mean, z_std), dim=cat_dim)
        z_hat_mean, z_hat_std = model.forward(z)
        
        z_mean = z_mean.view([-1, np.prod(latent_data_shape[1:])])
        z_std = z_std.view([-1, np.prod(latent_data_shape[1:])])
        z_hat_mean = z_hat_mean.view([-1, np.prod(latent_data_shape[1:])])
        z_hat_std = z_hat_std.view([-1, np.prod(latent_data_shape[1:])])

        if opt['loss'] == 'forward_kl':
            kl = (torch.log(z_hat_std + eps) - torch.log(z_std + eps) +
              (z_std ** 2 + (z_mean - z_hat_mean) ** 2) / (2 * z_hat_std ** 2) - 0.5).sum(1).mean(0)  # Forward KL
        # kl = (torch.log(z_hat_std + eps) +
        #       (z_std ** 2 + (z_mean - z_hat_mean) ** 2) / (2 * z_hat_std ** 2)).mean(1).mean(0)
        elif opt['loss'] == 'reverse_kl':
            kl = (torch.log(z_std + eps) - torch.log(z_hat_std + eps) +
              (z_hat_std ** 2 + (z_hat_mean - z_mean) ** 2) / (2 * z_std ** 2) - 0.5).sum(1).mean(0)  # Reverse KL
        elif opt['loss'] == 'mean':
            kl = ((z_hat_mean - z_mean) ** 2 + (z_hat_std - 1.) ** 2).sum(1).mean(0)  # Reverse KL
        else:
            raise NotImplementedError(opt['loss'])

        kl.backward()
        optimizer.step()
        # print("KL", kl.item())
        KL_list.append(kl.item())

    model.eval()
    gen_z, _ = model.sample(img_shape=gen_img_shape)

    if opt['visualize'] and (epoch) % opt['log_freq'] == 0:
        pxz_params = pretrained_VAE.decoder(gen_z)
        x_hat = pretrained_VAE.sample_op(pxz_params)
        fig, ax = plt.subplots(2, 10, figsize=(20, 6))
        ax = ax.flatten()
        for i in range(20):
            ax[i].imshow(x_hat[i, :].detach().cpu().numpy().transpose(1, 2, 0))
        plt.suptitle("Images generated from trained z")
        plt.show()

        plt.figure()
        plt.plot(KL_list)
        plt.show()

        KL_list = []

        z_dim = np.prod(latent_data_shape[1:])
        gen_z = gen_z.reshape(opt['gen_samples'], z_dim).to("cpu")
        fig, ax = plt.subplots(2, 5, figsize=(20, 6))
        ax = ax.flatten()
        for i in range(len(ax)):
            ax[i].hist(gen_z[:, i].numpy(), bins=50, density=True)
        plt.suptitle("p eta (z)")
        plt.show()

        sample_num = 20
        samples_list = []
        z_mean = z_mean.reshape([-1, z_dim]).to("cpu")
        z_std = z_std.reshape([-1, z_dim]).to("cpu")
        for j in range(z_mean.shape[0]):
            temp_mean = z_mean[j, :][None, :]
            temp_std = z_std[j, :][None, :]
            epsilon = torch.randn(sample_num, z_dim)
            samples = temp_mean + temp_std * epsilon
            samples_list.append(samples)
        samples_array = torch.cat(samples_list, dim=0)  ## (x_num * sample_num) * z_dim
        fig, ax = plt.subplots(2, 5, figsize=(20, 6))
        ax = ax.flatten()
        for i in range(len(ax)):
            ax[i].hist(samples_array[:, i].numpy(), bins=50, density=True)
        plt.suptitle("int q (z|x) p(x) dx")
        plt.show()

    if opt['if_save_model']:
        torch.save(model.state_dict(), f"{opt['save_path']}{opt['architecture']}_{str(opt['z_channels'])}.pth")

