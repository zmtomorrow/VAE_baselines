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

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/PixelCNN')

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
opt = {}
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
opt['z_channels'] = 16  ## 2*64
opt['epochs'] = 100
opt['dataset_path'] = f"./save/latent_CIFAR_{opt['z_channels']}.pickle"
opt['save_path'] = './save/'
opt['result_path'] = './result/'
opt['batch_size'] = 100
opt['test_batch_size'] = 200
opt['if_regularizer'] = False
opt['load_model'] = False
opt['lr'] = 1e-4
opt['data_aug'] = False
opt["seed"] = 0
opt['if_save_model'] = True
opt['save_epoch'] = 50
opt['additional_epochs'] = 100
opt['sample_size'] = 100
opt['if_save_latent'] = True
opt["c_hidden"] = 128
opt['layer_num'] = 15
opt['visualize'] = True
np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])
eps = 1e-7

train_data, test_data, train_data_evaluation = LoadData(opt)
latent_data_shape = [-1, 1, 32, 32]

model = PixelCNN(c_in=latent_data_shape[1] * 2, c_hidden=opt['c_hidden'], layer_num=opt['layer_num']).to(opt["device"])
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
        z = torch.cat((z_mean, z_std), dim=1)
        z_hat_mean, z_hat_std = model.forward(z)
        
        z_mean = z_mean.view([-1, np.prod(latent_data_shape[1:])])
        z_std = z_std.view([-1, np.prod(latent_data_shape[1:])])
        z_hat_mean = z_hat_mean.view([-1, np.prod(latent_data_shape[1:])])
        z_hat_std = z_hat_std.view([-1, np.prod(latent_data_shape[1:])])

        kl = (torch.log(z_hat_std + eps) - torch.log(z_std + eps) +
              (z_std ** 2 + (z_mean - z_hat_mean) ** 2) / (2 * z_hat_std ** 2) - 0.5).sum(1).mean(0)
        kl.backward()
        optimizer.step()
        # print("KL", kl.item())
        KL_list.append(kl.item())

    model.eval()
    gen_img_shape = [20] + [latent_data_shape[1] * 2] + latent_data_shape[2:]
    gen_z = model.sample(img_shape=gen_img_shape, device=opt["device"])

    if opt['visualize'] and (epoch + 1) % 2 == 0:
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

    if opt['if_save_model']:
        torch.save(model.state_dict(), f"{opt['save_path']}PixelCNN_{str(opt['z_channels'])}.pth")

