import torch
from torch import optim
from utils import *
from dataset import *
from model import *
import numpy as np
from tqdm import tqdm
import os
import utils
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir("/home/xzhoubi/hudson/VAE_baselines/")

print(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--data_set', type=str, default=None)
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

opt['x_dis'] = 'Logistic'  ## or MixLogistic
opt['dataset_path'] = '../data/'
opt['result_path'] = './result/'
opt['batch_size'] = 200
opt['test_batch_size'] = 200
opt['if_regularizer'] = False
opt['load_model'] = False
opt['data_aug'] = False
opt["seed"] = 0
opt['if_save_model'] = True
opt['save_epoch'] = 50
opt['additional_epochs'] = 100
opt['sample_size'] = 100
opt['if_save_latent'] = True

np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])

train_data, test_data, train_data_evaluation = LoadData(opt)
model = dense_VAE(opt).to(opt['device'])

if opt['load_model'] == True:
    model.load_state_dict(torch.load(f"{opt['save_path']}model_{opt['data_set']}_{str(opt['z_channels'])}.pth"))

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

test_BPD_list = []
for epoch in range(1, opt['epochs'] + 1):
    model.train()
    for x, y in tqdm(train_data):
        optimizer.zero_grad()
        y = one_hot_labels(y, opt['data_set'])
        L = model.joint_forward(x.to(opt['device']), y.to(opt['device']))
        L.backward()
        optimizer.step()
    
    ## visualize
    if epoch % 10 == 0:
        gen_images = model.sample()

        plt.figure()
        plt.imshow(gen_images.cpu().numpy()[0, 0, :, :], cmap='gray')
        plt.show()
    
    print('epoch:', epoch, L.item())

    if opt['if_save_model']:
        torch.save(model.state_dict(), f"{opt['save_path']}model.pth")

if opt['if_save_latent']:
    z_mu_list = []
    z_std_list = []
    with torch.no_grad():
        for x, _ in train_data:
            z_mu, z_std = model.encoder(x.to(opt['device']))
            z_std_list.append(z_std.cpu().numpy())
            z_mu_list.append(z_mu.cpu().numpy())

    z_dict = {}
    z_mu_array = np.concatenate(z_mu_list, axis=0).reshape([-1, 100])
    z_std_array = np.concatenate(z_std_list, axis=0).reshape([-1, 100])
    z_dict["mean"] = z_mu_array
    z_dict["std"] = z_std_array

    with open(f"{opt['save_path']}latent.pickle", 'wb') as handle:
        pickle.dump(z_dict, handle)
