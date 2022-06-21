from model import *
import sys
import argparse
import utils
from objectives import *

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/PixelCNN')
os.chdir('/import/home/xzhoubi/hudson/VAE_baselines/')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser()
parser.add_argument('--stage_one_arch', type=str, default='dc_vae')
parser.add_argument('--architecture', type=str, default='pixelcnn')
parser.add_argument('--loss', type=str, default='forward_kl')
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--data_set', type=str, default='BinaryMNIST')
parser.add_argument('--z_channels', type=int, default=2)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--mixture_num', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--x_dis', type=str, default='Bernoulli')
parser.add_argument('--alpha', type=float, default=0.0)
args = parser.parse_args()
opt = utils.process_args(args, stage_two=True)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt["device"] = torch.device("cuda:0")
    opt["if_cuda"] = True
else:
    opt["device"] = torch.device("cpu")
    opt["if_cuda"] = False

opt['save_path'] = opt['path']
# opt['x_dis'] = 'Bernoulli'  ## or Logistic
# opt['z_channels'] = 2  ## 2*64
opt['epochs'] = 10
opt['result_path'] = './result/'
opt['dataset_path'] = opt['path']
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

train_data, test_data, train_data_evaluation = LoadData(opt, latent=True)

if args.stage_one_arch == 'dc_vae':
    opt['z_dim'] = 100
    stage_one_model = dc_VAE(opt).to(opt['device'])
    stage_one_model.load_state_dict(torch.load(f"{opt['path']}stage_one_model.pth"))
else:
    raise NotImplementedError(args.stage_one_arch)

if args.architecture == "pixelcnn":
    latent_z_shape = [opt['batch_size'], opt['z_channels'], 8, 8]
    output_size = [opt['gen_samples']] + [latent_z_shape[1] * 2] + latent_z_shape[2:]
    cat_dim = 1
    model = PixelCNN(device=opt['device'], input_size=latent_z_shape, c_hidden=opt['c_hidden'],
                     output_size=output_size, layer_num=opt['layer_num'], mixture_num=opt['mixture_num'],
                     cat_dim=cat_dim).to(opt["device"])
elif args.architecture == 'rnn':
    latent_z_shape = [opt['batch_size'], opt['z_channels'] * 8 * 8, 1]
    output_size = [opt['gen_samples']] + [latent_z_shape[1]] + [latent_z_shape[2] * 2]
    cat_dim = 2
    model = RNN(device=opt['device'], input_size=latent_z_shape, hidden_size=opt['c_hidden'],
                num_layers=2, output_size=output_size).to(opt["device"])
elif args.architecture == 'vae':
    latent_z_shape = [opt['batch_size'], opt['z_dim']]
    output_size = [opt['gen_samples']] + [latent_z_shape[1]]
    cat_dim = 2
    model = linear_VAE(device=opt['device'], input_size=latent_z_shape, hidden_size=opt['c_hidden'],
                       output_size=output_size).to(opt["device"])
else:
    raise NotImplementedError(args.architecture)

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])


model.train()

loss_list = []
for epoch in range(1, opt['epochs'] + 1):
    model.train()
    for z_mean, z_std in tqdm(train_data):
        optimizer.zero_grad()
        z_mean = z_mean.reshape(latent_z_shape).to(opt["device"])
        z_std = z_std.reshape(latent_z_shape).to(opt["device"])
        z = z_mean + z_std * torch.randn_like(z_std)
        z1, z2, z3 = model.forward(z)

        if opt['loss'] == 'forward_kl':
            loss = forward_kl(latent_z_shape, z_mean, z_std, z_hat_mean=z1, z_hat_std=z2)
        elif opt['loss'] == 'reverse_kl':
            loss = reverse_kl(latent_z_shape, z_mean, z_std, z_hat_mean=z1, z_hat_std=z2)
        # elif opt['loss'] == 'mse':
        #     loss = mse(latent_z_shape, z_mean, z_std, z_hat_mean, z_hat_std)
        elif opt['loss'] == 'elbo':
            loss = elbo(latent_z_shape, z_mean, z_std, z_hat_mean=z1, z_hat_std=z2, s_mean=z3[0], s_std=z3[1])
        elif opt['loss'] == 'mixture_of_gaussian':
            loss = mixture_of_gaussian(latent_z_shape, z, z_hat_mean=z1, z_hat_std=z2, z_hat_pi=z3,
                                       mixture_num=opt['mixture_num'], cat_dim=cat_dim)
        else:
            raise NotImplementedError(opt['loss'])

        loss.backward()
        optimizer.step()
        # print("KL", kl.item())
        loss_list.append(loss.item())

    if opt['visualize'] and epoch % opt['log_freq'] == 0:
        model.eval()
        gen_z, _ = model.sample()
        pxz_params = stage_one_model.decoder(gen_z)
        x_hat = stage_one_model.sample_op(pxz_params)

        fig, ax = plt.subplots(2, 10, figsize=(20, 6))
        ax = ax.flatten()
        for i in range(20):
            x_vis = x_hat[i, :].detach().cpu().numpy().transpose(1, 2, 0)
            if x_vis.shape[-1] == 1:
                ax[i].imshow(x_vis, cmap='gray')
            else:
                ax[i].imshow(x_vis)
            ax[i].axis('off')
        plt.suptitle("Images generated from p eta (z)")
        fig.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(loss_list)
        plt.show()

        loss_list = []

        z_dim = np.prod(latent_z_shape[1:])
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
        torch.save(model.state_dict(), f"{opt['save_path']}stage_two_model.pth")
        with open(f"{opt['save_path']}stage_two_model_config.pickle", 'wb') as handle:
            pickle.dump(opt, handle)
