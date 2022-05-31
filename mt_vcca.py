from network import *
from tools import *
from utils import *
import torch.optim as optim
from tqdm import tqdm
import torch.distributions as dis
from torch.distributions.bernoulli import Bernoulli


class VCCA(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.z_dim = opt['z_dim']
        self.class_num = 10
        self.dataset = opt['data_set']
        if opt['data_set'] in ['CIFAR', 'SVHN']:
            self.encoder = dc_encoder(z_dim=self.z_dim)
            self.decoder = dc_decoder(z_dim=self.z_dim, out_channels=3, h_dim=256)
            self.criterion = lambda real, fake: discretized_mix_logistic_loss(real, fake)
        else:
            self.joint_encoder = densenet_encoder(input_dim=794, z_dim=self.z_dim)
            self.x_encoder = densenet_encoder(input_dim=784, z_dim=self.z_dim)
            if opt['data_set'] == 'BinaryMNIST':
                self.decoder = densenet_decoder(o_dim=1, z_dim=self.z_dim)
                self.criterion = lambda real, fake: -Bernoulli(logits=fake).log_prob(real).sum([1, 2, 3])
                self.sample_op = lambda x: Bernoulli(logits=x).sample()
            elif opt['data_set'] == 'MNIST':
                self.decoder = densenet_decoder(o_dim=9, z_dim=self.z_dim)
                self.criterion = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
                self.sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, 3)
        self.classifier = ClassNet(opt)
        self.device = opt['device']
        self.prior_mu = torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std = torch.ones(self.z_dim, requires_grad=False)
        self.optimizer = optim.Adam(self.parameters(), lr=opt['lr'])
        self.encoder_optimizer = optim.Adam(self.x_encoder.parameters(), lr=opt['lr'])

    def joint_forward(self, x, y, alpha=1.0, return_elbo=False):
        z_mu, z_std = self.joint_encoder(x, y)
        eps = torch.randn_like(z_mu).to(self.device)
        z = eps.mul(z_std).add_(z_mu)
        x_out = self.decoder(z)
        kl = batch_KL_diag_gaussian_std(z_mu, z_std, self.prior_mu.to(self.device), self.prior_std.to(self.device))
        neg_l = self.criterion(x, x_out)
        y_z_dis = self.classifier(z)
        if return_elbo:
            logpy_z = torch.sum(y * torch.log(y_z_dis + 1e-8), dim=1)
            return -neg_l + logpy_z - kl

        classication_loss = -torch.sum(y * torch.log(y_z_dis + 1e-8), dim=1).mean()
        loss = torch.mean(neg_l + kl, dim=0) + alpha * classication_loss
        return loss

    def logpyz(self, z, y):
        y_z_dis = self.classifier(z.view(-1, self.z_dim))
        logpy_z = torch.sum(y * torch.log(y_z_dis + 1e-8), dim=1)
        logp_z = dis.Normal(self.prior_mu.to(self.device), self.prior_std.to(self.device)).log_prob(z).sum(-1)
        return logpy_z + logp_z

    def x_forward(self, x):
        z_mu, z_std = self.x_encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        z = eps.mul(z_std).add_(z_mu)
        x_out = self.decoder(z)
        kl = batch_KL_diag_gaussian_std(z_mu, z_std, self.prior_mu.to(self.device), self.prior_std.to(self.device))
        neg_l = self.criterion(x, x_out)
        loss = torch.mean(neg_l + kl, dim=0)
        return loss

    def classify(self, x, z_num=1):
        batch_size = x.size(0)
        z_mu, z_std = self.x_encoder(x)
        z_batch_size = z_mu.size(0)
        eps = torch.randn(z_num, z_batch_size, self.z_dim).to(self.device)
        zs = eps.mul(z_std.unsqueeze(0)).add_(z_mu.unsqueeze(0))
        y_dis = self.classifier(zs.view(-1, self.z_dim)).view(z_num, batch_size, self.class_num).mean([0])
        return y_dis

    def conditional_sample(self, y):
        loss_list = []
        self.eval()
        y = one_hot(y, num_classes=10).to(opt['device'])
        z_opt = torch.randn(self.z_dim, requires_grad=True)
        optimizer = optim.Adam([z_opt], lr=5e-2)
        for i in range(0, 100):
            optimizer.zero_grad()
            L = -self.logpyz(z_opt, y)
            L.backward()
            optimizer.step()
            loss_list.append(L.item())
        with torch.no_grad():
            vae_out = self.decoder(z_opt.view(1, -1))
            x_sample = self.sample_op(vae_out)
            if self.dataset == 'BinaryMNIST':
                return x_sample, loss_list
            else:
                return rescaling_inv(x_sample), loss_list

    def sample(self, num=100):
        self.eval()
        with torch.no_grad():
            z = torch.randn([num, self.z_dim])
            vae_out = self.decoder(z)
            x_sample = self.sample_op(vae_out)
            if self.dataset == 'BinaryMNIST':
                return x_sample
            else:
                return rescaling_inv(x_sample)


def EncoderTrain(model, train_data, opt):
    loss_list = []
    model.x_encoder.train()
    model.decoder.eval()
    for x, _ in tqdm(train_data):
        if opt['data_set'] != 'BinaryMNIST':
            x = rescaling(x)
        model.encoder_optimizer.zero_grad()
        L = model.x_forward(x.to(opt['device']))
        L.backward()
        model.encoder_optimizer.step()
        loss_list.append(L.item())
    return loss_list


def UnsupTrain(model, train_data, opt):
    loss_list = []
    model.train()
    for x, _ in tqdm(train_data):
        if opt['data_set'] != 'BinaryMNIST':
            x = rescaling(x)
        model.optimizer.zero_grad()
        L = model.x_forward(x.to(opt['device']))
        L.backward()
        model.optimizer.step()
        loss_list.append(L.item())
    return loss_list


def BPDEval(model, dataloader, opt):
    with torch.no_grad():
        model.eval()
        eval_BPD = 0.
        for x, _ in dataloader:
            if opt['data_set'] != 'BinaryMNIST':
                x = rescaling(x)
            L = model.x_forward(x.to(opt['device']))
            eval_BPD += L.item() / np.log(2.0)
        return eval_BPD / (len(dataloader) * np.prod(x.size()[-3:]))


def JointTrain(model, train_data, opt):
    loss_list = []
    model.train()
    for x, y in tqdm(train_data):
        if opt['data_set'] != 'BinaryMNIST':
            x = rescaling(x)
        y = one_hot(y, num_classes=10)
        model.optimizer.zero_grad()
        L = model.joint_forward(x.to(opt['device']), y.to(opt['device']))
        L.backward()
        model.optimizer.step()
    loss_list.append(L.item())
    return loss_list


if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=int, required=True)
    # gpu= parser.parse_args().gpu
    opt = {}
    # opt=get_device(opt,gpu_index=gpu)

    opt['device'] = 'cpu'
    opt['data_set'] = 'BinaryMNIST'
    opt['dataset_path'] = '../data/'
    opt['epochs'] = 10
    opt['batch_size'] = 64
    opt['test_batch_size'] = 10
    opt['alpha_coef'] = 1.0
    opt['lr'] = 1e-3
    opt['z_dim'] = 10
    opt['seed'] = 0

    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    train_data_loader, test_data_loader, _ = LoadData(opt)
    model = VCCA(opt)

    # for i in range(0,opt['epochs']):
    #     JointTrain(model,train_data_loader,opt)
    # bpd=BPDEval(model,test_data_loader,opt)
    # print('epoch:',i,'BPD:',bpd)

    # torch.save(model.state_dict(),'./save/Joint_epoch10.pth')

    # model.load_state_dict(torch.load('./save/Joint_epoch10.pth'))
    # bpd=BPDEval(model,test_data_loader,opt)
    # print('BPD:',bpd)

    # for i in range(0,opt['epochs']):
    #     EncoderTrain(model,train_data_loader,opt)
    #     bpd=BPDEval(model,test_data_loader,opt)
    #     print('epoch:',i,'BPD:',bpd)

    # torch.save(model.state_dict(),'./save/all.pth')

    model.load_state_dict(torch.load('./save/all.pth'))
    # samples=model.sample(100)
    # gray_save_many(samples,'./img/yo.png')

    sample, loss = model.conditional_sample([4])
    plt.plot(loss)
    plt.savefig('./img/sample_loss')
    plt.close()
    plt.imshow(sample.view(28, 28))
    plt.show()


