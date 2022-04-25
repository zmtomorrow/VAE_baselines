import torch
import torch.nn as nn
from network import *
from distributions import *
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import torch.nn.functional as F
from utils import batch_KL_diag_gaussian_std



class VAE(nn.Module):
    def __init__(self,  opt):
        super().__init__()
        self.z_dim=opt['z_channels']*64
        self.device=opt['device']
        self.encoder=fc_encoder(latent_channels=opt['z_channels'])
        if opt['x_dis']=='MixLogistic':
            self.decoder=fc_decoder(latent_channels=opt['z_channels'],out_channels=100)       
            self.criterion  = lambda  data,params :discretized_mix_logistic_uniform(data, params)
            self.sample_op = lambda  params : discretized_mix_logistic_sample(params)
        elif opt['x_dis']=='Logistic':
            self.decoder=fc_decoder(latent_channels=opt['z_channels'],out_channels=9)       
            self.criterion  = lambda  data,params :discretized_logistic(data, params)
            self.sample_op = lambda params: discretized_logistic_sample(params)
            
        self.prior_mu=torch.zeros(self.z_dim, requires_grad=False)
        self.prior_std=torch.ones(self.z_dim, requires_grad=False)
        self.params = list(self.parameters())

    def posterior_forward(self, params,x):
        z_mu, z_std = params[0],F.softplus(params[1])
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo)/np.log(2.)
    
    def forward(self, x):
        z_mu, z_std = self.encoder(x)
        eps = torch.randn_like(z_mu).to(self.device)
        zs = eps.mul(z_std).add_(z_mu)
        pxz_params = self.decoder(zs)
        loglikelihood = self.criterion(x, pxz_params)
        kl = batch_KL_diag_gaussian_std(z_mu,z_std,self.prior_mu.to(self.device),self.prior_std.to(self.device))
        elbo = loglikelihood - kl
        return torch.mean(elbo)/np.log(2.)
    

    def sample(self,num=100):
        with torch.no_grad():
            eps = torch.randn(num,self.z_dim).to(self.device)
            pxz_params = self.decoder(eps)
            return self.sample_op(pxz_params)








    
    
    
    
