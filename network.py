import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassNet(nn.Module):
    def __init__(self,opt):
        super(ClassNet, self).__init__()
        self.z_dim=opt['z_dim']
        self.device=opt['z_dim']
        self.fc1= nn.Linear(self.z_dim, 200)
        self.bn=nn.BatchNorm1d(200)
        self.dropout=nn.Dropout(0,1)
        self.fc2 = nn.Linear(200, 10)

    def forward(self,x):
        h=F.relu(self.fc1(x))
        h=self.fc2(self.bn(self.dropout(h)))
        return torch.softmax(h,dim=-1)

    def predict(self,x):
        p=self.forward(x)
        pred=p.argmax(dim=-1)
        return pred


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class densenet_encoder(nn.Module):
    def __init__(self,  input_dim=784, h_dim=500, z_dim=50, if_bn=True):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.input_dim=input_dim
        
        self.fc1 = nn.Linear(input_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc31 = nn.Linear(self.h_dim, self.z_dim)
        self.fc32 = nn.Linear(self.h_dim, self.z_dim)

        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
            self.bn2 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()

    def forward(self, x, y=None):
        if y is not None:
            x = torch.flatten(x, start_dim=1)
            y = torch.flatten(y, start_dim=1)
            x=torch.cat([x, y], dim=1)
        x=x.view(-1,self.input_dim)
        h=F.relu(self.bn1(self.fc1(x)))
        h=F.relu(self.bn2(self.fc2(h)))
        mu=self.fc31(h)
        std=torch.nn.functional.softplus(self.fc32(h))
        return mu, std
        

class densenet_decoder(nn.Module):
    def __init__(self,o_dim=1,h_dim=500, z_dim=50, if_bn=True):
        super().__init__()
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.o_dim=o_dim

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3 = nn.Linear(self.h_dim, self.o_dim*784)
        if if_bn:
            self.bn1 = nn.BatchNorm1d(self.h_dim)
            self.bn2 = nn.BatchNorm1d(self.h_dim)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()

        
    def forward(self,z):
        h=F.relu(self.bn1(self.fc1(z)))
        h=F.relu(self.bn2(self.fc2(h)))
        h=self.fc3(h)
        return h.view(-1,self.o_dim,28,28)

class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class fc_encoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64):
        super(fc_encoder, self).__init__()
        self.latent_channels=latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_channels*2, 1)
        )

    def forward(self, x):
        z=self.encoder(x)
        return z[:,:self.latent_channels,:,:].view(x.size(0),-1),F.softplus(z[:,self.latent_channels:,:,:].view(x.size(0),-1))


class fc_decoder(nn.Module):
    def __init__(self, channels=256, latent_channels=64, out_channels=100):
        super(fc_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d( latent_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, 1)
        )

    def forward(self, z):
#         print('here',z.size(0))
        return  self.decoder(z.view(z.size(0),-1,8,8))
