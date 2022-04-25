import torch
from torch import optim
from utils import *
from model import *
import numpy as np
from tqdm import tqdm


opt = {}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    opt["device"] = torch.device("cuda:0")
    opt["if_cuda"] = True
else:
    opt["device"] = torch.device("cpu")
    opt["if_cuda"] = False

opt['data_set']='CIFAR'
opt['x_dis']='Logistic' ## or MixLogistic 
opt['z_channels']=2 ## 2*64
opt['epochs'] = 1000
opt['dataset_path']='../data/'
opt['save_path']='./save/'
opt['result_path']='./result/'
opt['batch_size'] = 100
opt['test_batch_size']=200
opt['if_regularizer']=False
opt['load_model']=False
opt['lr']=1e-4
opt['data_aug']=False
opt["seed"]=0
opt['if_save']=True
opt['save_epoch']=50
opt['additional_epochs']=100
opt['sample_size']=100
opt['if_save']=True


np.random.seed(opt['seed'])
torch.manual_seed(opt['seed'])

train_data,test_data,train_data_evaluation=LoadData(opt)
model=VAE(opt).to(opt['device'])

if opt['load_model']==True:
    model.load_state_dict(torch.load(opt['save_path']+opt['load_name']))

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

test_BPD_list=[]
for epoch in range(1, opt['epochs'] + 1):
    model.train()
    for x, _ in tqdm(train_data):
        optimizer.zero_grad()
        L = -model(x.to(opt['device']))
        L.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        test_BPD=0.
        for x, _ in test_data:
            test_BPD+=-model(x.to(opt['device'])).item()
        test_BPD=test_BPD/(len(test_data)*np.prod(x.size()[-3:]))

    print('epoch:',epoch,test_BPD)
    test_BPD_list.append(test_BPD)
    np.save(opt['save_path']+'test_BPD',test_BPD_list)
    
    if opt['if_save']:
        torch.save(model.state_dict(),opt['save_path']+'.pth')
