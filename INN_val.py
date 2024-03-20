######################################################################
### import libraries
######################################################################
import os
import sys
import tempfile
import numpy as np
import h5py
import torch
import horovod.torch as hvd
import argparse
from time import time
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())

#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)

#######################################################################
#Setting up the hyperparameters
#######################################################################
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Inverse Current Tomography training')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--num_coupling_nodes', type=int, default=1, metavar='coupling blocks', help='Number of coupling block in INN')
parser.add_argument('--num_subnet_layers', type=int, default=1, metavar='subnet layers', help='Number of subnet layers in each coupling block + 3')
parser.add_argument('--n_epochs', type=int, default=100, metavar='epochs', help='Number of epochs')
args = parser.parse_args()

train_split = 8000
val_split = 2000
l2_reg = 2e-5


#######################################################################
#import the training, validation and test data along with ct coordinates
#######################################################################
hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_bx_ss.h5', 'r')
bx_tot_train_norm = np.array(hf.get('bx_train_ss'))
bx_tot_val_norm = np.array(hf.get('bx_val_ss'))
hf.close()

hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_ct_ss.h5', 'r')
ct_tot_train_norm = np.array(hf.get('ct_train_ss'))
ct_tot_val_norm = np.array(hf.get('ct_val_ss'))
hf.close()


######################################################################
#Convert the numpy arrays into pytorch tensors 
######################################################################
bx_tensor_train_norm  = torch.tensor(bx_tot_train_norm, dtype = torch.float) 
ct_tensor_train_norm   = torch.tensor(ct_tot_train_norm, dtype = torch.float)

bx_tensor_val_norm = torch.tensor(bx_tot_val_norm, dtype = torch.float)
ct_tensor_val_norm = torch.tensor(ct_tot_val_norm, dtype = torch.float)


#######################################################################
#Define dataset and DataLoaders
#######################################################################


train_dataset = torch.utils.data.TensorDataset(ct_tensor_train_norm, bx_tensor_train_norm)
# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler = train_sampler)
    
val_dataset = torch.utils.data.TensorDataset(ct_tensor_val_norm, bx_tensor_val_norm)
# Partition dataset among workers using DistributedSampler
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler = val_sampler)
       

#######################################################################
#Setting up the model#
#######################################################################
ndim_ct = ct_tensor_train_norm.shape[1]
num_features = int(128)
ndim_bx = bx_tensor_train_norm.shape[1]
ndim_z =  ct_tensor_train_norm.shape[1] - bx_tensor_train_norm.shape[1]

def subnet_fc(c_in, c_out):
    layers = [nn.Linear(c_in, num_features), nn.Tanh()]   #tanh is smooth suitable for regression
    for i in range(args.num_subnet_layers):
        layers.append(nn.Linear(num_features,num_features))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(num_features, c_out))
    return nn.Sequential(*layers)

nodes = [InputNode(ndim_ct, name='input')]
for k in range(args.num_coupling_nodes):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))
                      
nodes.append(OutputNode(nodes[-1], name='output'))
model = ReversibleGraphNet(nodes, verbose=False).cuda()

    
# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)



###################################################################
#Testing the model
###################################################################
try:
    #pred_bx_val = []
    pred_ct_val = []
    #gt_bx_val = bx_tensor_val_norm.clone().detach().cpu().numpy()
    gt_ct_val = ct_tensor_val_norm.clone().detach().cpu().numpy()
    checkpoint = torch.load('/home/h1/s8993054/Inverse_Current_Tomography/INN_25mm/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr) + '/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr)  + '.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    batch_idx = 0 
    for i in range(gt_ct_val.shape[0]):
        z = torch.randn(1, ndim_z).cuda()   
        batch_idx += 1
        if batch_idx % 1000 == 0:
            print(batch_idx)
        ct_output, ctacobian = model(torch.cat((z,bx_tensor_val_norm[i:i+1,].cuda()), 1))
        #ct_output = model(bx_tensor_val_norm[i:i+1,].cuda())        
        pred_ct_val.append(ct_output.clone().detach().cpu().numpy())
    hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/INN_25mm/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr) + '/val_data.h5', 'w')
    hf.create_dataset('pred_ct_val', data = pred_ct_val)
    hf.create_dataset('gt_ct_val', data  = gt_ct_val)
    hf.close()
except KeyboardInterrupt:
    pass
    
    
#######################################################################
#Plot training and validation loss curves
#######################################################################
checkpoint = torch.load('/home/h1/s8993054/Inverse_Current_Tomography/INN_25mm/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr) + '/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr)  + '.pth')
loss_train_fit = np.array(checkpoint['loss_train_ct_fit'])
loss_val_fit = np.array(checkpoint['loss_val_ct_fit'])

plt.plot(loss_train_fit, 'g', label = 'train fit ct')
plt.plot(loss_val_fit, 'b', label = 'valid fit ct')
plt.xlabel('epoch')
plt.ylabel('L2 loss')
plt.legend()
plt.title('Epoch:' + str(args.n_epochs) + ', C:' + str(args.num_coupling_nodes) + ', L:' + str(args.num_subnet_layers +3))
plt.savefig('/home/h1/s8993054/Inverse_Current_Tomography/INN_25mm/model_' + str(args.batch_size) + '_' + str(args.num_coupling_nodes) + '_' + str(args.num_subnet_layers +3) + '_' + str(args.n_epochs) + '_' + str(args.lr) + '/Loss_curve_ct.png', dpi=200)
plt.close()





