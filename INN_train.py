######################################################################
### import libraries
######################################################################
import os
import sys
import tempfile
import numpy as np
import h5py
import torch
import argparse
from torch import Tensor
import logging

from time import time
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, InvertibleSigmoid
from Callbacks import EarlyStopping

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


EARLY_STOP = True
print(f"Early Stopping: {EARLY_STOP}")

#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
#torch.manual_seed(1234)
# Random number generators in other libraries
#np.random.seed(1234)

#######################################################################
#Setting up the hyperparameters
#######################################################################
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Inverse Current Tomography training')
parser.add_argument('--drop', type=int, default=5, metavar='n', help='sensors to be dropped (default:5)')
args = parser.parse_args()

# Hyperparameters:
batch_size = 100
lr = 0.0001
num_coupling_nodes = 4
num_subnet_layers = 1
n_epochs = 300

SENSOR_REDUCTION = True
if SENSOR_REDUCTION:
    naive = False
    PCA = True
    clust = False
    random = False
    if naive:
        n_sensors = 16
        print("Strategy Used: Naive")
        print(f"Sensors Active: {n_sensors}")
    elif PCA:
        n_sensors = 100 - (args.drop)
        print("Strategy Used: PCA")
        print(f"Sensors Active: {n_sensors}")
    elif clust:
        n_sensors = 100 - (args.drop)
        print("Strategy Used: Clustering")
        print(f"Sensors Active: {n_sensors}")
    elif random:
        n_sensors = 100 - (args.drop)
        print("Strategy Used: Random Sampling")
        print(f"Sensors Active: {n_sensors}")

if SENSOR_REDUCTION:
    if naive:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/' , exist_ok=True)
    elif PCA:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/' , exist_ok=True)
    elif clust:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/' , exist_ok=True)
    elif random:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/' , exist_ok=True)
else:
    os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/' , exist_ok=True)

# EXTRA:
train_split = 8000
val_split = 2000
l2_reg = 2e-5
min_epochs = 1

#######################################################################################
#import the training, validation and test data along with connductivity map coordinates
#######################################################################################
if SENSOR_REDUCTION:
    if naive:
        hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_16sens_bottom-right.h5', 'r')
    elif PCA:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_PCA_{n_sensors}sens_SA.h5', 'r')
    elif clust:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_clust_{n_sensors}sens.h5', 'r')
    elif random:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_random_{n_sensors}.h5', 'r')
else:
    hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_bx_ss.h5', 'r')
bx_tot_train_norm = np.array(hf.get('bx_train_ss'))
bx_tot_val_norm = np.array(hf.get('bx_val_ss'))
hf.close()

hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_ct_ss.h5', 'r')
ct_tot_train_norm = np.array(hf.get('ct_train_ss'))
ct_tot_val_norm = np.array(hf.get('ct_val_ss'))
hf.close()

    
######################################################################
#Convert the numpy arrays into pytorch tensors 
######################################################################
bx_tensor_train_norm  = torch.tensor(bx_tot_train_norm, dtype=torch.float).to(device)
ct_tensor_train_norm   = torch.tensor(ct_tot_train_norm, dtype=torch.float).to(device)

bx_tensor_val_norm = torch.tensor(bx_tot_val_norm, dtype=torch.float).to(device)
ct_tensor_val_norm = torch.tensor(ct_tot_val_norm, dtype=torch.float).to(device)


#######################################################################
#Define dataset and DataLoaders
#######################################################################

# Define dataset and DataLoaders 
train_dataset = torch.utils.data.TensorDataset(ct_tensor_train_norm, bx_tensor_train_norm)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(ct_tensor_val_norm, bx_tensor_val_norm)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset & DataLoaders created......")
      
#######################################################################
#Setting up the model#
#######################################################################
ndim_ct = ct_tensor_train_norm.shape[1]
num_features = int(128)
ndim_bx = bx_tensor_train_norm.shape[1]
ndim_z =  ct_tensor_train_norm.shape[1] - bx_tensor_train_norm.shape[1]

print('ndim_ct (Conductivity Dim) [x]: ', ndim_ct)
print('ndim_bx (Mag Field Dim) [y]: ', ndim_bx)
print('ndim_z (Latent Dim) [z]: ', ndim_z)

def subnet_fc(c_in, c_out):
    layers = [nn.Linear(c_in, num_features), nn.Tanh()]   #tanh is smooth suitable for regression
    for i in range(num_subnet_layers):
        layers.append(nn.Linear(num_features,num_features))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(num_features, c_out))
    return nn.Sequential(*layers)

nodes = [InputNode(ndim_ct, name='input')]
for k in range(num_coupling_nodes):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))
                      
nodes.append(OutputNode(nodes[-1], name='output'))
model = ReversibleGraphNet(nodes, verbose=False).to(device)

print(model)

#####################################################################
#loss functions#
#####################################################################
def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz
    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())
    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1
    return torch.mean(XX + YY - 2.*XY)   
    
def mse_fit(input, target):
    return torch.mean((input - target)**2)
    
def mae_fit(input, target):
    return torch.mean(torch.abs(input - target))
     
def l2_fit(input, target):
    return torch.sqrt(torch.mean((input - target)**2))
    
class SoftDiceLoss(nn.Module): 
    "Soft dice loss based on a measure of overlap between prediction and ground truth"
    def __init__(self, epsilon=1e-6, c = 1):
        super().__init__()
        self.epsilon = epsilon
        self.c = 1
    
    def forward(self, x:Tensor, y:Tensor):
        intersection = 2 * ((x*y).sum())
        union = (x**2).sum() + (y**2).sum() 
        return 1 - (intersection / (union + self.epsilon))    
    
loss_ = nn.MSELoss()                
loss_backward = nn.BCELoss()
loss_latent = MMD_multiscale
loss_fit = l2_fit    

####################################################################
#Training and validation functions
####################################################################                    
# Training and validation functions
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
# Initialize the optimizer with the model's parameters
optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9), eps=1e-6, weight_decay=l2_reg)   
    
    
@torch.enable_grad()
def train(epoch):
    model.train()
    t_start = time()
    loss_train_ct_fit = 0
    for batch_idx, (x, y) in enumerate(train_loader):  
        z = torch.randn(batch_size, ndim_z).cuda()         
        optimizer.zero_grad()
        output, jacobian = model(torch.cat((z,y.cuda()), 1)) 
        #output, jacobian = model(x.cuda())
        #loss_bx_fit = loss_fit(output[:,ndim_z:], y.cuda())
        loss_ct_fit = loss_fit(output, x.cuda())
        #output, jacobian = model(torch.cat((z,y.cuda()), 1))        
        loss = loss_ct_fit
        loss.backward()
        optimizer.step()
        loss_train_ct_fit += loss_ct_fit.data.item()
    return loss_train_ct_fit

     
@torch.no_grad()
def val(epoch):
    model.eval()
    loss_val_ct_fit = 0
    for batch_idx, (x, y) in enumerate(val_loader):
        z = torch.randn(batch_size, ndim_z).cuda()          
        output, jacobian = model(torch.cat((z,y.cuda()), 1)) 
        #output, jacobian = model(x.cuda())
        #loss_bx_fit = loss_fit(output[:,ndim_z:], y.cuda())
        loss_ct_fit = loss_fit(output, x.cuda())
        loss_val_ct_fit += loss_ct_fit.data.item()
    return loss_val_ct_fit
    
   
# Initialize the early stopping object
patience = 10
if EARLY_STOP:
    if SENSOR_REDUCTION:
        if naive:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/model_checkpoint.pth')  
        elif PCA:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_checkpoint.pth')
        elif clust:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_checkpoint.pth')
        elif random:
            early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_checkpoint.pth')
    else:
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_checkpoint.pth')     
    
    
try:
    t_start = time()
    epoch_loss_train_ct_fit = []
    epoch_loss_val_ct_fit = []
    for e in range(n_epochs):
        t_start = time()
        loss_train_ct_fit = train(e)
        t_end = time()
        loss_val_ct_fit = val(e)
        epoch_loss_train_ct_fit.append(loss_train_ct_fit/ len(train_loader))
        epoch_loss_val_ct_fit.append(loss_val_ct_fit/ len(val_loader))
        print(f'Time: {t_end - t_start} Epoch: {e} Train ct: {loss_train_ct_fit/ len(train_loader)} Val ct: {loss_val_ct_fit/ len(val_loader)}')  
        if e>=min_epochs:
            if EARLY_STOP:
                early_stopping(loss_val_ct_fit, model, e)
                if early_stopping.early_stop:
                    print("---------------------EARLY STOPPING----------------------")
                    break 
                    
    last_trained_epoch = e-patience+1
       
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_train_ct_fit': epoch_loss_train_ct_fit,'loss_val_ct_fit': epoch_loss_val_ct_fit}
    
except KeyboardInterrupt:
    pass

if SENSOR_REDUCTION:
    if naive:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right' , exist_ok=True)
    elif PCA:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}' , exist_ok=True)
    elif clust:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}' , exist_ok=True) 
    elif random:
        os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}' , exist_ok=True)
else:    
    os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) , exist_ok=True) 
    
    
if SENSOR_REDUCTION:
    if naive:
        torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right.pth')
    elif PCA:
        torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}.pth')
    elif clust:
        torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}.pth')
    elif random:
        torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}.pth')
else:
    torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + '.pth')


if SENSOR_REDUCTION:
    if naive:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/Info.log', level=logging.INFO)
    elif PCA:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
    elif clust:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
    elif random:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
else:   
    logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Info.log', level=logging.INFO)
logging.info("-------------------- TRAINING ------------------------------")
logging.info(f'CN: {num_coupling_nodes}')
logging.info(f'SL: {num_subnet_layers}') 
if EARLY_STOP:
    logging.info(f"Last trained epoch: {last_trained_epoch}") 
    logging.info(f"Loss @ last trained epoch: {epoch_loss_val_ct_fit[-(patience+1)]}")  

#######################################################################
#Plot training and validation loss curves
#######################################################################
if SENSOR_REDUCTION:
    if naive:
        checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}_bottom-right.pth')
    elif PCA:
        checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
    elif clust:
        checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
    elif random:
        checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
else:
    checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + '.pth')

loss_train_fit = np.array(checkpoint['loss_train_ct_fit'])
loss_val_fit = np.array(checkpoint['loss_val_ct_fit'])

plt.plot(loss_train_fit, 'g', label = 'train fit ct')
plt.plot(loss_val_fit, 'b', label = 'valid fit ct')
plt.xlabel('epoch')
plt.ylabel('L2 loss')
plt.legend()
plt.title('Epoch:' + str(last_trained_epoch) + ', C:' + str(num_coupling_nodes) + ', L:' + str(num_subnet_layers +3))
if SENSOR_REDUCTION:
    if naive:
        plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}_bottom-right/Loss_curve_ct.png', dpi=200)
    elif PCA:
        plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Loss_curve_ct.png', dpi=200)
    elif clust:
        plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Loss_curve_ct.png', dpi=200)
    elif random:
        plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Loss_curve_ct.png', dpi=200)
else:
    plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Loss_curve_ct.png', dpi=200)
plt.close()

#with open('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/log_CN_4.txt', "a") as f:
#    f.write(f"\nBS = {batch_size}, LR = {lr}, SL = {num_subnet_layers}, Epochs = {last_trained_epoch}, Avg. loss = {epoch_loss_val_ct_fit[-(patience+1)]}") 
