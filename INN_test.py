import numpy as np
import h5py
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pickle import dump, load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from torch import Tensor
import os
import logging

from time import time
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, InvertibleSigmoid

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EARLY_STOP = True

SENSOR_REDUCTION = True
print(f"Sensor Reduction Strategy: {SENSOR_REDUCTION}")
if SENSOR_REDUCTION:
    rand = True
    naive = False
    PCA = False
    clust = False
    n_sensors = 90

batch_size = 100
lr = 0.0001
num_coupling_nodes = 4
num_subnet_layers = 1
n_epochs = 300

## Setting up the data
num_tot_samples = 10000
train_split = 8000
val_split = 2000

if SENSOR_REDUCTION:
    if naive:
        ss_scaler_bx = load(open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_PCA_16sens_top-mid.pkl', 'rb'))
    elif PCA:
        ss_scaler_bx = load(open(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_PCA_{n_sensors}sens_SA.pkl', 'rb'))
    elif rand:
        ss_scaler_bx = load(open(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_random_{n_sensors}.pkl', 'rb'))
    elif clust:
        ss_scaler_bx = load(open(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_clust_{n_sensors}sens.pkl', 'rb'))
else:
    ss_scaler_bx = load(open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/ss_scaler_bx.pkl', 'rb'))
#save the scaled data
if SENSOR_REDUCTION:
    if naive:
        hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_16sens_top-mid.h5', 'r')
    elif PCA:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_PCA_{n_sensors}sens_SA.h5', 'r')
    elif clust:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_clust_{n_sensors}sens.h5', 'r')
    elif rand:
        hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_random_{n_sensors}.h5', 'r')
else:
    hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_bx_ss.h5', 'r')
bx_tot_train_norm = np.array(hf.get('bx_train_ss'))
bx_tot_val_norm = np.array(hf.get('bx_val_ss'))
hf.close()

ss_scaler_ct = load(open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/ss_scaler_ct.pkl', 'rb'))
#save the scaled data
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_ct_ss.h5', 'r')
ct_train_ss = np.array(hf.get('ct_train_ss'))
ct_val_ss = np.array(hf.get('ct_val_ss'))
hf.close()

print("Train and Val data loaded.....")

### Mag. data scaled
#bx_tot_train_norm = bx_train_ss
#bx_tot_val_norm = bx_val_ss

## Current data unscaled
ct_tot_train_norm = ct_train_ss
ct_tot_val_norm = ct_val_ss

bx_tensor_train_norm  = torch.tensor(bx_tot_train_norm, dtype = torch.float) 
ct_tensor_train_norm   = torch.tensor(ct_tot_train_norm, dtype = torch.float)

bx_tensor_val_norm = torch.tensor(bx_tot_val_norm, dtype = torch.float)
ct_tensor_val_norm = torch.tensor(ct_tot_val_norm, dtype = torch.float)

# Define dataset and DataLoaders 
train_dataset = torch.utils.data.TensorDataset(ct_tensor_train_norm, bx_tensor_train_norm)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(ct_tensor_val_norm, bx_tensor_val_norm)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset & DataLoaders created......")

ndim_ct = ct_tensor_train_norm.shape[1]
num_features = int(128)
ndim_bx = bx_tensor_train_norm.shape[1]
ndim_z =  ndim_ct - ndim_bx
print('ndim_ct (current Dim) [x]: ', ndim_ct)
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

model = ReversibleGraphNet(nodes, verbose=False)
print(model)

###################################################################
#Testing the model
###################################################################
print("Model Testing Initiated.....")
try:
    #pred_bx_val = []
    pred_ct_val = []
    #gt_bx_val = bx_tensor_val_norm.clone().detach().cpu().numpy()
    gt_ct_val = ct_tensor_val_norm.clone().detach().cpu().numpy()
    if SENSOR_REDUCTION:
        if naive:
            checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}_top-mid.pth')
        elif PCA:
            checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
        elif clust:
            checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
        elif rand:
            checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + f'_{n_sensors}.pth')
    else:
        checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + '.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    batch_idx = 0 
    for i in tqdm(range(gt_ct_val.shape[0])):
        z = torch.randn(1, ndim_z)   
        batch_idx += 1
        ct_output, ctctacobian = model(torch.cat((z,bx_tensor_val_norm[i:i+1,]), 1))
        #ct_output = model(bx_tensor_val_norm[i:i+1,].cuda())        
        pred_ct_val.append(ct_output.clone().detach().cpu().numpy())
    if SENSOR_REDUCTION:
        if naive:
            hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/val_data.h5', 'w')
        elif PCA:
            hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/val_data.h5', 'w')
        elif rand:
            hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/val_data.h5', 'w')
        elif clust:
            hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/val_data.h5', 'w')
    else:
        hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/val_data.h5', 'w')
    hf.create_dataset('pred_ct_val', data = pred_ct_val)
    hf.create_dataset('gt_ct_val', data  = gt_ct_val)
    hf.close()
except KeyboardInterrupt:
    pass
    
print("MODEL TESTING COMPLETE.......")

###################################################################
#Accuracy Calc.
###################################################################
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_permute_train_val_split.h5', 'r')
ct_val = np.array(hf.get('ct_val'))
hf.close()
#ct_coord = np.array(hf.get('ct_coord'))

hf = h5py.File("/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/ct_coord.h5", 'r')
ct_coord = np.array(hf.get('ct_coord'))
hf.close()

pred_ct_val = np.array(pred_ct_val)
pred_ct_val = pred_ct_val.reshape(pred_ct_val.shape[0], pred_ct_val.shape[1]*pred_ct_val.shape[2])

## arrays that go in inverse_transform should be of (2000, 510)
pred_ct_val = ss_scaler_ct.inverse_transform(pred_ct_val)
print("Shape of predictions: ", pred_ct_val.shape)
# groundtruth_value = ss_scaler_ct.inverse_transform(gt_ct_val)
groundtruth_value = gt_ct_val

preds = pred_ct_val.flatten()
groundtruth_value = groundtruth_value.flatten()
ctval = ct_val.flatten()

tolerance = 0.1  # Adctust this as needed for your application

# Calculate the differences and check if they're within the tolerance
accurate_predictions = np.abs(ctval - preds) <= tolerance
print("\n ----------------------------------------------- \n")
print(accurate_predictions)

# Calculate accuracy
accuracy = np.mean(accurate_predictions)
accuracy_percentage = accuracy * 100
print("\n ----------------------------------------------- \n")
print(f"Accuracy within a tolerance of {tolerance}: {accuracy_percentage} %")
# logging.info(f"\n [VALIDATION] Accuracy within a tolerance of {tolerance}: {accuracy_percentage} %")
print(" ----------------------------------------------- \n")

###################################################################
#GT vs Pred Plot
###################################################################
for i in tqdm(range(0, 100)):
    x = ct_coord[0,:,0].reshape(34,15)
    y = ct_coord[0,:,1].reshape(34,15)
    
    # First plot
    z = ct_val[i,:].reshape(34,15)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))  # Adctust figsize as needed
    plt_n = ax1.pcolormesh(x*100, y*100, z)
    ax1.set_title('GroundTruth')
    ax1.set_xlabel("x-coordinates [cm]", fontsize=18)
    ax1.set_ylabel("y-coordinates [cm]", fontsize=18)
    ax1.set_xticks((-8, -4, 0, 4, 8))
    ax1.set_yticks((-4, -2, 0, 2, 4))
    c = fig.colorbar(plt_n, ax=ax1, fraction=0.046 * 15 / 34, pad=0.01)
    c.set_label("$\sigma_{rel}$", fontsize=18)
    ax1.set_aspect('equal')

    # Second plot
    z = pred_ct_val[i,:].reshape(34,15)
    plt_n = ax2.pcolormesh(x*100, y*100, z, vmin=0.0, vmax=1.0)
    ax2.set_title('Prediction')
    ax2.set_xlabel("x-coordinates [cm]", fontsize=18)
    ax2.set_ylabel("y-coordinates [cm]", fontsize=18)
    ax2.set_xticks((-8, -4, 0, 4, 8))
    ax2.set_yticks((-4, -2, 0, 2, 4))
    c = fig.colorbar(plt_n, ax=ax2, fraction=0.046 * 15 / 34, pad=0.01)
    c.set_label("$\sigma_{rel}$", fontsize=18)
    ax2.set_aspect('equal')

    plt.tight_layout()
    
    # Directory path
    if SENSOR_REDUCTION:
        if naive:
            dir_path = f'/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_{batch_size}_{num_coupling_nodes}_{num_subnet_layers +3}_{n_epochs}_{lr}_{n_sensors}/pred'
        elif PCA:
            dir_path = f'/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_{batch_size}_{num_coupling_nodes}_{num_subnet_layers +3}_{n_epochs}_{lr}_{n_sensors}/pred'  
        elif clust:
            dir_path = f'/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_{batch_size}_{num_coupling_nodes}_{num_subnet_layers +3}_{n_epochs}_{lr}_{n_sensors}/pred' 
        elif rand:
            dir_path = f'/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_{batch_size}_{num_coupling_nodes}_{num_subnet_layers +3}_{n_epochs}_{lr}_{n_sensors}/pred' 
    else:
        dir_path = f'/beegfs/.global1/ws/arsi805e-Test/New/Models/model_{batch_size}_{num_coupling_nodes}_{num_subnet_layers +3}_{n_epochs}_{lr}/pred'
    os.makedirs(dir_path, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{dir_path}/pred_{i+1}.png', dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"Preds stored in location: {dir_path}")

if SENSOR_REDUCTION:
    if naive:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
    elif PCA:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
    elif rand:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
    elif clust:
        logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Clustering/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + f'_{n_sensors}/Info.log', level=logging.INFO)
else:
    logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Models/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Info.log', level=logging.INFO)  
logging.info("-------------------- TESTING ------------------------------")
logging.info(f"Accuracy within a tolerance of {tolerance}: {accuracy_percentage} %")
