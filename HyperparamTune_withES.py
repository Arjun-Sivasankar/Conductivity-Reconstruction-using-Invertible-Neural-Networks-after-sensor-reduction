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
from Callbacks import EarlyStopping
import logging

from time import time
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom, InvertibleSigmoid
from Callbacks import EarlyStopping

import optuna

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size=100

EARLY_STOP = True
print(f"Early Stoppage: {EARLY_STOP}")

## Setting up the data
num_tot_samples = 10000
train_split = 8000
val_split = 2000

#######################################################################################
#import the training, validation and test data along with connductivity map coordinates
#######################################################################################
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_bx_ss.h5', 'r')
bx_tot_train_norm = np.array(hf.get('bx_train_ss'))
bx_tot_val_norm = np.array(hf.get('bx_val_ss'))
hf.close()

hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_ct_ss.h5', 'r')
ct_tot_train_norm = np.array(hf.get('ct_train_ss'))
ct_tot_val_norm = np.array(hf.get('ct_val_ss'))
hf.close()

print("Train and Val data loaded.....")

######################################################################
#Convert the numpy arrays into pytorch tensors 
######################################################################
bx_tensor_train_norm  = torch.tensor(bx_tot_train_norm, dtype=torch.float).to(device)
ct_tensor_train_norm   = torch.tensor(ct_tot_train_norm, dtype=torch.float).to(device)

bx_tensor_val_norm = torch.tensor(bx_tot_val_norm, dtype=torch.float).to(device)
ct_tensor_val_norm = torch.tensor(ct_tot_val_norm, dtype=torch.float).to(device)

ndim_ct = ct_tensor_train_norm.shape[1]
num_features = int(128)
ndim_bx = bx_tensor_train_norm.shape[1]
ndim_z =  ct_tensor_train_norm.shape[1] - bx_tensor_train_norm.shape[1]

def train_and_evaluate_model(hyperparams):
    # Extract hyperparameters
    batch_size = 100
    lr = hyperparams['lr']
    num_coupling_nodes = hyperparams['num_coupling_nodes']
    num_subnet_layers = hyperparams['num_subnet_layers']
    n_epochs = hyperparams['n_epochs']
    
    # Define dataset and DataLoaders 
    train_dataset = torch.utils.data.TensorDataset(ct_tensor_train_norm, bx_tensor_train_norm)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(ct_tensor_val_norm, bx_tensor_val_norm)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Dataset & DataLoaders created......")
    
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
    #print(model)
    
    ## LOSS FUNCTION:
    def l2_fit(input, target):
        return torch.sqrt(torch.mean((input - target)**2))

    loss_fit = l2_fit
    l2_reg = 2e-5

    # Training and validation functions
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    # Initialize the optimizer with the model's parameters
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9), eps=1e-6, weight_decay=l2_reg)
    
    ## TRAINING AND VALIDATION FUNCTION:
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
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_checkpoint.pth')
    
    os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) , exist_ok=True) 
    
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
        print(f'Time: {t_end - t_start} Epoch: {e+1} Train ct: {loss_train_ct_fit/ len(train_loader)} Val ct: {loss_val_ct_fit/ len(val_loader)}')
        # Early stopping call
        if e>=1:
            if EARLY_STOP:
                early_stopping(loss_val_ct_fit, model, e)
                if early_stopping.early_stop:
                    print("---------------------EARLY STOPPING----------------------")
                    break 
            
    print("Training Complete......")  
     
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_train_ct_fit': epoch_loss_train_ct_fit,'loss_val_ct_fit': epoch_loss_val_ct_fit}
    torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + '.pth')
    
    avg_val_loss = loss_val_ct_fit/ len(val_loader)
    last_trained_epoch = early_stopping.last_checkpoint_epoch
    last_trained_loss = early_stopping.last_checkpoint_loss/(len(val_loader))
    print(f'Last trained Epoch: {last_trained_epoch+1}')
    print(f'Last trained loss: {last_trained_loss}')
     
    logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Info.log', level=logging.INFO)
    logging.info("-------------------- TRAINING ------------------------------")
    logging.info(f'CN: {num_coupling_nodes}')
    logging.info(f'SL: {num_subnet_layers}')
    logging.info(f'LR: {lr}')
    logging.info(f'Epochs: {n_epochs}')
    logging.info(f'Avg. loss = {loss_val_ct_fit/ len(val_loader)}')  
    if EARLY_STOP:
        logging.info(f"Last trained epoch: {last_trained_epoch+1}") 
        logging.info(f"Loss @ last trained epoch: {last_trained_loss}")  
    
    #######################################################################
    #Plot training and validation loss curves
    #######################################################################
    checkpoint = torch.load('/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr)  + '.pth')
    loss_train_fit = np.array(checkpoint['loss_train_ct_fit'])
    loss_val_fit = np.array(checkpoint['loss_val_ct_fit'])
    
    plt.plot(loss_train_fit, 'g', label = 'train fit ct')
    plt.plot(loss_val_fit, 'b', label = 'valid fit ct')
    plt.xlabel('epoch')
    plt.ylabel('L2 loss')
    plt.legend()
    plt.title('Epoch:' + str(n_epochs) + ', C:' + str(num_coupling_nodes) + ', L:' + str(num_subnet_layers +3))
    plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Loss_curve_ct.png', dpi=200)
    plt.show()
    plt.close()
    
    return avg_val_loss, last_trained_loss, last_trained_epoch, patience
    

###################################################################
# Setting the Objective for tuning
###################################################################
def objective(trial):
    # Define the hyperparameters to be tuned
    hyperparams = {
        'lr': trial.suggest_loguniform('lr', 1e-4, 0.0005),
        'num_coupling_nodes': trial.suggest_int('num_coupling_nodes', 1, 6),
        'num_subnet_layers': trial.suggest_int('num_subnet_layers', 1, 3),
        'n_epochs': trial.suggest_int('n_epochs', 1, 300)
    }
    
    print("###################################")
    print("The Hyperparameters used:")
    print(f"LR: {hyperparams['lr']}")
    print(f"num_coupling_nodes: {hyperparams['num_coupling_nodes']}")
    print(f"num_subnet_layers: {hyperparams['num_subnet_layers']}")
    print(f"n_epochs: {hyperparams['n_epochs']}")
    print("###################################\n")

    # Call the training and evaluation function
    val_loss, last_val_loss, last_epoch, patience = train_and_evaluate_model(hyperparams)
    
    # Log trial details
    with open("/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/HypTune_log.txt", "a") as log_file:
        log_file.write(f"Trial {trial.number}: {hyperparams}, Loss: {last_val_loss}, Last Epoch: {last_epoch+1}\n")
    
    return last_val_loss

# Create a study object and specify the optimization direction as 'minimize'.
study = optuna.create_study(direction='minimize')

# Start the optimization - you can adjust the number of trials
study.optimize(objective, n_trials=50)

# Optional: Log the best hyperparameters
with open("/beegfs/.global1/ws/arsi805e-Test/New/TunedModels/EarlyStop/HypTune_log.txt", "a") as log_file:
    log_file.write(f"Best hyperparameters: {study.best_trial.params}\n")

# Print the best hyperparameters
print('Best hyperparameters:', study.best_trial.params)

print("\n Tuning Complete.....")

