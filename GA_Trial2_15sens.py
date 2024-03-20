import numpy as np
import h5py
import torch
import random
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
import argparse

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

batch_size = 100
lr = 0.0001
num_coupling_nodes = 4
num_subnet_layers = 1
n_epochs = 300

## Setting up the data
num_tot_samples = 10000
train_split = 8000
val_split = 2000
l2_reg = 2e-5

# Load the magnetic data
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_permute_train_val_split.h5', 'r')
bx_train = np.array(hf.get('bx_train'))
bx_val = np.array(hf.get('bx_val'))
ct_train = np.array(hf.get('ct_train'))
ct_val = np.array(hf.get('ct_val'))
hf.close()

hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_ct_ss.h5', 'r')
ct_tot_train_norm = np.array(hf.get('ct_train_ss'))
ct_tot_val_norm = np.array(hf.get('ct_val_ss'))
hf.close()

os.makedirs('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_GA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '_15sens', exist_ok=True)

# Configure the logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the logging level as needed

# Create a custom log format
log_format = "%(message)s"

# Create a file handler and set the formatter
file_handler = logging.FileHandler('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_GA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '_15sens' + '/Info.log')
file_handler.setFormatter(logging.Formatter(log_format))

# Add the file handler to the logger
logger.addHandler(file_handler)

#logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_GA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '/Info.log', level=logging.INFO)

# Function to apply the activation pattern to the sensor data
def apply_activation_pattern(data, sensors_config):
    activation_pattern = np.array(sensors_config)
    return data[:, activation_pattern==1]               # Select only the columns where the corresponding sensor is active
    
# L2 loss function
def l2_fit(input, target):
    return torch.sqrt(torch.mean((input - target)**2))


def evaluate_model_with_sensors(sensors_config):
    print('-'*100)
#    logger.info('-'*100)
    # Apply the sensor configuration to the dataset
    activation_pattern = np.array(sensors_config, dtype=int)
    print(f'Activation pattern: {activation_pattern}')
#    logger.info(f'Activation pattern: {activation_pattern}')
    print(f'Sensors Active: {sum(activation_pattern)}')
#    logger.info(f'Sensors Active: {sum(activation_pattern)}')
    bx_train_new = apply_activation_pattern(bx_train, activation_pattern)
    bx_val_new = apply_activation_pattern(bx_val, activation_pattern)
    
    print(f'New bx train shape: {bx_train_new.shape}')
    print(f'New bx val shape: {bx_val_new.shape}')

    # Scale the data
    ss_scaler_bx_new = StandardScaler()
    ss_scaler_bx_new.fit(bx_train_new)
    bx_train_ss = ss_scaler_bx_new.transform(bx_train_new)
    bx_val_ss = ss_scaler_bx_new.transform(bx_val_new)
    #save the new scaler
    dump(ss_scaler_bx_new, open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_GA.pkl', 'wb'))


    bx_tensor_train_norm  = torch.tensor(bx_train_ss, dtype=torch.float).to(device)
    ct_tensor_train_norm   = torch.tensor(ct_tot_train_norm, dtype=torch.float).to(device)
    
    bx_tensor_val_norm = torch.tensor(bx_val_ss, dtype=torch.float).to(device)
    ct_tensor_val_norm = torch.tensor(ct_tot_val_norm, dtype=torch.float).to(device)
    
 
    train_dataset = torch.utils.data.TensorDataset(ct_tensor_train_norm, bx_tensor_train_norm)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(ct_tensor_val_norm, bx_tensor_val_norm)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  
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
    print("Model loaded....")
    
    loss_fit = l2_fit
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9), eps=1e-6, weight_decay=l2_reg)
    
    # Train the model
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

    # Validate the model
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
    
    patience = 10
    if EARLY_STOP:    
        early_stopping = EarlyStopping(patience=patience, verbose=True, path='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_GA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '_15sens' + '/model_checkpoint.pth')
        
    # Training and validation loops
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
        # Early stopping call
        if e>=1:
            if EARLY_STOP:
                early_stopping(loss_val_ct_fit, model, e)
                if early_stopping.early_stop:
                    print("---------------------EARLY STOPPING----------------------")
                    break 
                    
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_train_ct_fit': epoch_loss_train_ct_fit,'loss_val_ct_fit': epoch_loss_val_ct_fit}
    torch.save(state, '/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_GA/model_' + str(batch_size) + '_' + str(num_coupling_nodes) + '_' + str(num_subnet_layers +3) + '_' + str(n_epochs) + '_' + str(lr) + '_15sens' + '/model_'  + str(batch_size) + '_'  + str(num_coupling_nodes) + '_' + str(num_subnet_layers+3) + '_' + str(n_epochs) + '_' + str(lr) + '_15sens' + '.pth')
                    
    last_trained_epoch = early_stopping.last_checkpoint_epoch
    last_trained_loss = early_stopping.last_checkpoint_loss/(len(val_loader))
    
    print(f'Last trained Epoch: {last_trained_epoch+1}')
    print(f'Last trained loss: {last_trained_loss}')
#    logger.info(f'Last trained Epoch: {last_trained_epoch+1}')
#    logger.info(f'Last trained Loss: {last_trained_loss}')

    return last_trained_loss


# Initialize the population with individuals having exactly 25 active sensors
def initialize_population(pop_size, num_sensors, num_active_sensors=15):
    population = []
    for _ in range(pop_size):
        individual = np.zeros(num_sensors, dtype=bool)
        active_indices = np.random.choice(num_sensors, num_active_sensors, replace=False)
        individual[active_indices] = True
        population.append(individual)
    return population

# Evaluate the fitness of each individual in the population
def calculate_fitness(population):
    return np.array([evaluate_model_with_sensors(individual) for individual in population])

# Select the best individuals in the current generation as parents for producing the offspring of the next generation
def select_parents(population, fitness, num_parents):
    parents_indices = np.argsort(fitness)[:num_parents]
    return [population[i] for i in parents_indices]

# Crossover (mates) between individuals to produce offspring
def crossover(parents, offspring_size):
    offspring = []
    for k in range(offspring_size):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        crossover_point = np.random.randint(1, len(parents[0])-1)
        offspring.append(np.concatenate((parents[parent1_idx][:crossover_point], parents[parent2_idx][crossover_point:])))
    return np.array(offspring)

# Mutation changes a number of genes as defined by the mutation rate in the offspring
def mutate(offspring, mutation_rate, num_active_sensors=15):
    for idx in range(len(offspring)):
        for _ in range(int(mutation_rate * len(offspring[idx]))):
            active_indices = np.where(offspring[idx])[0]
            inactive_indices = np.where(~offspring[idx])[0]

            if len(active_indices) > 0 and len(inactive_indices) > 0:
                deactivate_idx = np.random.choice(active_indices)
                activate_idx = np.random.choice(inactive_indices)

                offspring[idx][deactivate_idx] = False
                offspring[idx][activate_idx] = True
    return offspring

# Genetic Algorithm settings
pop_size = 5 # Population size
num_sensors = 100  # Total number of sensors
num_generations = 50  # Number of generations
num_parents = 3  # Number of parents for mating
mutation_rate = 0.1  # Mutation rate

def log_details(iteration, config, fitness):
    active_sensors = [i for i, val in enumerate(config) if val]
    dropped_sensors = [i for i, val in enumerate(config) if not val]
    logger.info('-'*100)
    logger.info(f"Iteration {iteration}: \nBest Sensor Config: {config} \nSensors Active: {np.sum(np.array(config, dtype=int))} \nBest Loss: {fitness}")
    logger.info(f"Active Sensors: {active_sensors}")
    logger.info(f"Dropped Sensors: {dropped_sensors}")

# Initialize the population
population = initialize_population(pop_size, num_sensors)

# Start the genetic algorithm
for generation in range(num_generations):
    # Calculate the fitness of each individual
    fitness = calculate_fitness(population)
    
    # Select the best parents
    parents = select_parents(population, fitness, num_parents)
    
    # Generate offspring through crossover
    offspring_crossover = crossover(parents, offspring_size=pop_size - num_parents)
    
    # Add mutation to the offspring
    offspring_mutation = mutate(offspring_crossover, mutation_rate)
    
    # Create the new population based on the parents and offspring
    population[0:num_parents] = parents
    population[num_parents:] = offspring_mutation
    
    # Get the best result in the current iteration
    best_fitness_idx = np.argmin(fitness)
    
    log_details(generation, population[best_fitness_idx], fitness[best_fitness_idx])
    print(f"Generation {generation}, Best Loss: {fitness[best_fitness_idx]}, Best Sensor Config: {population[best_fitness_idx]}")

# The best configuration after all generations
best_fitness = calculate_fitness(population)
best_idx = np.argmin(best_fitness)
print("Best configuration after all generations:", population[best_idx])
print("With loss:", best_fitness[best_idx])