######################################################################################
### import libraries
######################################################################################
import numpy as np
import h5py
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pickle import dump, load
from sklearn.decomposition import PCA
import argparse
import random
import logging

## Setting up the data
num_tot_samples = 10000
train_split = 8000
val_split = 2000

hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_permute_data.h5', 'r')
bx_tot = np.array(hf.get('bx_tot'))
ct_tot = np.array(hf.get('ct_tot'))
ct_coord = np.array(hf.get('ct_coord'))
hf.close()
#########################################################################################
# Training validation split
#########################################################################################
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_permute_train_val_split.h5', 'r')
bx_train = np.array(hf.get('bx_train'))
bx_val = np.array(hf.get('bx_val'))
ct_train = np.array(hf.get('ct_train'))
ct_val = np.array(hf.get('ct_val'))
hf.close()

########################################################################################
# Sensor Reduction
########################################################################################
parser = argparse.ArgumentParser(description='Sensor Reduction strategy - Naive or PCA')
parser.add_argument('--strat', default='naive', metavar='n', help='Sensor Reduction Strat. (default:naive)')
parser.add_argument('--drop', type=int, default=5, metavar='n', help='sensors to be dropped (default:5)')
args = parser.parse_args()
n_sensors = 100 - (args.drop)
if args.strat == 'naive':
    naive = True
else:
    naive = False
    
if args.strat == 'PCA':
    PCA = True
else:
    PCA = False
    
if args.strat == 'random':
    rand = True
else:
    rand = False


if naive:
    grid_size = 10
    activation_pattern = np.zeros((grid_size, grid_size), dtype=int)
#    activation_pattern = np.ones((grid_size, grid_size), dtype=int)  # Start with all sensors active
    
    ## Define the rows and columns for the 6x6 middle grid
    start_row, end_row = 1, 9  # 3rd to 8th row (0-indexed)
    start_col, end_col = 2, 8  # 3rd to 8th column (0-indexed)
    #
    ## Activate the middle 6x6 grid
    activation_pattern[start_row:end_row, start_col:end_col] = 1
    
    # Deactivate the 
#    activation_pattern[:, [0,1,2,-3,-2,-1]] = 0
    
    # Flatten the activation pattern to match the sensor data shape
    activation_pattern_flat = activation_pattern.flatten()
    
    # Function to apply the activation pattern to the sensor data
    def apply_activation_pattern(data, activation_pattern):
        return data[:, activation_pattern == 1]
        
    # Apply the activation pattern
    bx_train_new = apply_activation_pattern(bx_train, activation_pattern_flat)
    bx_val_new = apply_activation_pattern(bx_val, activation_pattern_flat)
    
    print("Shape of activated bx_train data:", bx_train_new.shape)
    print("Shape of activated bx_val data:", bx_val_new.shape)
    
elif PCA:    
    # List of indices for sensors to be activated
#    active_sensors_indices = [5, 95, 4, 94, 15, 85, 14, 84, 99, 9, 0, 89, 90, 19, 10, 25, 80, 75, 24, 74, 79, 29, 6, 20, 96, 70, 93, 16,
#                               86, 3, 83, 13, 26, 76, 35, 69, 39, 73, 1, 65, 34, 23, 30, 64, 91, 60, 11, 8, 98, 81, 18, 88, 36, 66, 21, 63,
#                               71, 33, 28, 78, 7, 97, 17, 87, 31, 92, 27, 38, 77, 61, 2, 68, 82, 59, 12, 40, 45, 49, 44, 72, 55, 54, 50, 22,
#                               37, 46, 67, 56, 53, 62, 43, 32, 41, 48, 58, 47, 51, 52, 57, 42]
     # For clustering                          
#    active_sensors_indices = [24, 77, 20, 27, 78, 21, 74, 29, 70, 22, 72, 71, 79, 28, 25,
#                               25, 80, 75, 24, 74, 79, 29, 6, 20, 96, 70, 93, 16,
#                               86, 3, 83, 13, 26, 76, 35, 69, 39, 73, 1, 65, 34, 23, 30, 64, 91, 60, 11, 8, 98, 81, 18, 88, 36, 66, 21, 63,
#                               71, 33, 28, 78, 7, 97, 17, 87, 31, 92, 27, 38, 77, 61, 2, 68, 82, 59, 12, 40, 45, 49, 44, 72, 55, 54, 50, 22,
#                               37, 46, 67, 56, 53, 62, 43, 32, 41, 48, 58, 47, 51, 52, 57, 42]
                               
#    active_sensors_indices = [41, 31, 50, 61, 60, 70, 51, 40, 79, 29, 72, 69, 62, 39, 71, 89, 32, 21, 63, 22, 19, 73, 33, 82, 38, 99, 68, 9, 80, 23, 12, 30, 45, 81, 92, 28, 48, 35, 78, 83, 42, 11, 67, 43, 47, 58, 49, 52, 65, 37, 53, 34, 59, 55, 2, 46, 44, 57, 64, 13, 91, 36, 56, 66, 54, 90, 1, 93, 20, 18, 88, 77, 3, 74, 24, 25, 75, 27, 100, 10, 8, 98, 76, 26, 84, 87, 14, 17, 85, 15, 94, 97, 4, 86, 7, 16, 95, 5, 96, 6]
    
    active_sensors_indices = [1,8,18,22,32,38,64,66,72,75,85,87]
                               
    active_sensors_indices = [i-1 for i in active_sensors_indices]
                     
#    active_sensors = active_sensors_indices[:n_sensors]
    active_sensors = active_sensors_indices
    print(f"No. of active sensors: {len(active_sensors)}")
    print(f"Activated Sensors are: {active_sensors}")
    
    # Create an activation pattern based on the given indices
    activation_pattern = np.zeros(100, dtype=int)  # Assuming there are 100 sensors
    activation_pattern[active_sensors] = 1
    print(activation_pattern)
    
    # Function to apply the activation pattern to the sensor data
    def apply_activation_pattern(data, activation_pattern):
        return data[:, activation_pattern == 1]
        
    # Apply the activation pattern
    bx_train_new = apply_activation_pattern(bx_train, activation_pattern)
    bx_val_new = apply_activation_pattern(bx_val, activation_pattern)
    
    print("Shape of activated bx_train data:", bx_train_new.shape)
    print("Shape of activated bx_val data:", bx_val_new.shape)
    
elif rand:
    NUM_SENSORS_DROPPED = args.drop
    
    sensors_dropped = random.sample(range(0,100), NUM_SENSORS_DROPPED)
    print(f"Dropping sensors: {sensors_dropped}")
    n_sensors = 100 - NUM_SENSORS_DROPPED
    print(f"No. of sensors used in this iteration: {n_sensors}")
    
    # Create an activation pattern based on the given indices
    activation_pattern = np.ones(100, dtype=int)  # Assuming there are 100 sensors
    activation_pattern[sensors_dropped] = 0
    
    # Function to apply the activation pattern to the sensor data
    def apply_activation_pattern(data, activation_pattern):
        return data[:, activation_pattern == 1]
        
    # Apply the activation pattern
    bx_train_new = apply_activation_pattern(bx_train, activation_pattern)
    bx_val_new = apply_activation_pattern(bx_val, activation_pattern)
        
    print("Shape of activated bx_train data:", bx_train_new.shape)
    print("Shape of activated bx_val data:", bx_val_new.shape)
    
    
    logging.basicConfig(filename='/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_Random/' + 'RandomSamplingLog.log', level=logging.INFO)    
    logging.info(f'Sensors used: {n_sensors}')
    logging.info(f'Dropped Sensors: {sensors_dropped}')
#################################################################################################
# Use standard scaler from sklearn to transform the magnetic field data
#################################################################################################
ss_scaler_bx_new   = StandardScaler()

# fit the standard scaler on the total set
ss_scaler_bx_new.fit(bx_train_new)
bx_train_ss = ss_scaler_bx_new.transform(bx_train_new)
bx_val_ss = ss_scaler_bx_new.transform(bx_val_new)

#save the scaler
if naive:
    dump(ss_scaler_bx_new, open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_8x6.pkl', 'wb'))
elif PCA:
    dump(ss_scaler_bx_new, open(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_PCA_{n_sensors}sens_SA.pkl', 'wb'))
elif rand:
    dump(ss_scaler_bx_new, open(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_bx_random_{n_sensors}.pkl', 'wb'))

#save the scaled data
if naive:
    hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_8x6.h5', 'w')
elif PCA:
    hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_PCA_{n_sensors}sens_SA.h5', 'w')
elif rand:
    hf = h5py.File(f'/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_bx_ss_random_{n_sensors}.h5', 'w')
hf.create_dataset('bx_train_ss', data = bx_train_ss)
hf.create_dataset('bx_val_ss', data = bx_val_ss)
hf.close()

#################################################################################################
# Use standard scaler from sklearn to transform the conductivity data
#################################################################################################
ss_scaler_ct   = StandardScaler()

# fit the standard scaler on the total set
ss_scaler_ct.fit(ct_train)
ct_train_ss = ss_scaler_ct.transform(ct_train)
ct_val_ss = ss_scaler_ct.transform(ct_val)


#save the scaler
dump(ss_scaler_ct, open('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/ss_scaler_ct.pkl', 'wb'))


#save the scaled data
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sensor_reduction_scaled/sim_ct_ss.h5', 'w')
hf.create_dataset('ct_train_ss', data = ct_train_ss)
hf.create_dataset('ct_val_ss', data = ct_val_ss)
hf.close()


#################################################################################################