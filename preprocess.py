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

## Setting up the data
num_tot_samples = 10000
train_split = 8000
val_split = 2000

bx_tot = []
bx_tot_train = []
bx_tot_val = []

ct_tot = []
ct_tot_train = []
ct_tot_val = []
ct_coord = []

#####################################################################################
#First things first, extract j coordinates
#####################################################################################
directory = "/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/nx10_ny10_nz0_mesh5/1/"   
# load .npz file
data = np.load(directory + 'm' + str(1) + '.npz')
# relevant j-data
ct_data = data['arr_1']   
ct_coordinates = ct_data[:, 0:2]   # x-, y-coordinates of the output [m] -> z = 0.005 m
ct_coord.append(ct_coordinates.squeeze())
ct_coord = np.asarray(ct_coord)

hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/ct_coord.h5', 'w')
hf.create_dataset('ct_coord', data = ct_coord)
hf.close()


#####################################################################################
# Extract the data (select the highest current strength right now)
#####################################################################################
for j in range(1,11):
    for i in range(1,1001):
        directory = "/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/nx10_ny10_nz0_mesh5"   
        # load .npz files
        data = np.load(directory + '/' + str(j) + '/' + 'm' + str(i) + '.npz')    
        # relevant b-data
        b_data = data['arr_0']
        b_coordinates = b_data[:, 0:2]   # x-, y-coordinates of the input [m] -> z = 0 m
        bx = b_data[:, 2]                # input
        # relevant j-data
        ct_data = data['arr_1']
        ct_coordinates = ct_data[:, 0:2]   # x-, y-coordinates of the output [m] -> z = 0.005 m
        ct_rel = ct_data[:, 2]             # output / estimation
        sigma_bn = data['arr_3'].reshape([ct_data.shape[0], 1])   # output / estimation
        bx_tot.append(bx.squeeze())
        ct_tot.append(sigma_bn.squeeze())
        if i % 500 == 0:
            print(i)
    
bx_tot = np.asarray(bx_tot)
ct_tot  = np.asarray(ct_tot)

print(bx_tot.shape)
print(ct_tot.shape)
print(ct_coord.shape)

    
#####################################################################################
# Shuffle the data
#####################################################################################
idx_shuffle1 = load(open('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_5mm/idx_shuffle1.pkl', 'rb'))
bx_tot,ct_tot = bx_tot[idx_shuffle1], ct_tot[idx_shuffle1]

idx_shuffle2 = load(open('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_5mm/idx_shuffle2.pkl', 'rb'))
bx_tot,ct_tot = bx_tot[idx_shuffle2], ct_tot[idx_shuffle2]

idx_shuffle3 = load(open('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_5mm/idx_shuffle3.pkl', 'rb'))
bx_tot,ct_tot = bx_tot[idx_shuffle3], ct_tot[idx_shuffle3]


hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_permute_data.h5', 'w')
hf.create_dataset('bx_tot', data = bx_tot)
hf.create_dataset('ct_tot', data = ct_tot)
hf.create_dataset('ct_coord', data = ct_coord)
hf.close()


#########################################################################################
# Training validation split
#########################################################################################
ct_train = ct_tot[:train_split,:]
ct_val = ct_tot[train_split:train_split+val_split,:]

bx_train = bx_tot[:train_split,:]
bx_val   = bx_tot[train_split:train_split+val_split,:]


hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_permute_train_val_split.h5', 'w')
hf.create_dataset('bx_train', data = bx_train)
hf.create_dataset('bx_val', data = bx_val)
hf.create_dataset('ct_train', data = ct_train)
hf.create_dataset('ct_val', data = ct_val)
hf.close()


#################################################################################################
# Use standard scaler from sklearn to transform the magnetic field data
#################################################################################################
ss_scaler_bx   = StandardScaler()

# fit the standard scaler on the total set
ss_scaler_bx.fit(bx_train)
bx_train_ss = ss_scaler_bx.transform(bx_train)
bx_val_ss = ss_scaler_bx.transform(bx_val)

#save the scaler
dump(ss_scaler_bx, open('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/ss_scaler_bx.pkl', 'wb'))

#save the scaled data
hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_bx_ss.h5', 'w')
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
dump(ss_scaler_ct, open('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/ss_scaler_ct.pkl', 'wb'))


#save the scaled data
hf = h5py.File('/home/h1/s8993054/Inverse_Current_Tomography/Data/Conductivity/dist_25mm/sim_ct_ss.h5', 'w')
hf.create_dataset('ct_train_ss', data = ct_train_ss)
hf.create_dataset('ct_val_ss', data = ct_val_ss)
hf.close()


#################################################################################################