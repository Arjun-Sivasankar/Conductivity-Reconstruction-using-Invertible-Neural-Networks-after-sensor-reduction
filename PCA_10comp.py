import numpy as np
import h5py
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pickle import dump, load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Assuming bx_train_ss is your sensor data
# Standardize the data
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_bx_ss.h5', 'r')
bx_train_ss = np.array(hf.get('bx_train_ss'))
bx_val_ss = np.array(hf.get('bx_val_ss'))
print(f"bx_train shape: {bx_train_ss.shape}")
print(f"bx_val shape: {bx_val_ss.shape}")

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(bx_train_ss)

# Extract the loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=[f'Sensor{i+1}' for i in range(bx_train_ss.shape[1])])

loadings.to_excel('PCA.xlsx')

# Analyze loadings for the first few PCs to determine important sensors
important_pcs = loadings[['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8','PC9','PC10']]  # Adjust based on how many PCs you consider important
print(important_pcs.abs().sort_values(by='PC1', ascending=False))  # Sort by absolute value of loadings in PC1

## Optionally, you can visualize the loadings to better understand the importance
#import seaborn as sns
#import matplotlib.pyplot as plt
#
#plt.figure(figsize=(10, 8))
#sns.heatmap(important_pcs.abs(), cmap='YlGnBu')  # Using absolute value for visualization
#plt.title('Absolute Loadings of Sensors on Principal Components')
#plt.savefig('PCA_comp.png')
#plt.show()
absolute_loadings = loadings[['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8','PC9','PC10']].abs()

# Step 2: Aggregate loadings for each sensor
# Here we use the sum of absolute loadings, but you could also use mean, max, etc.
aggregate_loadings = absolute_loadings.sum(axis=1)

# Step 3: Rank sensors from highest to lowest based on aggregate loadings
ranked_sensors = aggregate_loadings.sort_values(ascending=False)

print("\nSensors ranked from highest to lowest importance based on aggregate loadings:")
print(ranked_sensors)

# Print just the sensor numbers in ranked order
ranked_sensors = aggregate_loadings.sort_values(ascending=False).index.tolist()
ranked_sensor_numbers = [sensor.replace('Sensor', '') for sensor in ranked_sensors]
ranked_sensor_numbers = [int(i) for i in ranked_sensor_numbers]
print("\nRanked order of Sensors (PCA): ", ranked_sensor_numbers)