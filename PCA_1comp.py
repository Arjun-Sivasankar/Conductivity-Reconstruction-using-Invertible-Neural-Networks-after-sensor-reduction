import numpy as np
import h5py
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pickle import dump, load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming bx_train_ss is your sensor data
# Standardize the data
hf = h5py.File('/beegfs/.global1/ws/arsi805e-Test/New/Data_Cond/dist_25mm/sim_bx_ss.h5', 'r')
bx_train_ss = np.array(hf.get('bx_train_ss'))
bx_val_ss = np.array(hf.get('bx_val_ss'))
print(f"bx_train shape: {bx_train_ss.shape}")
print(f"bx_val shape: {bx_val_ss.shape}")

# Apply PCA
pca = PCA(n_components=100)
pca.fit(bx_train_ss)

# Get the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance: ", explained_variance)

# Analyze the components to see which sensors are most important
# Each component in pca.components_ corresponds to one of the original features
components = pca.components_
print(f"Components are: {components}")

# You can then examine the components to understand sensor importance
# For example, looking at the first principal component
first_component = components[0]
sensor_importance = abs(first_component)                           

# Sort the sensor indices by their importance in the first component
important_sensors = np.argsort(sensor_importance)[::-1]
print(f"Importance Sensors are: {important_sensors}")

# Calculate the cumulative sum of explained variances
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components that explain at least 95% of the variance
num_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1  # Adding 1 because index starts at 0

# Plotting
plt.figure(figsize=(10,6))
plt.plot(cumulative_explained_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=num_components_95, color='r', linestyle='--')
plt.text(num_components_95, 0.85, f'95% variance\n{num_components_95} components', color = "red")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.savefig('/beegfs/.global1/ws/arsi805e-Test/New/Reduced_Sensors_Models_PCA/PCA_results.png')
plt.show()