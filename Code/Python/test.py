from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



fig,ax=plt.subplots(2)

data_folder_file_path       = "Code/Data/Run 3/"
physical_data               = pd.read_csv(data_folder_file_path+"filtered_data.csv", index_col=False, header=None)
physical_data               = np.array(physical_data)
physical_data               = physical_data[:,[0,5,6,7,8,9,10,12]]
for i in range(1,physical_data.shape[1]):
    ax[0].plot(physical_data[:,0],physical_data[:,i],label=str(i))
ax[0].legend()
    
data_folder_file_path       = "Code/Data/Run 3/"
physical_data               = pd.read_csv(data_folder_file_path+"simulated_data.csv", index_col=False, header=None)
physical_data               = np.array(physical_data)
for i in range(1,physical_data.shape[1]):
    ax[1].plot(physical_data[:,0],physical_data[:,i],label=str(i))
ax[1].legend()

plt.show()