from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder_file_path       = "Code/Data/Run 2/"
physical_data               = pd.read_csv(data_folder_file_path+"simulated_data.csv", index_col=False, header=None)
physical_data               = np.array(physical_data)[-200:]

for i in range(1,physical_data.shape[1]):
    plt.plot(physical_data[:,i])
plt.show()