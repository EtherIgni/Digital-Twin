import numpy as np
from scipy.ndimage import convolve1d
import pandas as pd

data_folder_file_path = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/"

window_size = 10
num_passes  = 10

def filter_Data(run_number):
    unfiltered_data                     = np.genfromtxt(data_folder_file_path+"Run "+str(run_number)+"/unfiltered_data_ready.txt")
    
    filtered_data                       = np.zeros(unfiltered_data.shape)
    avg_kernel                          = np.ones(window_size)/window_size
    for series in range(1,13):
        if(series in [4]):
            filter_data[:,series]       = np.ones(unfiltered_data.shape[0])*unfiltered_data[:,series]
        else:
            for i in range(num_passes):
                filtered_data[:,series] = convolve1d(unfiltered_data[:,series],  avg_kernel)
    
    
    data_frame                          = pd.DataFrame.from_records(filtered_data)
    data_frame.to_csv(data_folder_file_path+"Run "+str(run_number)+"/filtered_data.csv",mode="a",header=False,index=False)

filter_data(1)