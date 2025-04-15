import numpy as np
from scipy.ndimage import convolve1d
import pandas as pd

data_folder_file_path = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/"

window_size = 10
num_passes  = 10

def filter_data(run_number):
    unfiltered_data                     = np.genfromtxt(data_folder_file_path+"Run "+str(run_number)+"/raw_data.txt")
    if(unfiltered_data.size>100):
        unfiltered_data=unfiltered_data[-110:]
    else:
        unfiltered_data=unfiltered_data[-100:]
    
    filtered_data                       = np.zeros(unfiltered_data.shape)
    avg_kernel                          = np.ones(window_size)/window_size
    filtered_data[:,0]                  = unfiltered_data[:,0]
    for series in range(1,13):
        if(series in [4]):
            filtered_data[:,series]     = np.ones(unfiltered_data.shape[0])*unfiltered_data[:,series]
        else:
            for i in range(num_passes):
                filtered_data[:,series] = convolve1d(unfiltered_data[:,series],  avg_kernel)
    
    
    data_frame                          = pd.DataFrame.from_records(filtered_data[-100:])
    data_frame.to_csv(data_folder_file_path+"Run "+str(run_number)+"/filtered_data.csv",mode="a",header=False,index=False)