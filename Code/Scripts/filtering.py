import numpy as np
from scipy.ndimage import convolve1d
import pandas as pd

data_folder_file_path = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/"

window_size = 10
num_passes  = 10

def filter_data(run_number):
    unfiltered_data = np.genfromtxt(data_folder_file_path+"Run 1/unfiltered_data_ready.txt")
    
    filtered_data = np.zeros(unfiltered_data.shape)
    filtered_data[:,0:5]=unfiltered_data[:,0:5]
    
    avg_kernel          = np.ones(window_size)/window_size
    offset          = 5
    for series in range(8):
        for i in range(num_passes):
            filtered_data[:,series+offset]  = convolve1d(unfiltered_data[:,series+offset],  avg_kernel)
    
    
    data_frame=pd.DataFrame.from_records(filtered_data)
    data_frame.to_csv(data_folder_file_path+"Run 1/filtered_data.csv",mode="a",header=False,index=False)

filter_data(1)