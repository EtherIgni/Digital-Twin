import numpy as np
from scipy.ndimage import convolve1d
import pandas as pd

def filter_Data(data_folder_file_path):
    
    unfiltered_data                     = np.genfromtxt(data_folder_file_path+"/raw_data.txt",delimiter=",")
        
    filtered_data                       = np.zeros(unfiltered_data.shape)

    filtered_data[:,0]                  = unfiltered_data[:,0]        
    
    avg_kernel                          = np.ones(100)/100
    for series in range(1,4):
        for i in range(10):
                filtered_data[:,series] = convolve1d(unfiltered_data[:,series],  avg_kernel)

    with open(data_folder_file_path + "control.txt",'r') as control:
            L = control.readline()
    if L =="":            
        filtered_data[-10:,4]     = np.ones(10)*np.max(unfiltered_data[-10:,4]) #this needs to be fixed. now only the last 10 is ever saved
    else: 
        filtered_data[:,4]     = np.ones(unfiltered_data.shape[0])*np.max(unfiltered_data[:int(L),4])
    
    avg_kernel                          = np.ones(10)/10
    for series in range(5,13):
        for i in range(5):
                filtered_data[:,series] = convolve1d(unfiltered_data[:,series],  avg_kernel)
            
    
    
    data_frame                          = pd.DataFrame.from_records(filtered_data)
    data_frame.to_csv(data_folder_file_path+"filtered_data.csv",header=False,index=False)
    
