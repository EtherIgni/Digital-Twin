import numpy as np
from scipy.ndimage import convolve1d
import pandas as pd

data_folder_file_path = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/"

def filter_data(run_number):
    unfiltered_data = np.genfromtxt(data_folder_file_path+"Run 1/unfiltered_data_ready.txt")
    
    data_frame=pd.DataFrame.from_records(unfiltered_data)
    data_frame.to_csv(data_folder_file_path+"Run 1/filtered_data.csv",mode="a",header=False,index=False)

filter_data(1)