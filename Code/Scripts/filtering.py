import numpy as np

data_folder_file_path = "C/users/daq-user/documents/labview data/"

def filter_data(run_number):
    window_size = 10
    num_passes  = 10
    
    #Reads data from file
    data          = np.genfromtxt(data_folder_file_path+"run "+str(run_number)+"/raw data.txt", delimiter=",")
    filtered_data = np.zeros(data.shape)
    
    #Applies an averaging filter to the input data
    avg_kernel                    = np.ones(window_size)/window_size
    for series in range(data.shape[0]):
        for i in range(num_passes):
            filtered_data[series] = convolve1d(data[series], avg_kernel)
    
    #Outputs to file
    np.savetxt(data_folder_file_path+"run "+str(run_number)+"/filtered data.txt", filter_data, delimiter=",")