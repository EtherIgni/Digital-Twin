import numpy             as np
import matplotlib.pyplot as plt
from scipy.signal    import butter,  filtfilt
from nptdms          import TdmsFile
from scipy.ndimage   import convolve1d






#Filtering Functions
def lowpass_Filter(cutoff_frequency, sample_rate, quadratic_order):
    #Creates a generic filter function for arbitrary data
    def filter(data):
        #Applies a butter type low pass filter to the input data
        normal_cutoff = cutoff_frequency/(sample_rate*0.5)
        b, a          = butter(quadratic_order,
                            normal_cutoff,
                            btype='low',
                            analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    return(filter)



def averaging_Filter(window_size, num_passes):
    #Creates a generic filter function for arbitrary data
    def filter(data):
        #Applies an averaging filter to the input data
        avg_kernel    = np.ones(window_size)/window_size
        filtered_data = data
        for i in range(num_passes):
            filtered_data = convolve1d(filtered_data, avg_kernel)
        return(filtered_data)
    return(filter)






def plot_File(data,
              time_zero,
              filter,
              plot_labels,
              plot_colors,
              plot_title,
              plot_y_label,
              plot_save_address,
              set_y_lim):
    #Sets the time to be zeroed to simulation start and sets it to be in mins instead of ms
    time        = (data[-1]-time_zero)/60000



    #Plots Results
    
    for idx in range(data.shape[0]-1):
        plt.plot(time, filter(data[idx]), color=plot_colors[idx], label=plot_labels[idx])
    plt.legend()
    plt.title(plot_title)
    plt.ylabel(plot_y_label)
    plt.xlabel("Time (mins)")
    if(set_y_lim):
        plt.ylim(bottom=0)
    plt.xlim((np.min(time), np.max(time)))
    plt.savefig(plot_save_address, dpi=1000)
    plt.close()
    # plt.show()

    #Demonstrates Time Linearity (Allows checking for collection discontinuities)
    # plt.plot(time)
    # plt.show()






# data_file = TdmsFile.read("Data/FlowMeters.tdms")
# data      = []
# for group in data_file.groups():
#     for idx, channel in enumerate(group.channels()[1:]):
#         data.append(channel[:])
# flow_data      = np.array(data)

# plot_File(flow_data[:,:33985],
#           0,
#           lambda x: x,
#           ["Flow in Loop 1", "Flow in Loop 2", "Flow in Loop 3"],
#           ["red", "green", "blue"],
#           "Mass Flow Rates",
#           "Flow Rate (Gal/min)",
#           "Images/Data Set 1/Mass Flow Rates No Filter.png",
#           True)



# plot_File(flow_data[:,:33985],
#           0,
#           averaging_Filter(200,13),
#           ["Flow in Loop 1", "Flow in Loop 2", "Flow in Loop 3"],
#           ["red", "green", "blue"],
#           "Mass Flow Rates",
#           "Flow Rate (Gal/min)",
#           "Images/Data Set 1/Mass Flow Rates AV Filter.png",
#           True)



# data_file = TdmsFile.read("Data/LoopInput.tdms")
# data      = []
# for group in data_file.groups():
#     for idx, channel in enumerate(group.channels()[1:]):
#         data.append(channel[:])
# input_data      = np.array(data)[0:len(data):2]

# plot_File(input_data[:,:2701],
#           0,
#           lambda x: x,
#           ["Pump 1", "Pump 2", "Valve", "Heater"],
#           ["red", "green", "blue", "orange"],
#           "Voltage Inputs",
#           "Voltage (V)",
#           "Images/Data Set 1/Voltage Inputs No Filter.png",
#           True)



data_file = TdmsFile.read("Data/TempData.tdms")
data      = []
for group in data_file.groups():
    for idx, channel in enumerate(group.channels()[1:]):
        data.append(channel[:35185])
temp_data      = np.array(data)[0:len(data):2]

plot_File(temp_data[[0,1,2,3,-1],1:33776],
          0,
          lambda x: x,
          ["Loop 1-1", "Loop 1-2", "Loop 1-3", "Loop 1-4"],
          ["red", "green", "blue", "orange"],
          "Temperatures for Loop 1",
          "Temperature (C)",
          "Images/Data Set 1/Temperatures Loop 1 No Filter.png",
          False)

plot_File(temp_data[[0,1,2,3,-1],1:33776],
          0,
          averaging_Filter(250,13),
          ["Loop 1-1", "Loop 1-2", "Loop 1-3", "Loop 1-4"],
          ["red", "green", "blue", "orange"],
          "Temperatures for Loop 1",
          "Temperature (C)",
          "Images/Data Set 1/Temperatures Loop 1 AV Filter.png",
          False)

plot_File(temp_data[[4,5,-1],1:33776],
          0,
          averaging_Filter(250,13),
          ["Loop 2-1", "Loop 2-2"],
          ["red", "green"],
          "Temperatures for Loop 2",
          "Temperature (C)",
          "Images/Data Set 1/Temperatures Loop 2 AV Filter.png",
          False)

plot_File(temp_data[[6,7,-1],1:33776],
          0,
          averaging_Filter(250,13),
          ["Loop 3-1", "Loop 3-2"],
          ["blue", "orange"],
          "Temperatures for Loop 3",
          "Temperature (C)",
          "Images/Data Set 1/Temperatures Loop 3 AV Filter.png",
          False)