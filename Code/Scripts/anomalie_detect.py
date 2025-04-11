import numpy as np
import matplotlib.pyplot as plt

#data_folder_file_path = "C/users/daq-user/documents/labview data/"
data_folder_file_path = "C:/Users/james/.spyder-py3/assignments/NE_471/"

def anomaly_detection(run_number):
    # load physical and model temperature data from .txt
    # will include time as first column (12 in total)
    physical_data = np.genfromtxt(data_folder_file_path+"run "+str(run_number)+"/filter_data.txt", delimiter=",")[-100:]
    
    # for model, just the 8 temperature data
    model_data = np.genfromtxt(data_folder_file_path+"run "+str(run_number)+"/model_data.txt", delimiter=",")[-100:]
    
    physical_temps = physical_data[:-1, 3:]  # physical temperatures
    model_temps = model_data[:-1, 3:]  # model temperatures
    
    #threshold determination
    threshold = 0.05 # semi placeholder. from uncertainty analysis of temp data
    
    # calculate residuals
    residuals = model_temps - physical_temps
    
    # residuals exceeding threshold
    flagged_residuals = np.abs(residuals) > threshold
    
    # Check for anomalies over a range
    # maybe change for the full 10 second range?
    range_size = 100 #change to what gets to like 5 seconds
    anomaly_threshold =  40 #Number of flagged residuals to consider as an anomaly. based on binomial distribution
    
    # Initialize anomaly detection over ranges for each case
    anomalies_range = np.zeros_like(residuals, dtype=bool)
    
    #populate anomalies_range
    for j in range(residuals.shape[1]):
        for i in range(0, len(residuals[:,j]), range_size):
            end_index = min(i + range_size, len(residuals[:,j]))
            if np.sum(flagged_residuals[i:end_index,j]) >= anomaly_threshold:
                anomalies_range[i:end_index,j] = True
    
    #voting logic around each hx 
    anomalies_range_hx1 = anomalies_range[:,:6].copy()
    anomalies_range_hx2 = anomalies_range[:,4:].copy()

    anomalies_hx = np.zeros([len(anomalies_range[:,0]),2], dtype=bool)
    
    hx1_min_vote = 4
    hx2_min_vote = 3
    

    for i in range(1, len(anomalies_range_hx1[:,0])):
        vote = np.sum(anomalies_range_hx1[i,:])
        if vote >= hx1_min_vote:
            anomalies_hx[i,0] = True

    for i in range(1, len(anomalies_range_hx2[:,0])):
        vote = np.sum(anomalies_range_hx2[i,:])
        if vote >= hx2_min_vote:
            anomalies_hx[i,1] = True
    
    #system level anomalies based on either hx flagging
    anomalies_system = np.zeros([len(anomalies_range[:,0]),1], dtype=bool)
    
    for i in range(len(anomalies_range[:,0])):
        if anomalies_hx[i,0] or anomalies_hx[i,1]:
            anomalies_system[i] = True
    # Write anomalies to an output text file
    output_file_path = data_folder_file_path+"run "+str(run_number)+"/anomalies.txt"
    
    # Open the file for writing
    with open(output_file_path, "w") as file:
        # Write the data row by row
        for i in range(len(anomalies_range)):
            # Convert each column of anomalies_range to a string
            anomalies_range_row = ",".join(map(str, anomalies_range[i].astype(int)))
            # Convert each column of anomalies_hx to a string
            anomalies_hx_row = ",".join(map(str, anomalies_hx[i].astype(int)))
            # Convert anomalies_system to a string
            anomalies_system_row = str(int(anomalies_system[i, 0]))
            # Combine all columns into a single row
            file.write(f"{anomalies_range_row},{anomalies_hx_row},{anomalies_system_row}\n")
    return anomalies_range, anomalies_hx, anomalies_system
    

# # Plot residuals for each case
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# cases = [
#     (residuals_case1, anomalies_case1, anomalies_range_case1, "Case 1 (T_hot_in replaced)"),
#     (residuals_case2, anomalies_case2, anomalies_range_case2, "Case 2 (T_hot_out replaced)"),
#     (residuals_case3, anomalies_case3, anomalies_range_case3, "Case 3 (T_cold_in replaced)"),
#     (residuals_case4, anomalies_case4, anomalies_range_case4, "Case 4 (T_cold_out replaced)"),
# ]

# for ax, (residuals, anomalies, anomalies_range, title) in zip(axs.flat, cases):
#     ax.plot(residuals, label="Residuals")
#     ax.scatter(np.where(anomalies)[0], residuals[anomalies], color="red", label="Exceeded threshold")
#     for i in range(0, len(anomalies_range), range_size):
#         end_index = min(i + range_size, len(anomalies_range))
#         if np.any(anomalies_range[i:end_index]):
#             x = [i, end_index, end_index, i]
#             y = [min(residuals), min(residuals), max(residuals), max(residuals)]
#             ax.fill(x, y, color="red", alpha=0.2)
#     ax.set_title(title)
#     ax.set_xlabel("Time Step")
#     ax.set_ylabel("Residual")
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()