import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data folder path
data_folder_file_path = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/"

def anomaly_detection(run_number):
    # Load physical and model temperature data from .txt
    physical_data = pd.read_csv(data_folder_file_path + f"run {run_number}/filtered_data.csv", delimiter=",").tail(100).to_numpy()
    model_data = pd.read_csv(data_folder_file_path + f"run {run_number}/model_data.csv", delimiter=",").tail(100).to_numpy()
    
    time = physical_data[:, 0].copy().reshape(-1, 1)  # time data
    physical_temps = physical_data[:, 4:].copy()  # physical temperatures
    physical_temps = np.delete(physical_temps, 11, axis=1)  # remove temperature probe 7, which is an input to the model
    model_temps = model_data[:, 1:].copy()  # model temperatures
    
    # Known standard deviation and model error for each probe (example values)
    # temperatures calibrated with +-0.5 C 
    k = np.sqrt(3) # coverage factor for 95% confidence interval
    calibration_error = 0.5 / k  # 0.5 C calibration error
    std_dev = np.array([0.0288, 0.0337, 0.0260, 0.0483, 0.0330, 0.0288, 0.0316, 0.0398])  # std from noise analysis
    model_error = 0.1  # std of assumed gaussian model error
    
    # Calculate thresholds for each probe
    thresholds = np.sqrt(calibration_error**2 + std_dev**2 + model_error**2)  # add in quadricature
    
    # Calculate residuals
    residuals = model_temps - physical_temps
    
    # Residuals exceeding thresholds
    flagged_residuals = np.abs(residuals) > thresholds  # Compare each column with its corresponding threshold
    
    # Check for anomalies over a range
    range_size = 25  # subject to change
    anomaly_threshold = 12  # Number of flagged residuals to consider as an anomaly
    
    # Initialize anomaly detection over ranges for each case
    anomalies_range = np.zeros_like(residuals, dtype=bool)
    
    # Populate anomalies_range
    for j in range(residuals.shape[1]):
        for i in range(0, len(residuals[:, j]), range_size):
            end_index = min(i + range_size, len(residuals[:, j]))
            if np.sum(flagged_residuals[i:end_index, j]) >= anomaly_threshold:
                anomalies_range[i:end_index, j] = True
    
    # Voting logic around each hx
    anomalies_range_hx1 = anomalies_range[:, :6].copy()
    anomalies_range_hx2 = anomalies_range[:, 4:].copy()

    anomalies_hx = np.zeros([len(anomalies_range[:, 0]), 2], dtype=bool)
    
    hx1_min_vote = 4
    hx2_min_vote = 2
    
    for i in range(0, len(anomalies_range_hx1[:, 0])):
        vote = np.sum(anomalies_range_hx1[i, :])
        if vote >= hx1_min_vote:
            anomalies_hx[i, 0] = True

    for i in range(0, len(anomalies_range_hx2[:, 0])):
        vote = np.sum(anomalies_range_hx2[i, :])
        if vote >= hx2_min_vote:
            anomalies_hx[i, 1] = True
    
    # large deviation in single temp probe
    anomalies_large_probe = np.zeros_like(anomalies_range, dtype=bool)
    
    residuals_percent = np.abs(residuals) / physical_temps
    percent_threshold = 0.075  # 7.5% | subject to change
    
    for j in range(anomalies_large_probe.shape[1]):
        for i in range(0, len(anomalies_large_probe[:, j])):
            if np.abs(residuals_percent[i, j]) > percent_threshold:
                anomalies_large_probe[i, j] = True
    
    # System-level anomalies based on either hx flagging or single temp probe w/ large deviation
    anomalies_system = np.zeros([len(anomalies_range[:, 0]), 1], dtype=bool)
    
    for i in range(len(anomalies_range[:, 0])):
        if anomalies_hx[i, 0] or anomalies_hx[i, 1] or np.any(anomalies_large_probe[i, :]):
            anomalies_system[i] = True
    
    # Write anomalies to an output csv\

    combined_anomalies = np.hstack((time,anomalies_range, anomalies_hx, anomalies_system))
    
    data_frame=pd.DataFrame.from_records(combined_anomalies)
    data_frame.to_csv(data_folder_file_path+f"run {run_number}/anomaly.csv",mode="a",header=False,index=False)
    
    return anomalies_range, anomalies_hx, anomalies_system
