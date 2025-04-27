from filtering import filter_Data
from physics_model_changed import simulate_Data
from anomaly_detect import anomaly_detection
import os
#from GUI_plotting import launch_gui
import numpy as np
import global_run_number as g
import subprocess
import psutil


data_file_path_parent = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop"

def is_plot_running(script_name=r"C:\Users\DAQ-User\Documents\Repos\Digital-Twin\Code\Scripts\GUI_plotting.py"):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and script_name in ' '.join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    return False


def master_Control(run_number,Heater, Pump1, Pump2, Wall):
    run_path = data_file_path_parent+"/Run "+str(run_number)

    data_file_path=data_file_path_parent+"/Run "+str(run_number)+"/"
    # unfiltered_data = np.genfromtxt(data_file_path+"raw_data.txt",delimiter=",")
    # if(not(np.min(unfiltered_data[:,0])==0)):
    
    if(os.path.isfile(data_file_path+"filtered_data.csv")):
        # Step 1: Run your data pipeline
        filter_Data(data_file_path)
        simulate_Data(data_file_path,Heater,Pump1,Pump2,Wall)
        anomaly_detection(data_file_path)

        # Step 2: Launch GUI if not already running

        if not is_plot_running():


            subprocess.Popen(["python", r"C:\Users\DAQ-User\Documents\Repos\Digital-Twin\Code\Scripts\GUI_plotting.py",run_path], shell=True)

    else:
        filter_Data(data_file_path)

#is_plot_running(script_name=r"C:\Users\DAQ-User\Documents\Repos\Digital-Twin\Code\Scripts\GUI_plotting.py")
master_Control(1,0, 0, 0, 0)