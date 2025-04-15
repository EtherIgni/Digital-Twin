from filtering import filter_Data
from physics_model import simulate_Data
from anomaly_detect import anomaly_detection

import subprocess
import psutil


def is_plot_running(script_name=r"C:\Users\DAQ-User\Documents\Repos\Digital-Twin\Code\Scripts\GUI_plotting.py"):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and script_name in ' '.join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    return False


def master_Control(run_num,Heater, Pump1, Pump2, Wall):
    # Step 1: Run your data pipeline
    filter_Data(run_num)
    simulate_Data(run_num)
    anomaly_detection(run_num)

    # Step 2: Launch GUI if not already running
    if not is_plot_running():
        subprocess.Popen(["python", r"C:\Users\DAQ-User\Documents\Repos\Digital-Twin\Code\Scripts\GUI_plotting.py"], shell=True)
