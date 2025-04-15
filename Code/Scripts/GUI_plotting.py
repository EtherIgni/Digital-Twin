import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os
import sys

# File paths
data_folder_file_path     =  r"C:\Users\DAQ-User\Documents\LabVIEW Data\3Loop\Run 1"

physical_data_path = data_folder_file_path + r"\filtered_data.csv"
model_data_path = data_folder_file_path + r"\simulated_data.csv"
anomaly_data_path = data_folder_file_path + r"\anomaly.csv"

class AnomalyPlotterApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Plotter")
        
        # Create a frame for the plot
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize the last file size to detect changes
        self.last_file_size = 0
        
        # Start the update loop
        self.update_id = None  # Store the after ID
        self.update_plot()
        
        # Bind the close event to clean up
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_plot(self):
        
        # Read the updated CSV file
        try:
            # Load data from CSV files
            physical_data = pd.read_csv(physical_data_path, header=None).to_numpy()
            model_data = pd.read_csv(model_data_path, header=None).to_numpy()
            anomaly_data = pd.read_csv(anomaly_data_path, header=None).to_numpy()
            
            time_data_physical = physical_data[:, 0]  # Assuming time is in the first column
            time_data_model = physical_data[:, 0]  # Assuming time is in the first column
            
            # Clear the previous plot
            self.ax.clear()
            
            # Plot physical temperature data
            for col in range(3, physical_data.shape[1]):  # Loop through each temperature column
                self.ax.plot(time_data_physical, physical_data[:, col], label=f"Physical Temp {col-2}", color="blue")

            # Plot model temperature data
            for col in range(3, model_data.shape[1]):  # Loop through each temperature column
                self.ax.plot(time_data_model, model_data[:, col], label=f"Model Temp {col-2}", color="green")
            
            # Highlight system-level anomalies
            anom_detect = 0  # Initialize anomaly detection state
            for i in range(len(anomaly_data[:, -1])):
                if anomaly_data[i, -1] == 1:  # System-level anomaly detected
                    if anom_detect == 0:  # Start of a new anomaly range
                        anom_detect = 1
                        start_index = i
                elif anomaly_data[i, -1] == 0 and anom_detect == 1:  # End of an anomaly range
                    anom_detect = 0
                    end_index = i - 1  # The last index of the anomaly range
                    # Shade the anomaly range
                    self.ax.axvspan(anomaly_data[start_index, 0], anomaly_data[end_index, 0], color="red", alpha=0.2)

            # Handle the case where the last row is part of an anomaly
            if anom_detect == 1:
                end_index = len(anomaly_data[:, -1]) - 1
                self.ax.axvspan(anomaly_data[start_index, 0], anomaly_data[end_index, 0], color="red", alpha=0.2)
                                                    
            # Set plot titles and labels with increased font size
            self.ax.set_title("Anomaly Data", fontsize=20)  # Increase title font size
            self.ax.set_xlabel("Time", fontsize=20)  # Increase x-axis label font size
            self.ax.set_ylabel("Temperature", fontsize=20)  # Increase y-axis label font size

            # Increase font size of tick labels
            self.ax.tick_params(axis='both', which='major', labelsize=16)  # Major ticks
            self.ax.tick_params(axis='both', which='minor', labelsize=14)  # Minor ticks

            # Increase font size of the legend
            self.ax.legend(fontsize=12)  # Increase legend font size
            
            # Refresh the canvas
            self.canvas.draw()
        except Exception as e:
            print(f"Error reading or plotting data: {e}")
    
        # Schedule the next update
        self.update_id = self.root.after(1000, self.update_plot)  # Update every 1 second

    def on_close(self):
        # Cancel the scheduled callback
        if self.update_id is not None:
            self.root.after_cancel(self.update_id)
        
        # Destroy the root window
        self.root.destroy()
        
        # Exit the program
        sys.exit()

def launch_gui():
    import tkinter as tk
    root = tk.Tk()
    app = AnomalyPlotterApp(root)
    root.mainloop()
if __name__ == "__main__":
    launch_gui()
