import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import traceback
import numpy as np
import global_run_number as g
from matplotlib.patches import Patch

# Set Matplotlib backend
plt.switch_backend("Agg")  # Avoid some GUI conflicts
import sys
import os

# Default fallback (optional, or set to None if you want to require an argument)

print("[GUI_plotting.py] sys.argv =", sys.argv)
data_folder=str(sys.argv[1])

# File paths
#data_folder = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/Run 1"


#data_folder= data_folder + str({run_number})
physical_data_path = os.path.join(data_folder, "filtered_data.csv")
model_data_path    = os.path.join(data_folder, "simulated_data.csv")
anomaly_data_path  = os.path.join(data_folder, "anomaly.csv")


class AnomalyPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Plotter")

        # UI setup
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_id = None
        self.update_plot()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_plot(self):
        try:
            # Read the latest data (wrapped with file check)
            if not all(map(os.path.exists, [physical_data_path, model_data_path, anomaly_data_path])):

                raise FileNotFoundError("One or more data files are missing.")

            # Subtract the first value in the first column [0, 0] for each file
            physical_data = pd.read_csv(physical_data_path, header=None).to_numpy()


            model_data = pd.read_csv(model_data_path, header=None).to_numpy()[:, :8]

            physical_data[:, 0] -= model_data[0, 0]
            physical_data = physical_data[20:,:]
            model_data[:, 0] -= model_data[0, 0]
            anomaly_data = pd.read_csv(anomaly_data_path, header=None).to_numpy()
            anomaly_data[:, 0] -= anomaly_data[0, 0]


            self.ax.clear()

            # Plot physical temperature
            colors = plt.cm.viridis(np.linspace(0, 1, 7))
            for idx, col in enumerate(range(5, physical_data.shape[1]-1)):
                if col == 11:
                    self.ax.plot(physical_data[:,0], physical_data[:, col+1], label=f"Physical Temp {col-3}", color=colors[idx])
                else:
                    self.ax.plot(physical_data[:,0], physical_data[:, col], label=f"Physical Temp {col-4}", color=colors[idx])



            for idx, col in enumerate(range(1, model_data.shape[1])):
                if col == 7:
                    self.ax.plot(model_data[:,0], model_data[:, col],'--', label=f"Model Temp {col+1}", color=colors[idx])

                else:
                    self.ax.plot(model_data[:,0], model_data[:, col],'--', label=f"Model Temp {col}", color=colors[idx])
            # Plot anomaly regions
            is_anomaly = False
            for i in range(len(anomaly_data)):
                if anomaly_data[i, -1] == 1 and not is_anomaly:
                    is_anomaly = True
                    start_idx = i
                elif anomaly_data[i, -1] == 0 and is_anomaly:
                    is_anomaly = False
                    end_idx = i - 1
                    self.ax.axvspan(anomaly_data[start_idx, 0], anomaly_data[end_idx, 0], color='red', alpha=0.3)
            if is_anomaly:
                self.ax.axvspan(anomaly_data[start_idx, 0], anomaly_data[-1, 0], color='red', alpha=0.3, label= "Anomaly")

            # Labels and legend
            self.ax.set_title("Anomaly Detection", fontsize=20)
            self.ax.set_xlabel("Time (s)", fontsize=16)
            self.ax.set_ylabel("Temperature (C)", fontsize=16)
            self.ax.tick_params(axis='both', labelsize=14)
            from matplotlib.lines import Line2D
            solid_line= Line2D([0],[0], color= 'black', linestyle='-', label='Measured Data')
            dotted_line= Line2D([0],[0], color= 'black', linestyle='--', label='Physics Model')
            highlight_patch = Patch(color='red',alpha=.3, label='Anomaly')

# Add everything to the legend
            self.ax.legend(handles=[solid_line, dotted_line, highlight_patch], fontsize=12)

            # Refresh the plot
            self.canvas.draw_idle()
            self.figure.canvas.flush_events()

        except Exception as e:
            print(f"[Plot Update Error]: {e}")
            traceback.print_exc()

        # Schedule next update
        self.update_id = self.root.after(1000, self.update_plot)

    def on_close(self):
        if self.update_id is not None:
            self.root.after_cancel(self.update_id)
        self.root.destroy()
        sys.exit()


def launch_gui():
    root = tk.Tk()
    app = AnomalyPlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
