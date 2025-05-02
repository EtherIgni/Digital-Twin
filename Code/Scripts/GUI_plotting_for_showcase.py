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

data_folder="C:/Users/james\Documents/repo\Digital-Twin/Code/Data/bypass_both_valves"

# File paths
#data_folder = "C:/Users/DAQ-User/Documents/LabVIEW Data/3Loop/Run 1"


#data_folder= data_folder + str({run_number})
physical_data_path = os.path.join(data_folder, "filtered_data_new.csv")
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

        # Initialize data tracking
        self.current_index = 0  # Tracks how much data has been plotted
        self.data_chunk_size = 10  # Number of points to add per update

        # Load full data into memory
        self.full_physical_data = pd.read_csv(physical_data_path, header=None).to_numpy()
        self.full_model_data = pd.read_csv(model_data_path, header=None).to_numpy()[:, :8]
        self.full_anomaly_data = pd.read_csv(anomaly_data_path, header=None).to_numpy()

        # Align time axes
        self.full_physical_data[:, 0] -= self.full_model_data[0, 0]
        self.full_physical_data = self.full_physical_data[320:1020,:]
        self.full_model_data[:, 0] -= self.full_model_data[0, 0]
        self.full_model_data = self.full_model_data[300:1020, :]
        self.full_anomaly_data[:, 0] -= self.full_anomaly_data[0, 0]
        self.full_anomaly_data = self.full_anomaly_data[300:1000, :]
        print(self.full_physical_data[0, 0])
        print(self.full_model_data[0, 0])
        print(self.full_anomaly_data[0, 0])

        # Initialize empty data for plotting
        self.physical_data = np.empty((0, self.full_physical_data.shape[1]))
        self.model_data = np.empty((0, self.full_model_data.shape[1]))
        self.anomaly_data = np.empty((0, self.full_anomaly_data.shape[1]))

        self.update_plot()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_plot(self):
        try:
            # Add the next chunk of data
            next_index = self.current_index + self.data_chunk_size
            
            # Check if the index exceeds the maximum value
            if next_index >= len(self.full_physical_data):
                # Reset the index and clear the plot
                self.current_index = 0
                self.physical_data = np.empty((0, self.full_physical_data.shape[1]))
                self.model_data = np.empty((0, self.full_model_data.shape[1]))
                self.anomaly_data = np.empty((0, self.full_anomaly_data.shape[1]))
                self.ax.clear()
                next_index = self.current_index + self.data_chunk_size
            
            # Append new data to the existing data
            self.physical_data = np.vstack(
                [self.physical_data, self.full_physical_data[self.current_index:next_index, :]]
            )
            self.model_data = np.vstack(
                [self.model_data, self.full_model_data[self.current_index:next_index, :]]
            )
            self.anomaly_data = np.vstack(
                [self.anomaly_data, self.full_anomaly_data[self.current_index:next_index, :]]
            )

            # Update the current index
            self.current_index = next_index

            # Clear the previous plot
            self.ax.clear()

            # Plot physical temperature
            colors = plt.cm.viridis(np.linspace(0, 1, 7))
            for idx, col in enumerate(range(5, self.physical_data.shape[1] - 1)):
                if col == 11:
                    self.ax.plot(
                        self.physical_data[:, 0],
                        self.physical_data[:, col + 1],
                        label=f"Physical Temp {col - 3}",
                        color=colors[idx],
                    )
                else:
                    self.ax.plot(
                        self.physical_data[:, 0],
                        self.physical_data[:, col],
                        label=f"Physical Temp {col - 4}",
                        color=colors[idx],
                    )

            # Plot model temperature
            for idx, col in enumerate(range(1, self.model_data.shape[1])):
                if col == 7:
                    self.ax.plot(
                        self.model_data[:, 0],
                        self.model_data[:, col],
                        "--",
                        label=f"Model Temp {col + 1}",
                        color=colors[idx],
                    )
                else:
                    self.ax.plot(
                        self.model_data[:, 0],
                        self.model_data[:, col],
                        "--",
                        label=f"Model Temp {col}",
                        color=colors[idx],
                    )

            # Plot anomaly regions
            is_anomaly = False
            for i in range(len(self.anomaly_data)):
                if self.anomaly_data[i, -1] == 1 and not is_anomaly:
                    is_anomaly = True
                    start_idx = i
                elif self.anomaly_data[i, -1] == 0 and is_anomaly:
                    is_anomaly = False
                    end_idx = i - 1
                    self.ax.axvspan(
                        self.anomaly_data[start_idx, 0],
                        self.anomaly_data[end_idx, 0],
                        color="red",
                        alpha=0.3,
                    )
            if is_anomaly:
                self.ax.axvspan(
                    self.anomaly_data[start_idx, 0],
                    self.anomaly_data[-1, 0],
                    color="red",
                    alpha=0.3,
                    label="Anomaly",
                )

            # Labels and legend
            self.ax.set_title("Anomaly Detection", fontsize=20)
            self.ax.text(
                0.87, 1.01,  # Position: x=0.5 (centered), y=1.02 (slightly above the plot)
                "10x actual update speed",  # Text to display
                fontsize=12,  # Smaller font size
                ha="center",  # Horizontal alignment
                va="bottom",  # Vertical alignment
                transform=self.ax.transAxes  # Use axis coordinates (0 to 1)
            )   
            self.ax.set_xlabel("Time (s)", fontsize=16)
            self.ax.set_ylabel("Temperature (C)", fontsize=16)
            self.ax.tick_params(axis="both", labelsize=14)
            from matplotlib.lines import Line2D

            solid_line = Line2D(
                [0], [0], color="black", linestyle="-", label="Measured Data"
            )
            dotted_line = Line2D(
                [0], [0], color="black", linestyle="--", label="Physics Model"
            )
            highlight_patch = Patch(color="red", alpha=0.3, label="Anomaly")

            # Add everything to the legend
            self.ax.legend(
                handles=[solid_line, dotted_line, highlight_patch], fontsize=12
            )

            # Refresh the plot
            self.canvas.draw_idle()
            self.figure.canvas.flush_events()

        except Exception as e:
            print(f"[Plot Update Error]: {e}")
            traceback.print_exc()

        # Schedule next update
        self.update_id = self.root.after(1000, self.update_plot)  # Update every second

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
