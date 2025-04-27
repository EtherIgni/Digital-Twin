import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# File paths
data_folder = "C:/Users/james/Documents/repo/Digital-Twin/Code/Data/heater_increase"

physical_data_path = os.path.join(data_folder, "filtered_data_new.csv")
model_data_path = os.path.join(data_folder, "simulated_data.csv")
anomaly_data_path = os.path.join(data_folder, "anomaly.csv")

# Subtract the first value of model data to align the time axis
#read data
physical_data = pd.read_csv(physical_data_path, header=None).to_numpy()
model_data = pd.read_csv(model_data_path, header=None).to_numpy()[:, :8]
anomaly_data = pd.read_csv(anomaly_data_path, header=None).to_numpy()

# align time axis
physical_data[:, 0] -= model_data[0, 0]
physical_data = physical_data[20:-10,:]
print(physical_data[0,0])
model_data[:, 0] -= model_data[0, 0]
model_data = model_data[:,:]
anomaly_data[:, 0] -= anomaly_data[0, 0]
anomaly_data = anomaly_data[:,:]

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot physical temperature
colors = plt.cm.viridis(np.linspace(0, 1, 7))
for idx, col in enumerate(range(5, physical_data.shape[1] - 1)):
    if col == 11:
        ax.plot(physical_data[:, 0], physical_data[:, col + 1], label=f"Physical Temp {col - 3}", color=colors[idx])
    else:
        ax.plot(physical_data[:, 0], physical_data[:, col], label=f"Physical Temp {col - 4}", color=colors[idx])

# Plot model temperature
for idx, col in enumerate(range(1, model_data.shape[1])):
    if col == 7:
        ax.plot(model_data[:, 0], model_data[:, col], '--', label=f"Model Temp {col + 1}", color=colors[idx])
    else:
        ax.plot(model_data[:, 0], model_data[:, col], '--', label=f"Model Temp {col}", color=colors[idx])

# Plot anomaly regions
is_anomaly = False
for i in range(len(anomaly_data)):
    if anomaly_data[i, -1] == 1 and not is_anomaly:
        is_anomaly = True
        start_idx = i
    elif anomaly_data[i, -1] == 0 and is_anomaly:
        is_anomaly = False
        end_idx = i - 1
        ax.axvspan(anomaly_data[start_idx, 0], anomaly_data[end_idx, 0], color='red', alpha=0.3)
if is_anomaly:
    ax.axvspan(anomaly_data[start_idx, 0], anomaly_data[-1, 0], color='red', alpha=0.3, label="Anomaly")

# Add a vertical bar at a specific time (e.g., time = 100)
event_start = 280  # Example time for the vertical bar
event_end   = 643
ax.axvline(x=event_start, color='red', linestyle='--', linewidth=2, label='Anomaly Start')
ax.axvline(x=event_end, color='red', linestyle='--', linewidth=2, label='Anomaly End')

# Labels and legend
ax.set_xlabel("Time (s)", fontsize=16)
ax.set_ylabel("Temperature (C)", fontsize=16)
ax.tick_params(axis='both', labelsize=14)

# Custom legend
solid_line = Line2D([0], [0], color='black', linestyle='-', label='Measured Data')
dotted_line = Line2D([0], [0], color='black', linestyle='--', label='Physics Model')
highlight_patch = Patch(color='red', alpha=0.3, label='Anomaly Detected')
start_line = Line2D([0], [0], color='red', linestyle='--', label='Anomaly Bounds')


# Add everything to the legend
ax.legend(handles=[solid_line, dotted_line, highlight_patch, start_line], fontsize=12)


# Show the plot
plt.tight_layout()
plt.show()