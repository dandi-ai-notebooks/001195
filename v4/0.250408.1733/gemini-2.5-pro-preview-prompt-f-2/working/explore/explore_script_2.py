# This script loads and plots a small segment of a VoltageClampSeries
# from the NWB file to visualize the raw data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn theme
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access a VoltageClampSeries
# From the nwb-file-info, 'voltage_clamp-response-22-ch-0' is an available series.
series_name = "voltage_clamp-response-22-ch-0"
voltage_clamp_series = nwb.acquisition[series_name]

# Load a small segment of data (e.g., first 1000 points)
data_segment = voltage_clamp_series.data[:1000]
sampling_rate = voltage_clamp_series.rate
time_segment = np.arange(len(data_segment)) / sampling_rate

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time_segment, data_segment)
plt.xlabel(f"Time (s)")
plt.ylabel(f"Current ({voltage_clamp_series.unit})") # Unit for VoltageClampSeries is usually Amperes
plt.title(f"Segment of {series_name}")
plt.grid(True)
plt.savefig("explore/plot_voltage_clamp_22_ch0.png")
plt.close()

print(f"Successfully plotted a segment of {series_name} to explore/plot_voltage_clamp_22_ch0.png")

io.close()