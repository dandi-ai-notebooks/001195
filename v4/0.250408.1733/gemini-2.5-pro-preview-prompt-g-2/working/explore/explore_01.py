# This script loads and plots the current_clamp-response-01-ch-0 data
# from the NWB file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the data
acquisition = nwb.acquisition
current_clamp_response_01_ch_0 = acquisition["current_clamp-response-01-ch-0"]
data = current_clamp_response_01_ch_0.data[:]
rate = current_clamp_response_01_ch_0.rate
starting_time = current_clamp_response_01_ch_0.starting_time
timestamps = np.arange(len(data)) / rate + starting_time

# Plot
sns.set_theme()
plt.figure(figsize=(10, 5))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel(f"Voltage ({current_clamp_response_01_ch_0.unit})")
plt.title("Current Clamp Response 01 ch 0")
plt.savefig("explore/plot_01.png")
plt.close()

print("Plot saved to explore/plot_01.png")