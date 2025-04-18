# This script loads an NWB file, extracts current clamp response and stimulus data, and generates plots.
# It saves the plots to the explore/ directory as PNG images for further analysis.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract data for analysis
current_clamp_series = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus_series = nwb.stimulus["stimulus-01-ch-0"]

# Time axis for the plot
data_length = current_clamp_series.data.shape[0]
time = np.arange(data_length) / current_clamp_series.rate

# Ensure the datasets are converted to NumPy arrays
current_clamp_data = current_clamp_series.data[:]
stimulus_data = stimulus_series.data[:]

# Plot current clamp response
plt.figure(figsize=(12, 6))
plt.plot(time, current_clamp_data, label="Response", color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Current (V)')
plt.title('Current Clamp Response')
plt.legend()
plt.savefig('explore/current_clamp_response.png')
plt.close()

# Plot stimulus series
plt.figure(figsize=(12, 6))
plt.plot(time, stimulus_data, label="Stimulus", color='green')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Stimulus Series')
plt.legend()
plt.savefig('explore/stimulus_series.png')
plt.close()

# Close file
io.close()
remote_file.close()