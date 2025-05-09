# This script loads and plots the first current clamp response and stimulus.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get data
response = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus = nwb.stimulus["stimulus-01-ch-0"]

# Get time vector
sampling_rate = response.rate
t = np.arange(response.data.shape[0]) / sampling_rate + response.starting_time

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, response.data[:] * response.conversion, label='Response')
plt.ylabel(f'Voltage ({response.unit})')
plt.title('Current Clamp Series 01 - Channel 0')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, stimulus.data[:] * stimulus.conversion, label='Stimulus', color='orange')
plt.xlabel(f'Time ({response.starting_time_unit})')
plt.ylabel(f'Current ({stimulus.unit})')
plt.legend()

plt.tight_layout()
plt.savefig('explore/current_clamp_series_01_ch_0.png')

io.close()