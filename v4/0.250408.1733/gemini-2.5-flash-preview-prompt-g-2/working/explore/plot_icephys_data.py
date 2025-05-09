# This script loads and plots the first two current clamp response and stimulus series from the NWB file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

acquisition = nwb.acquisition
stimulus = nwb.stimulus

# Get the first two current clamp response series channel 0
response_series_01 = acquisition["current_clamp-response-01-ch-0"]
response_series_02 = acquisition["current_clamp-response-02-ch-0"]

# Get the first two current clamp stimulus series channel 0
stimulus_series_01 = stimulus["stimulus-01-ch-0"]
stimulus_series_02 = stimulus["stimulus-02-ch-0"]

# Get the data and time information
data_response_01 = response_series_01.data[:]
sampling_rate_response_01 = response_series_01.rate
starting_time_response_01 = response_series_01.starting_time
timestamps_response_01 = starting_time_response_01 + np.arange(len(data_response_01)) / sampling_rate_response_01

data_response_02 = response_series_02.data[:]
sampling_rate_response_02 = response_series_02.rate
starting_time_response_02 = response_series_02.starting_time
timestamps_response_02 = starting_time_response_02 + np.arange(len(data_response_02)) / sampling_rate_response_02

data_stimulus_01 = stimulus_series_01.data[:]
sampling_rate_stimulus_01 = stimulus_series_01.rate
starting_time_stimulus_01 = stimulus_series_01.starting_time
timestamps_stimulus_01 = starting_time_stimulus_01 + np.arange(len(data_stimulus_01)) / sampling_rate_stimulus_01

data_stimulus_02 = stimulus_series_02.data[:]
sampling_rate_stimulus_02 = stimulus_series_02.rate
starting_time_stimulus_02 = stimulus_series_02.starting_time
timestamps_stimulus_02 = starting_time_stimulus_02 + np.arange(len(data_stimulus_02)) / sampling_rate_stimulus_02


# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(timestamps_response_01, data_response_01, label='Response 01 (ch 0)')
plt.plot(timestamps_response_02, data_response_02, label='Response 02 (ch 0)')
plt.ylabel(f'Voltage ({response_series_01.unit})')
plt.title('Current Clamp Responses (Channel 0)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(timestamps_stimulus_01, data_stimulus_01, label='Stimulus 01 (ch 0)')
plt.plot(timestamps_stimulus_02, data_stimulus_02, label='Stimulus 02 (ch 0)')
plt.xlabel('Time (s)')
plt.ylabel(f'Current ({stimulus_series_01.unit})')
plt.title('Current Clamp Stimuli (Channel 0)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/icephys_channel_0_plot.png')

plt.figure(figsize=(12, 8))

# Get the first two current clamp response series channel 1
response_series_01_ch1 = acquisition["current_clamp-response-01-ch-1"]
response_series_02_ch1 = acquisition["current_clamp-response-02-ch-1"]

# Get the first two current clamp stimulus series channel 1
stimulus_series_01_ch1 = stimulus["stimulus-01-ch-1"]
stimulus_series_02_ch1 = stimulus["stimulus-02-ch-1"]

# Get the data and time information
data_response_01_ch1 = response_series_01_ch1.data[:]
sampling_rate_response_01_ch1 = response_series_01_ch1.rate
starting_time_response_01_ch1 = response_series_01_ch1.starting_time
timestamps_response_01_ch1 = starting_time_response_01_ch1 + np.arange(len(data_response_01_ch1)) / sampling_rate_response_01_ch1

data_response_02_ch1 = response_series_02_ch1.data[:]
sampling_rate_response_02_ch1 = response_series_02_ch1.rate
starting_time_response_02_ch1 = response_series_02_ch1.starting_time
timestamps_response_02_ch1 = starting_time_response_02_ch1 + np.arange(len(data_response_02_ch1)) / sampling_rate_response_02_ch1

data_stimulus_01_ch1 = stimulus_series_01_ch1.data[:]
sampling_rate_stimulus_01_ch1 = stimulus_series_01_ch1.rate
starting_time_stimulus_01_ch1 = stimulus_series_01_ch1.starting_time
timestamps_stimulus_01_ch1 = starting_time_stimulus_01_ch1 + np.arange(len(data_stimulus_01_ch1)) / sampling_rate_stimulus_01_ch1

data_stimulus_02_ch1 = stimulus_series_02_ch1.data[:]
sampling_rate_stimulus_02_ch1 = stimulus_series_02_ch1.rate
starting_time_stimulus_02_ch1 = stimulus_series_02_ch1.starting_time
timestamps_stimulus_02_ch1 = starting_time_stimulus_02_ch1 + np.arange(len(data_stimulus_02_ch1)) / sampling_rate_stimulus_02_ch1

# Plotting
plt.subplot(2, 1, 1)
plt.plot(timestamps_response_01_ch1, data_response_01_ch1, label='Response 01 (ch 1)')
plt.plot(timestamps_response_02_ch1, data_response_02_ch1, label='Response 02 (ch 1)')
plt.ylabel(f'Voltage ({response_series_01_ch1.unit})')
plt.title('Current Clamp Responses (Channel 1)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(timestamps_stimulus_01_ch1, data_stimulus_01_ch1, label='Stimulus 01 (ch 1)')
plt.plot(timestamps_stimulus_02_ch1, data_stimulus_02_ch1, label='Stimulus 02 (ch 1)')
plt.xlabel('Time (s)')
plt.ylabel(f'Current ({stimulus_series_01_ch1.unit})')
plt.title('Current Clamp Stimuli (Channel 1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/icephys_channel_1_plot.png')

# Close the HDF5 file and remote file to free up resources
io.close()
h5_file.close()
remote_file.close()