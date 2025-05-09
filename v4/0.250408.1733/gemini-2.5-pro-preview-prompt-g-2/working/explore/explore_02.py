# This script loads and plots the current_clamp-response-01-ch-0 data
# and its corresponding stimulus from the NWB file.
# It attempts to correct the voltage scale.

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

# Get the response data
acquisition = nwb.acquisition
current_clamp_response_01_ch_0 = acquisition["current_clamp-response-01-ch-0"]
response_data = current_clamp_response_01_ch_0.data[:] * 1000 # Convert to mV assuming original unit was V
response_rate = current_clamp_response_01_ch_0.rate
response_starting_time = current_clamp_response_01_ch_0.starting_time
response_timestamps = np.arange(len(response_data)) / response_rate + response_starting_time
response_unit = "mV" # Assuming Volts to mV conversion

# Get the stimulus data
stimulus = nwb.stimulus
stimulus_01_ch_0 = stimulus["stimulus-01-ch-0"]
stimulus_data = stimulus_01_ch_0.data[:]
stimulus_rate = stimulus_01_ch_0.rate
stimulus_starting_time = stimulus_01_ch_0.starting_time
stimulus_timestamps = np.arange(len(stimulus_data)) / stimulus_rate + stimulus_starting_time
stimulus_unit = stimulus_01_ch_0.unit
stimulus_conversion = stimulus_01_ch_0.conversion
stimulus_data_converted = stimulus_data * stimulus_conversion

# Plot
sns.set_theme()
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(f"Voltage ({response_unit})", color=color)
ax1.plot(response_timestamps, response_data, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title("Current Clamp Response 01 ch 0 and Stimulus")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel(f"Stimulus ({stimulus_unit})", color=color)
ax2.plot(stimulus_timestamps, stimulus_data_converted, color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("explore/plot_02.png")
plt.close()

print("Plot saved to explore/plot_02.png")
print(f"Response data description: {current_clamp_response_01_ch_0.description}")
print(f"Stimulus data description: {stimulus_01_ch_0.description}")