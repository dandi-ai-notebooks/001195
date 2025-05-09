# This script loads and plots the current_clamp-response-02-ch-0 data
# and its corresponding stimulus from the NWB file, applying conversion factors.

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
response_series_name = "current_clamp-response-02-ch-0"
response_series = acquisition[response_series_name]
response_data_raw = response_series.data[:]
response_conversion = response_series.conversion
response_data_converted = response_data_raw * response_conversion
response_rate = response_series.rate
response_starting_time = response_series.starting_time
response_timestamps = np.arange(len(response_data_converted)) / response_rate + response_starting_time
response_unit = response_series.unit

# Get the stimulus data
stimulus = nwb.stimulus
stimulus_series_name = "stimulus-02-ch-0"
stimulus_series = stimulus[stimulus_series_name]
stimulus_data_raw = stimulus_series.data[:]
stimulus_conversion = stimulus_series.conversion
stimulus_data_converted = stimulus_data_raw * stimulus_conversion
stimulus_rate = stimulus_series.rate
stimulus_starting_time = stimulus_series.starting_time
stimulus_timestamps = np.arange(len(stimulus_data_converted)) / stimulus_rate + stimulus_starting_time
stimulus_unit = stimulus_series.unit

# Plot
sns.set_theme()
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(f"Voltage ({response_unit})", color=color)
ax1.plot(response_timestamps, response_data_converted, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title(f"{response_series_name} and {stimulus_series_name}")

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel(f"Stimulus ({stimulus_unit})", color=color)
ax2.plot(stimulus_timestamps, stimulus_data_converted, color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig("explore/plot_03.png")
plt.close()

print("Plot saved to explore/plot_03.png")
print(f"Response data description: {response_series.description}")
print(f"Stimulus data description: {stimulus_series.description}")