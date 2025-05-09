# This script loads and plots a single sweep (sweep 15)
# from the current clamp episodic stimulation protocol,
# ensuring the x-axis is correctly scaled to the sweep duration.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

acquisition = nwb.acquisition
stimulus_dict = nwb.stimulus

# Define sweep to plot
sweep_num_str = "15"

sns.set_theme()
fig, ax1 = plt.subplots(figsize=(10, 5))

# Get response data
response_series_name = f"current_clamp-response-{sweep_num_str}-ch-0"
response_series = acquisition[response_series_name]
response_data_raw = response_series.data[:]
response_conversion = response_series.conversion
response_data_converted = response_data_raw * response_conversion
response_rate = response_series.rate
response_starting_time = response_series.starting_time
response_timestamps = (np.arange(len(response_data_converted)) / response_rate) # Relative time for this sweep
response_unit = response_series.unit

# Get stimulus data
stimulus_series_name = f"stimulus-{sweep_num_str}-ch-0"
stimulus_series = stimulus_dict[stimulus_series_name]
stimulus_data_raw = stimulus_series.data[:]
stimulus_conversion = stimulus_series.conversion
stimulus_data_converted = stimulus_data_raw * stimulus_conversion
stimulus_rate = stimulus_series.rate
stimulus_starting_time = stimulus_series.starting_time # Absolute start time
stimulus_timestamps = (np.arange(len(stimulus_data_converted)) / stimulus_rate) # Relative time for this sweep
stimulus_unit = stimulus_series.unit

stim_val_pA = 0
if len(np.unique(stimulus_data_raw)) > 1 and stimulus_conversion != 0:
    non_zero_stim = stimulus_data_raw[stimulus_data_raw != 0]
    if len(non_zero_stim) > 0:
        stim_val_pA = np.median(non_zero_stim) * stimulus_conversion * 1e12

# Plot response
color_response = 'tab:blue'
ax1.set_xlabel(f"Time within sweep (s) (Absolute start: {response_starting_time:.2f}s)")
ax1.set_ylabel(f"Voltage ({response_unit})", color=color_response)
ax1.plot(response_timestamps, response_data_converted, color=color_response)
ax1.tick_params(axis='y', labelcolor=color_response)
ax1.set_title(f"Sweep {sweep_num_str} (Stim: ~{stim_val_pA:.0f} pA)")
ax1.set_xlim(response_timestamps[0], response_timestamps[-1]) # Explicitly set x-limits

# Plot stimulus on a twin axis
ax2 = ax1.twinx()
color_stimulus = 'tab:red'
ax2.set_ylabel(f"Stimulus ({stimulus_unit})", color=color_stimulus)
ax2.plot(stimulus_timestamps, stimulus_data_converted, color=color_stimulus, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color_stimulus)
# ax2.set_xlim(stimulus_timestamps[0], stimulus_timestamps[-1]) # X limits are shared

fig.tight_layout()
plt.savefig("explore/plot_05.png")
plt.close()

print("Plot saved to explore/plot_05.png")
print(f"Plotted sweep: {sweep_num_str}")
print(f"Response series description: {response_series.description}")
print(f"Stimulus series description: {stimulus_series.description}")
print(f"Response data shape: {response_data_converted.shape}, Rate: {response_rate} Hz, Duration: {len(response_data_converted)/response_rate:.3f} s")