# This script loads and plots multiple sweeps from the current clamp
# episodic stimulation protocol, along with their corresponding stimuli.

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

# Define sweeps to plot
sweep_numbers_to_plot = ["05", "10", "15"] # Example sweeps
num_sweeps = len(sweep_numbers_to_plot)

sns.set_theme()
fig, axes = plt.subplots(num_sweeps, 1, figsize=(10, 4 * num_sweeps), sharex=True)
if num_sweeps == 1: # Make axes an array even if only one subplot
    axes = [axes]

for i, sweep_num_str in enumerate(sweep_numbers_to_plot):
    ax1 = axes[i]

    # Get response data
    response_series_name = f"current_clamp-response-{sweep_num_str}-ch-0"
    if response_series_name not in acquisition:
        print(f"Response series {response_series_name} not found. Skipping.")
        continue
    response_series = acquisition[response_series_name]
    response_data_raw = response_series.data[:]
    response_conversion = response_series.conversion
    response_data_converted = response_data_raw * response_conversion
    response_rate = response_series.rate
    response_starting_time = response_series.starting_time
    response_timestamps = np.arange(len(response_data_converted)) / response_rate + response_starting_time
    response_unit = response_series.unit

    # Get stimulus data
    stimulus_series_name = f"stimulus-{sweep_num_str}-ch-0"
    if stimulus_series_name not in stimulus_dict:
        print(f"Stimulus series {stimulus_series_name} not found. Skipping.")
        continue
    stimulus_series = stimulus_dict[stimulus_series_name]
    stimulus_data_raw = stimulus_series.data[:]
    stimulus_conversion = stimulus_series.conversion
    # For display, find the main stimulus amplitude (pA)
    # The data is often constant during the pulse.
    # We multiply by 1e12 to get pA from A if original unit is A
    stim_val_pA = 0
    if len(np.unique(stimulus_data_raw)) > 1 and stimulus_conversion != 0: # Check if stimulus is not flat zero
        # find a segment where stimulus is likely non-zero
        non_zero_stim = stimulus_data_raw[stimulus_data_raw != 0]
        if len(non_zero_stim) > 0:
             # take the median of non-zero values as representative
            stim_val_pA = np.median(non_zero_stim) * stimulus_conversion * 1e12


    stimulus_data_converted = stimulus_data_raw * stimulus_conversion
    stimulus_rate = stimulus_series.rate
    stimulus_starting_time = stimulus_series.starting_time
    stimulus_timestamps = np.arange(len(stimulus_data_converted)) / stimulus_rate + stimulus_starting_time
    stimulus_unit = stimulus_series.unit

    # Plot response
    color_response = 'tab:blue'
    ax1.set_ylabel(f"Voltage ({response_unit})", color=color_response)
    ax1.plot(response_timestamps, response_data_converted, color=color_response)
    ax1.tick_params(axis='y', labelcolor=color_response)
    ax1.set_title(f"Sweep {sweep_num_str} (Stim: ~{stim_val_pA:.0f} pA)")


    # Plot stimulus on a twin axis
    ax2 = ax1.twinx()
    color_stimulus = 'tab:red'
    ax2.set_ylabel(f"Stimulus ({stimulus_unit})", color=color_stimulus)
    ax2.plot(stimulus_timestamps, stimulus_data_converted, color=color_stimulus, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_stimulus)

    if i == num_sweeps - 1: # Only set xlabel for the last subplot
        ax1.set_xlabel("Time (s)")

fig.tight_layout()
plt.savefig("explore/plot_04.png")
plt.close()

print("Plot saved to explore/plot_04.png")
print(f"Plotted sweeps: {', '.join(sweep_numbers_to_plot)}")