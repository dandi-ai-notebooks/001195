"""
This script explores the electrophysiology data from the Dandiset.
Specifically, it loads an NWB file containing intracellular recordings and plots 
some of the current clamp responses to understand neuronal activity patterns.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for saving plots
os.makedirs('explore', exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB File Information:")
print(f"Session description: {nwb.session_description}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject description: {nwb.subject.description}")
print(f"Cell ID: {nwb.lab_meta_data['DandiIcephysMetadata'].cell_id}")
print(f"Slice ID: {nwb.lab_meta_data['DandiIcephysMetadata'].slice_id}")
print(f"Targeted layer: {nwb.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")

# Get a list of all the current clamp response data
current_clamp_responses = [nwb.acquisition[key] for key in nwb.acquisition.keys() 
                          if 'current_clamp-response' in key and 'ch-0' in key]

# Get a list of all the stimulus data
stimuli = [nwb.stimulus[key] for key in nwb.stimulus.keys() 
          if 'stimulus' in key and 'ch-0' in key]

# Print information about the current_clamp_responses
print("\nCurrent Clamp Response Information:")
print(f"Number of current clamp responses: {len(current_clamp_responses)}")
print(f"Description of first response: {current_clamp_responses[0].description}")
print(f"Sampling rate: {current_clamp_responses[0].rate} Hz")
print(f"Unit: {current_clamp_responses[0].unit}")
print(f"Data shape of first response: {current_clamp_responses[0].data.shape}")

# Plot a few current clamp responses to visualize the data
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# Time vector (in ms)
time = np.arange(0, current_clamp_responses[0].data.shape[0]) / current_clamp_responses[0].rate * 1000  # Convert to ms

# Plot 5 current clamp responses
for i in range(5):
    idx = i * 4  # Select every 4th response for more variation
    if idx < len(current_clamp_responses):
        data = current_clamp_responses[idx].data[:] * current_clamp_responses[idx].conversion  # Apply conversion factor
        axs[i].plot(time, data, 'b')
        axs[i].set_ylabel('Voltage (V)')
        axs[i].set_title(f'Current Clamp Response {idx+1}')

axs[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('explore/ephys_current_clamp_responses.png')

# Plot stimulus and response together for a specific recording
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Select a specific recording
idx = 10
if idx < len(current_clamp_responses) and idx < len(stimuli):
    # Plot the stimulus
    stimulus_data = stimuli[idx].data[:] * stimuli[idx].conversion  # Apply conversion factor
    axs[0].plot(time, stimulus_data, 'r')
    axs[0].set_ylabel('Current (A)')
    axs[0].set_title(f'Stimulus {idx+1}')
    
    # Plot the response
    response_data = current_clamp_responses[idx].data[:] * current_clamp_responses[idx].conversion  # Apply conversion factor
    axs[1].plot(time, response_data, 'b')
    axs[1].set_ylabel('Voltage (V)')
    axs[1].set_title(f'Response {idx+1}')

axs[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('explore/ephys_stimulus_response_pair.png')

# Plot current-voltage relationship (I-V curve) for a specific time point
# We'll use the peak response for each stimulus level
fig, ax = plt.subplots(figsize=(8, 6))

# Get stimulus amplitudes and peak responses
num_responses = min(20, len(current_clamp_responses))  # Limit to first 20 recordings
stimulus_amplitudes = []
peak_responses = []

for i in range(num_responses):
    if i < len(stimuli):
        # Get stimulus amplitude (use the mean current value during the stimulus)
        stimulus_data = stimuli[i].data[1000:2000] * stimuli[i].conversion  # Use a segment during the stimulus
        stimulus_amplitude = np.mean(stimulus_data)
        stimulus_amplitudes.append(stimulus_amplitude)
        
        # Get peak response (maximum voltage value)
        response_data = current_clamp_responses[i].data[:] * current_clamp_responses[i].conversion
        peak_response = np.max(response_data)
        peak_responses.append(peak_response)

# Sort by stimulus amplitude
sorted_indices = np.argsort(stimulus_amplitudes)
sorted_stimulus_amplitudes = np.array(stimulus_amplitudes)[sorted_indices]
sorted_peak_responses = np.array(peak_responses)[sorted_indices]

# Plot I-V curve
ax.scatter(sorted_stimulus_amplitudes, sorted_peak_responses, c='b', marker='o')
ax.plot(sorted_stimulus_amplitudes, sorted_peak_responses, 'b-', alpha=0.6)
ax.set_xlabel('Current (A)')
ax.set_ylabel('Peak Voltage (V)')
ax.set_title('I-V Relationship')
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('explore/ephys_iv_curve.png')

print("\nPlots saved to explore directory.")