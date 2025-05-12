"""
This script analyzes the current step protocols in the NWB file.
We'll examine multiple recordings, extract responses to different current levels,
and calculate basic electrophysiological properties.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create the directory for plots if it doesn't exist
os.makedirs('explore', exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/9029e9f3-949a-4cf3-ade5-48542a1a59b7/download/"
print(f"Loading NWB file from {url}...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"NWB file loaded for subject {nwb.subject.subject_id}, session {nwb.session_description}")

# Get information about the stimulus types
seq_df = nwb.icephys_sequential_recordings.to_dataframe()
print("\nAvailable stimulus types:")
for i, stim_type in enumerate(seq_df['stimulus_type'].unique()):
    print(f"  {i+1}. {stim_type}")

# Find all the recordings with step protocols
step_protocol_recordings = []
for series_name, series in nwb.acquisition.items():
    if 'step' in series.description.lower() and 'ch-0' in series_name:  # Focus on channel 0
        step_protocol_recordings.append(series_name)

print(f"\nFound {len(step_protocol_recordings)} recordings with step protocols on channel 0")

# Organize recordings by current amplitude
print("\nOrganizing step recordings by current amplitude...")

# Dictionary to store results: recording_name -> (series, stimulus, current_amplitude)
step_data = {}

for rec_name in step_protocol_recordings:
    # Get the corresponding stimulus name
    stim_name = rec_name.replace('current_clamp-response', 'stimulus')
    stim_name = stim_name.replace('-ch-0', '-ch-0')  # Ensure we're looking at the right channel
    
    if stim_name in nwb.stimulus:
        stimulus = nwb.stimulus[stim_name]
        response = nwb.acquisition[rec_name]
        
        # Extract stimulus amplitude - assuming a step protocol with constant current during the step
        # Taking the most common non-zero value as the step amplitude
        stim_data = stimulus.data[:] * stimulus.conversion
        non_zero_values = stim_data[stim_data != 0]
        if len(non_zero_values) > 0:
            # Use the most frequent non-zero value (mode) as the current amplitude
            unique_values, counts = np.unique(non_zero_values, return_counts=True)
            current_amplitude = unique_values[np.argmax(counts)]
            step_data[rec_name] = (response, stimulus, current_amplitude)

# Sort recordings by current amplitude
sorted_recordings = sorted(step_data.items(), key=lambda x: x[1][2])

print(f"Processed {len(sorted_recordings)} step recordings")

# Create an I-V curve - analyze responses to different current amplitudes
current_amplitudes = []
voltage_responses = []

# Time window for measuring steady-state voltage (seconds from start of step)
steady_state_window = (0.8, 0.9)  # 800-900ms after step start

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)

# Color map for different current amplitudes
cmap = plt.cm.viridis
colors = [cmap(i/len(sorted_recordings)) for i in range(len(sorted_recordings))]

# Plot a selection of traces (not too many to keep the plot readable)
max_traces = 10
step_selection = np.linspace(0, len(sorted_recordings)-1, min(max_traces, len(sorted_recordings))).astype(int)

for i in step_selection:
    recording_name, (response, stimulus, current_amp) = sorted_recordings[i]
    
    # Create time array
    time = np.arange(response.data.shape[0]) / response.rate + response.starting_time
    
    # Convert data
    voltage_data = response.data[:] * response.conversion
    
    # Find step start and end
    stim_data = stimulus.data[:] * stimulus.conversion
    step_start_idx = np.where(np.diff(stim_data) != 0)[0][0] + 1
    step_end_idx = np.where(np.diff(stim_data) != 0)[0][1] + 1
    
    step_start_time = time[step_start_idx]
    step_end_time = time[step_end_idx]
    
    # Plot the trace
    plt.plot(time, voltage_data, label=f"{current_amp*1e12:.0f} pA", color=colors[i])
    
    # Calculate steady-state voltage response (average over the specified window)
    steady_start_idx = step_start_idx + int(steady_state_window[0] * response.rate)
    steady_end_idx = step_start_idx + int(steady_state_window[1] * response.rate)
    steady_voltage = np.mean(voltage_data[steady_start_idx:steady_end_idx])
    
    # Store for I-V curve
    current_amplitudes.append(current_amp)
    voltage_responses.append(steady_voltage)

plt.axvspan(step_start_time, step_end_time, alpha=0.2, color='gray')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Selected Voltage Responses to Different Current Steps')
plt.grid(True)
plt.legend(loc='upper right')

# Plot I-V curve
plt.subplot(2, 1, 2)
plt.scatter(np.array(current_amplitudes) * 1e12, np.array(voltage_responses) * 1e3, color='blue')
plt.plot(np.array(current_amplitudes) * 1e12, np.array(voltage_responses) * 1e3, 'b-')
plt.xlabel('Current (pA)')
plt.ylabel('Steady-State Voltage (mV)')
plt.title('Current-Voltage Relationship (I-V Curve)')
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/iv_curve_analysis.png')
plt.close()

# Calculate membrane properties
print("\nCalculating basic membrane properties...")

# Let's use a hyperpolarizing step for passive membrane properties (choose a negative current)
neg_current_recordings = [rec for rec in sorted_recordings if rec[1][2] < 0]
if neg_current_recordings:
    # Select a hyperpolarizing step
    recording_name, (response, stimulus, current_amp) = neg_current_recordings[0]
    
    # Create time array
    time = np.arange(response.data.shape[0]) / response.rate + response.starting_time
    
    # Convert data
    voltage_data = response.data[:] * response.conversion
    
    # Find step start and end
    stim_data = stimulus.data[:] * stimulus.conversion
    step_changes = np.where(np.diff(stim_data) != 0)[0]
    if len(step_changes) >= 2:
        step_start_idx = step_changes[0] + 1
        step_end_idx = step_changes[1] + 1
        
        step_start_time = time[step_start_idx]
        step_end_time = time[step_end_idx]
        
        # Get baseline and steady-state voltage
        baseline_window = (step_start_idx - int(0.1 * response.rate), step_start_idx)
        baseline_voltage = np.mean(voltage_data[baseline_window[0]:baseline_window[1]])
        
        steady_start_idx = step_start_idx + int(0.8 * response.rate)
        steady_end_idx = step_start_idx + int(0.9 * response.rate)
        steady_voltage = np.mean(voltage_data[steady_start_idx:steady_end_idx])
        
        # Calculate voltage deflection
        delta_v = steady_voltage - baseline_voltage
        
        # Calculate input resistance (R = V/I)
        input_resistance = delta_v / current_amp
        
        # Estimate membrane time constant
        # Find the exponential decay after step onset
        decay_window = 100  # number of samples for fitting
        decay_data = voltage_data[step_start_idx:step_start_idx+decay_window]
        decay_time = time[step_start_idx:step_start_idx+decay_window] - time[step_start_idx]
        
        # Normalize the decay curve
        norm_decay = (decay_data - steady_voltage) / (decay_data[0] - steady_voltage)
        
        # Fit exponential: y = exp(-t/tau)
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(t, tau):
                return np.exp(-t/tau)
            
            popt, _ = curve_fit(exp_decay, decay_time, norm_decay)
            tau_m = popt[0]  # membrane time constant in seconds
            
            print(f"\nMembrane properties (from hyperpolarizing step {current_amp*1e12:.0f} pA):")
            print(f"  Baseline voltage: {baseline_voltage*1e3:.2f} mV")
            print(f"  Steady-state voltage: {steady_voltage*1e3:.2f} mV")
            print(f"  Voltage deflection: {delta_v*1e3:.2f} mV")
            print(f"  Input resistance: {input_resistance*1e-6:.2f} MΩ")
            print(f"  Membrane time constant: {tau_m*1e3:.2f} ms")
            
            # Plot the hyperpolarizing step with tau_m estimate
            plt.figure(figsize=(12, 5))
            plt.plot(time, voltage_data*1e3)
            plt.axvspan(time[step_start_idx], time[step_start_idx+decay_window], alpha=0.3, color='red', label=f'Tau estimate: {tau_m*1e3:.2f} ms')
            plt.axhspan(baseline_voltage*1e3, steady_voltage*1e3, alpha=0.2, color='blue', label=f'ΔV: {delta_v*1e3:.2f} mV')
            plt.axvspan(time[baseline_window[0]], time[baseline_window[1]], alpha=0.2, color='green', label='Baseline')
            plt.axvspan(time[steady_start_idx], time[steady_end_idx], alpha=0.2, color='purple', label='Steady-state')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (mV)')
            plt.title(f'Hyperpolarizing Step Response ({current_amp*1e12:.0f} pA)')
            plt.legend()
            plt.grid(True)
            plt.savefig('explore/membrane_properties.png')
            plt.close()
        
        except Exception as e:
            print(f"Error fitting exponential decay: {e}")
    else:
        print("Could not identify clear step changes in the stimulus")
else:
    print("No hyperpolarizing step recordings found")


# Look for depolarizing steps that might have action potentials
print("\nLooking for action potentials in depolarizing steps...")

# Helper function to detect action potentials
def detect_spikes(voltage, threshold=-0.03):  # -30 mV threshold
    # Find peaks above threshold
    crossings = np.where(np.diff((voltage > threshold).astype(int)) > 0)[0]
    return crossings

# Find a depolarizing step with spikes
for recording_name, (response, stimulus, current_amp) in reversed(sorted_recordings):  # Start from highest current
    if current_amp > 0:  # Only depolarizing steps
        # Create time array
        time = np.arange(response.data.shape[0]) / response.rate + response.starting_time
        
        # Convert data
        voltage_data = response.data[:] * response.conversion
        
        # Find step start and end
        stim_data = stimulus.data[:] * stimulus.conversion
        step_changes = np.where(np.diff(stim_data) != 0)[0]
        if len(step_changes) >= 2:
            step_start_idx = step_changes[0] + 1
            step_end_idx = step_changes[1] + 1
            
            # Look for spikes during the step
            step_voltage = voltage_data[step_start_idx:step_end_idx]
            spike_indices = detect_spikes(step_voltage)
            
            if len(spike_indices) > 0:
                print(f"Found {len(spike_indices)} action potentials in recording {recording_name} with {current_amp*1e12:.0f} pA")
                
                # Plot this trace with detected spikes
                plt.figure(figsize=(12, 5))
                plt.plot(time, voltage_data*1e3)
                
                # Mark the step duration
                plt.axvspan(time[step_start_idx], time[step_end_idx], alpha=0.2, color='gray', label='Current step')
                
                # Mark detected spikes
                for spike_idx in spike_indices:
                    absolute_idx = step_start_idx + spike_idx
                    plt.axvline(time[absolute_idx], color='red', linestyle='--', alpha=0.7)
                
                plt.xlabel('Time (s)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Action Potentials During {current_amp*1e12:.0f} pA Current Step')
                plt.legend()
                plt.grid(True)
                plt.savefig('explore/action_potentials.png')
                plt.close()
                
                # Also create a plot focusing on a single spike if there are any
                if len(spike_indices) > 0:
                    # Focus on the first spike
                    first_spike_idx = step_start_idx + spike_indices[0]
                    
                    # Window around the spike (ms)
                    window_ms = 5
                    samples_per_ms = response.rate / 1000
                    window_samples = int(window_ms * samples_per_ms)
                    
                    spike_start = max(0, first_spike_idx - window_samples)
                    spike_end = min(len(time), first_spike_idx + window_samples)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(time[spike_start:spike_end], voltage_data[spike_start:spike_end]*1e3)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Voltage (mV)')
                    plt.title('Action Potential Waveform')
                    plt.grid(True)
                    plt.savefig('explore/action_potential_waveform.png')
                    plt.close()
                
                break  # Found a recording with spikes, no need to continue
            else:
                print(f"No action potentials detected in recording {recording_name} with {current_amp*1e12:.0f} pA")

print("\nAnalysis complete!")