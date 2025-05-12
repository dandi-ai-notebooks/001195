"""
This script directly searches for recordings with optogenetics-related descriptions
and analyzes the first few found.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
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

# Direct search for optogenetic recordings
opto_responses = []
for series_name, series in nwb.acquisition.items():
    desc = series.description.lower()
    if 'laser' in desc or 'light' in desc or '635nm' in desc:
        opto_responses.append((series_name, desc))

print(f"\nFound {len(opto_responses)} acquisition series with optogenetic descriptions")

if opto_responses:
    print("\nFirst 5 optogenetic recordings found:")
    for i, (name, desc) in enumerate(opto_responses[:5]):
        print(f"  {i+1}. {name}: {desc}")
    
    # Analyze up to the first 3 optogenetic responses
    for i, (response_name, _) in enumerate(opto_responses[:3]):
        response = nwb.acquisition[response_name]
        
        print(f"\nAnalyzing optogenetic response {i+1}: {response_name}")
        print(f"Description: {response.description}")
        print(f"Data shape: {response.data.shape}, Unit: {response.unit}")
        
        # Create time array
        time = np.arange(response.data.shape[0]) / response.rate + response.starting_time
        
        # Plot the trace
        plt.figure(figsize=(12, 5))
        voltage_data = response.data[:] * response.conversion
        plt.plot(time, voltage_data*1e3)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Optogenetic Response: {response_name}')
        plt.grid(True)
        plt.savefig(f'explore/optogenetic_response_{i+1}.png')
        plt.close()
        
        # Look for the corresponding stimulus if available
        stim_name = response_name.replace('response', 'stimulus')
        stim_name = stim_name.replace('current_clamp-', '')
        
        if stim_name in nwb.stimulus:
            stimulus = nwb.stimulus[stim_name]
            stim_data = stimulus.data[:] * stimulus.conversion
            
            # Plot stimulus and response together
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            color = 'tab:blue'
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Voltage (mV)', color=color)
            ax1.plot(time, voltage_data*1e3, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Current (pA)', color=color)
            ax2.plot(time, stim_data*1e12, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f'Optogenetic Stimulus and Response {i+1}')
            fig.tight_layout()
            plt.savefig(f'explore/optogenetic_stimulus_response_{i+1}.png')
            plt.close()
            
            # Try to focus on a shorter segment where something interesting happens
            # First attempt to find stimulus pulses
            pulse_threshold = np.percentile(stim_data, 95)  # Use 95th percentile as threshold
            pulse_indices = np.where(np.diff((stim_data > pulse_threshold).astype(int)) > 0)[0]
            
            if len(pulse_indices) > 0:
                print(f"  Identified {len(pulse_indices)} potential stimulus pulses")
                
                # Focus on the first pulse
                first_pulse_idx = pulse_indices[0]
                
                # Window around the pulse (ms)
                window_ms = 500  # Larger window to see response
                samples_per_ms = response.rate / 1000
                window_samples = int(window_ms * samples_per_ms)
                
                pulse_start = max(0, first_pulse_idx - int(window_samples * 0.1))  # 10% before pulse
                pulse_end = min(len(time), first_pulse_idx + int(window_samples * 0.9))  # 90% after pulse
                
                # Plot the pulse and response
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                color = 'tab:blue'
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Voltage (mV)', color=color)
                ax1.plot(time[pulse_start:pulse_end], voltage_data[pulse_start:pulse_end]*1e3, color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Current (pA)', color=color)
                ax2.plot(time[pulse_start:pulse_end], stim_data[pulse_start:pulse_end]*1e12, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                plt.title(f'Single Optogenetic Pulse and Response {i+1}')
                fig.tight_layout()
                plt.savefig(f'explore/single_optogenetic_pulse_{i+1}.png')
                plt.close()
            else:
                print("  No clear stimulus pulses identified")
                
                # If no clear pulses, just plot a segment in the middle
                segment_length = min(5000, len(time) // 4)  # 5000 samples or 1/4 of data, whichever is smaller
                start_idx = len(time) // 3  # Start 1/3 of the way through
                end_idx = start_idx + segment_length
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                color = 'tab:blue'
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Voltage (mV)', color=color)
                ax1.plot(time[start_idx:end_idx], voltage_data[start_idx:end_idx]*1e3, color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Current (pA)', color=color)
                ax2.plot(time[start_idx:end_idx], stim_data[start_idx:end_idx]*1e12, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                plt.title(f'Optogenetic Recording Segment {i+1}')
                fig.tight_layout()
                plt.savefig(f'explore/optogenetic_segment_{i+1}.png')
                plt.close()
        else:
            print(f"  Corresponding stimulus {stim_name} not found")
else:
    print("No optogenetic responses found")

print("\nExploration complete!")