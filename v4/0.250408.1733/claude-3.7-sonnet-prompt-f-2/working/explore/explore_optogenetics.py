"""
This script explores optogenetic stimulation data in the NWB file.
We look for recordings with laser stimulation and analyze the responses.
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

# Get information about the stimulus types to identify optogenetic protocols
seq_df = nwb.icephys_sequential_recordings.to_dataframe()
print("\nAvailable stimulus types:")
for i, stim_type in enumerate(seq_df['stimulus_type'].unique()):
    print(f"  {i+1}. {stim_type}")

# Look for stimulus types involving optogenetic stimulation
opto_stim_types = [stim for stim in seq_df['stimulus_type'].unique() 
                  if 'laser' in stim.lower() or 'light' in stim.lower()]

if opto_stim_types:
    print(f"\nFound {len(opto_stim_types)} types of optogenetic stimulation:")
    for i, stim in enumerate(opto_stim_types):
        print(f"  {i+1}. {stim}")
    
    # Get sequential recordings for these stimulus types
    opto_recordings = seq_df[seq_df['stimulus_type'].isin(opto_stim_types)]
    print(f"\nFound {len(opto_recordings)} sequential recordings with optogenetic stimulation")
    
    # Access a specific optogenetic recording
    if not opto_recordings.empty:
        # Take the first optogenetic recording
        rec_id = opto_recordings.index[0]
        print(f"\nExamining recording with ID {rec_id} and stimulus type: {opto_recordings.iloc[0]['stimulus_type']}")
        
        # Get simultaneous recordings
        sim_rec_ids = opto_recordings.loc[rec_id, 'simultaneous_recordings']
        if isinstance(sim_rec_ids, list):
            print(f"Number of simultaneous recordings: {len(sim_rec_ids)}")
            
            if len(sim_rec_ids) > 0:
                # Get recording data from a simultaneous recording
                sim_rec_df = nwb.icephys_simultaneous_recordings.to_dataframe()
                rec_df = nwb.intracellular_recordings.to_dataframe()
                
                # Find recording series in this simultaneous recording
                recordings = []
                
                # This structure can be complex, try different approaches
                try:
                    # Get indices of intracellular recordings
                    sim_rec_df_idx = nwb.icephys_simultaneous_recordings.id.get(sim_rec_ids[0])
                    
                    if sim_rec_df_idx is not None:
                        # Get recordings for this simultaneous recording
                        recording_idx = nwb.icephys_simultaneous_recordings.recordings.get(sim_rec_df_idx)
                        
                        if recording_idx is not None and len(recording_idx) > 0:
                            # Iterate through recordings to find response/stimulus pairs
                            for rec_idx in recording_idx:
                                try:
                                    # Get the recording data
                                    stim_series = rec_df.loc[rec_idx, 'stimulus']
                                    response_series = rec_df.loc[rec_idx, 'response']
                                    electrode_idx = rec_df.loc[rec_idx, 'electrode']
                                    
                                    # Map to acquisition/stimulus series
                                    recordings.append({
                                        'stimulus': stim_series,
                                        'response': response_series,
                                        'electrode': electrode_idx
                                    })
                                except Exception as e:
                                    print(f"Error accessing recording {rec_idx}: {e}")
                            
                            print(f"Successfully retrieved {len(recordings)} recordings")
                        else:
                            print("No recording indices found in simultaneous recordings.")
                    else:
                        print(f"No data found for simultaneous recording {sim_rec_ids[0]}")
                except Exception as e:
                    print(f"Error accessing simultaneous recording data: {e}")
                
                # Direct approach: find all series with laser/light in description
                print("\nSearching for optogenetic recordings directly in acquisition data...")
                opto_responses = []
                for series_name, series in nwb.acquisition.items():
                    desc = series.description.lower()
                    if 'laser' in desc or 'light' in desc:
                        opto_responses.append(series_name)
                
                print(f"Found {len(opto_responses)} acquisition series with optogenetic descriptions")
                
                if len(opto_responses) > 0:
                    # Analyze first optogenetic response
                    response_name = opto_responses[0]
                    response = nwb.acquisition[response_name]
                    
                    print(f"\nAnalyzing optogenetic response: {response_name}")
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
                    plt.savefig('explore/optogenetic_response.png')
                    plt.close()
                    
                    # Look for the corresponding stimulus if available
                    stim_name = response_name.replace('response', 'stimulus')
                    stim_name = stim_name.replace('current_clamp-', '')
                    
                    if stim_name in nwb.stimulus:
                        stimulus = nwb.stimulus[stim_name]
                        stim_data = stimulus.data[:] * stimulus.conversion
                        
                        # Plot the stimulus
                        plt.figure(figsize=(12, 5))
                        plt.plot(time, stim_data*1e12)  # Convert to pA for visualization
                        plt.xlabel('Time (s)')
                        plt.ylabel('Current (pA)')
                        plt.title(f'Optogenetic Stimulus: {stim_name}')
                        plt.grid(True)
                        plt.savefig('explore/optogenetic_stimulus.png')
                        plt.close()
                        
                        # Plot both stimulus and response together
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
                        
                        plt.title('Optogenetic Stimulus and Response')
                        fig.tight_layout()
                        plt.savefig('explore/optogenetic_stimulus_response.png')
                        plt.close()
                        
                        # Try to identify laser pulse times
                        # For optogenetic data, we need to identify where pulses occur
                        # This is a simple approach - adjust thresholds as needed
                        pulse_threshold = np.max(stim_data) * 0.5
                        pulse_indices = np.where(np.diff((stim_data > pulse_threshold).astype(int)) > 0)[0]
                        
                        if len(pulse_indices) > 0:
                            print(f"\nIdentified {len(pulse_indices)} potential laser pulses")
                            
                            # Plot the first laser pulse and response
                            if len(pulse_indices) > 0:
                                # Focus on the first pulse
                                first_pulse_idx = pulse_indices[0]
                                
                                # Window around the pulse (ms)
                                window_ms = 200  # Larger window to see response
                                samples_per_ms = response.rate / 1000
                                window_samples = int(window_ms * samples_per_ms)
                                
                                pulse_start = max(0, first_pulse_idx - int(window_samples * 0.2))  # 20% before pulse
                                pulse_end = min(len(time), first_pulse_idx + int(window_samples * 0.8))  # 80% after pulse
                                
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
                                
                                plt.title('Single Optogenetic Pulse and Response')
                                fig.tight_layout()
                                plt.savefig('explore/single_optogenetic_pulse.png')
                                plt.close()
                        else:
                            print("No clear laser pulses identified in the stimulus data")
                    else:
                        print(f"Corresponding stimulus {stim_name} not found")
                else:
                    print("No optogenetic responses found in acquisition data")
        else:
            print("No simultaneous recording IDs found")
else:
    print("No optogenetic stimulus types found")

print("\nExploration complete!")