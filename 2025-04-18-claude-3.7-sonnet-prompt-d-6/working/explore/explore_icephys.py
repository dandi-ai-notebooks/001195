"""
This script explores intracellular electrophysiology data from a sample NWB file in the dataset.
It focuses on loading the data and visualizing the current clamp responses to different stimuli.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# Save figures to files instead of displaying them (to avoid hanging)
plt.ioff()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/9029e9f3-949a-4cf3-ade5-48542a1a59b7/download/"
print(f"Loading NWB file from URL: {url}")

try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    print("NWB file loaded successfully!")
    print(f"Session description: {nwb.session_description}")
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Experiment date: {nwb.session_start_time}")
    
    # Get one of the current clamp response series
    response_name = "current_clamp-response-02-ch-0"
    stimulus_name = "stimulus-02-ch-0"
    
    if response_name in nwb.acquisition:
        response = nwb.acquisition[response_name]
        print(f"\nFound {response_name}")
        print(f"Description: {response.description}")
        print(f"Starting time: {response.starting_time} {response.starting_time_unit}")
        print(f"Rate: {response.rate} Hz")
        print(f"Data shape: {response.data.shape}")
        
        # Load a subset of data
        data_subset = response.data[0:10000]  # First 1 second of data (at 10kHz sampling rate)
        
        # Convert to actual voltage
        voltage_data = data_subset * response.conversion
        
        # Load corresponding stimulus
        if stimulus_name in nwb.stimulus:
            stimulus = nwb.stimulus[stimulus_name]
            print(f"\nFound {stimulus_name}")
            print(f"Description: {stimulus.description}")
            
            # Load stimulus data
            stim_data_subset = stimulus.data[0:10000]  # First 1 second, matching the response
            
            # Convert to actual current
            current_data = stim_data_subset * stimulus.conversion  # In amperes
            current_data_pA = current_data * 1e12  # Convert to pA for easier reading
            
            # Create a time vector (in seconds)
            time = np.arange(len(voltage_data)) / response.rate
            
            # Plot the response and stimulus
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot voltage response
            axs[0].plot(time, voltage_data, 'b')
            axs[0].set_ylabel('Voltage (V)')
            axs[0].set_title(f'Voltage Response - {response.description}')
            
            # Plot current stimulus
            axs[1].plot(time, current_data_pA, 'r')
            axs[1].set_xlabel('Time (seconds)')
            axs[1].set_ylabel('Current (pA)')
            axs[1].set_title(f'Current Stimulus - {stimulus.description}')
            
            plt.tight_layout()
            plt.savefig('icephys_response.png')
            print("Figure saved to 'icephys_response.png'")
            
            # Now create an I-V curve
            # Let's find several response traces with different stimulus intensities
            response_series = []
            stimulus_series = []
            
            for i in range(2, 10):  # Look at a range of stimulus intensities
                response_key = f"current_clamp-response-0{i}-ch-0"
                stimulus_key = f"stimulus-0{i}-ch-0"
                
                if response_key in nwb.acquisition and stimulus_key in nwb.stimulus:
                    response_series.append(nwb.acquisition[response_key])
                    stimulus_series.append(nwb.stimulus[stimulus_key])
            
            if response_series:
                print(f"\nFound {len(response_series)} stimulus-response pairs for I-V analysis")
                
                # Extract steady-state responses and stimulus intensities
                steady_state_voltages = []
                stimulus_intensities = []
                
                for resp, stim in zip(response_series, stimulus_series):
                    # Get the middle of the trace (avoiding transients)
                    resp_data = resp.data[5000:9000]  # From 0.5s to 0.9s (assuming 1s stimulation)
                    stim_data = stim.data[5000:9000]  # Same time range
                    
                    # Convert to physical units
                    voltage = np.mean(resp_data) * resp.conversion
                    current = np.mean(stim_data) * stim.conversion * 1e12  # pA
                    
                    steady_state_voltages.append(voltage)
                    stimulus_intensities.append(current)
                
                # Plot I-V relationship
                plt.figure(figsize=(8, 6))
                plt.scatter(stimulus_intensities, steady_state_voltages, c='b', marker='o')
                plt.plot(stimulus_intensities, steady_state_voltages, 'b-', alpha=0.5)
                plt.xlabel('Injected Current (pA)')
                plt.ylabel('Membrane Potential (V)')
                plt.title('I-V Relationship')
                plt.grid(True, alpha=0.3)
                plt.savefig('icephys_iv_curve.png')
                print("I-V curve saved to 'icephys_iv_curve.png'")
            
        else:
            print(f"Could not find stimulus {stimulus_name}")
    else:
        print(f"Could not find response {response_name}")

except Exception as e:
    print(f"Error: {e}")
    
print("Script completed.")