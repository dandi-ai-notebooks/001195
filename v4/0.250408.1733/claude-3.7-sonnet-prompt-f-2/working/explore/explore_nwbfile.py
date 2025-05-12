"""
This script explores the structure and content of an NWB file from the Dandiset.
The goal is to understand the data organization and available data types to inform our notebook creation.
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
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print("NWB File Information:")
print(f"  Session description: {nwb.session_description}")
print(f"  Identifier: {nwb.identifier}")
print(f"  Session start time: {nwb.session_start_time}")
print(f"  Experimenter: {nwb.experimenter}")
print(f"  Lab: {nwb.lab}")
print(f"  Institution: {nwb.institution}")

# Print subject information
print("\nSubject Information:")
print(f"  Subject ID: {nwb.subject.subject_id}")
print(f"  Species: {nwb.subject.species}")
print(f"  Age: {nwb.subject.age}")
print(f"  Sex: {nwb.subject.sex}")
print(f"  Genotype: {nwb.subject.genotype}")
print(f"  Description: {nwb.subject.description}")

# List available electrodes
print("\nAvailable electrodes:")
for electrode_id, electrode in nwb.icephys_electrodes.items():
    print(f"  {electrode_id}: {electrode}")

# List available acquisition data
print("\nAcquisition data:")
acquisition_count = 0
for series_name, series in nwb.acquisition.items():
    acquisition_count += 1
    if acquisition_count <= 5:  # Only print the first 5 to avoid too much output
        print(f"  {series_name}: {type(series).__name__}")
        print(f"    Description: {series.description}")
        print(f"    Data shape: {series.data.shape}")
        print(f"    Unit: {series.unit}")
        print(f"    Starting time: {series.starting_time} {series.starting_time_unit}")
        print(f"    Rate: {series.rate} Hz")
print(f"Total acquisition series: {acquisition_count}")

# List available stimulus data
print("\nStimulus data:")
stimulus_count = 0
for series_name, series in nwb.stimulus.items():
    stimulus_count += 1
    if stimulus_count <= 5:  # Only print the first 5 to avoid too much output
        print(f"  {series_name}: {type(series).__name__}")
        print(f"    Description: {series.description}")
        print(f"    Data shape: {series.data.shape}")
        print(f"    Unit: {series.unit}")
        print(f"    Starting time: {series.starting_time} {series.starting_time_unit}")
        print(f"    Rate: {series.rate} Hz")
print(f"Total stimulus series: {stimulus_count}")

# Generate a plot of example current-clamp response and corresponding stimulus
# Selecting one pair of response-stimulus for visualization
response_key = list(nwb.acquisition.keys())[0]
response = nwb.acquisition[response_key]
stim_key = response_key.replace('response', 'stimulus')
stim_key = stim_key.replace('current_clamp-', '')
stimulus = nwb.stimulus[stim_key]

# Create time array based on starting time and sampling rate
time = np.arange(response.data.shape[0]) / response.rate + response.starting_time

# Generate a figure with two subplots
plt.figure(figsize=(12, 8))

# Plot the response data
plt.subplot(2, 1, 1)
# Convert raw data to appropriate units using the conversion factor
response_data = response.data[:] * response.conversion
plt.plot(time, response_data)
plt.title(f'Response: {response_key}')
plt.ylabel(f'Voltage ({response.unit})')
plt.grid(True)

# Plot the stimulus data
plt.subplot(2, 1, 2)
# Convert raw data to appropriate units using the conversion factor
stim_data = stimulus.data[:] * stimulus.conversion
plt.plot(time, stim_data)
plt.title(f'Stimulus: {stim_key}')
plt.xlabel(f'Time ({response.starting_time_unit})')
plt.ylabel(f'Current ({stimulus.unit})')
plt.grid(True)

plt.tight_layout()
plt.savefig('explore/example_response_stimulus.png')
plt.close()

# Print information about the sequential recordings table
print("\nSequential Recordings Table:")
seq_df = nwb.icephys_sequential_recordings.to_dataframe()
print(seq_df.head())
print(f"Number of sequential recordings: {len(seq_df)}")
print(f"Columns: {seq_df.columns.tolist()}")
print(f"Stimulus types: {seq_df['stimulus_type'].unique().tolist()}")

# Explore a specific recording series in more detail - looking for current-step protocols
print("\nExploring specific recording protocols:")
# Find recordings with step protocols (based on description)
step_protocols = []
for series_name, series in nwb.acquisition.items():
    if 'step' in series.description.lower():
        step_protocols.append(series_name)
        
print(f"Found {len(step_protocols)} recordings with step protocols")
if step_protocols:
    print(f"Example step protocol: {step_protocols[0]}")
    step_series = nwb.acquisition[step_protocols[0]]
    print(f"  Description: {step_series.description}")
    print(f"  Data shape: {step_series.data.shape}")
    
    # Plot this step protocol response
    time = np.arange(step_series.data.shape[0]) / step_series.rate + step_series.starting_time
    plt.figure(figsize=(12, 6))
    step_data = step_series.data[:] * step_series.conversion
    plt.plot(time, step_data)
    plt.title(f'Step Protocol Response: {step_protocols[0]}')
    plt.xlabel(f'Time ({step_series.starting_time_unit})')
    plt.ylabel(f'Voltage ({step_series.unit})')
    plt.grid(True)
    plt.savefig('explore/step_protocol_response.png')
    plt.close()
    
    # Find the corresponding stimulus
    stim_key = step_protocols[0].replace('response', 'stimulus')
    stim_key = stim_key.replace('current_clamp-', '')
    if stim_key in nwb.stimulus:
        step_stim = nwb.stimulus[stim_key]
        stim_data = step_stim.data[:] * step_stim.conversion
        
        # Plot the stimulus
        plt.figure(figsize=(12, 6))
        plt.plot(time, stim_data)
        plt.title(f'Step Protocol Stimulus: {stim_key}')
        plt.xlabel(f'Time ({step_stim.starting_time_unit})')
        plt.ylabel(f'Current ({step_stim.unit})')
        plt.grid(True)
        plt.savefig('explore/step_protocol_stimulus.png')
        plt.close()