"""
This script explores the intracellular electrophysiology (icephys) data from the Dandiset.
We'll examine the current clamp recordings and plot a few response traces to current injections.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print(f"Experiment: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject description: {nwb.subject.description}")

# Let's look at some current clamp recordings
responses = []
stimulus = []
times = []

# Select a few responses to different stimulus intensities
for i in [4, 10, 16]:  # Different stimulus intensities
    key = f"current_clamp-response-{i:02d}-ch-0"
    stim_key = f"stimulus-{i:02d}-ch-0"
    
    if key in nwb.acquisition and stim_key in nwb.stimulus:
        # Get the response data
        response = nwb.acquisition[key].data[:]
        conversion = nwb.acquisition[key].conversion  # V to mV conversion
        response_mV = response * conversion * 1000  # convert to mV
        
        # Get the stimulus data (in pA)
        stim = nwb.stimulus[stim_key].data[:] * 1e12  # convert to pA
        
        # Create time vector based on sampling rate
        rate = nwb.acquisition[key].rate
        t = np.arange(len(response)) / rate
        
        responses.append(response_mV)
        stimulus.append(stim)
        times.append(t)
        
        # Print stimulus characteristics
        mean_stim = np.mean(stim[int(rate*0.2):int(rate*0.8)])  # Measure during the steady part
        print(f"Stimulus {i}: Mean current = {mean_stim:.2f} pA")

# Plot the responses
plt.figure(figsize=(12, 8))

# Plot membrane potential responses
plt.subplot(2, 1, 1)
for i, (t, resp) in enumerate(zip(times, responses)):
    plt.plot(t, resp, label=f"Stimulus {i+1}")
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential Responses to Different Current Injections')
plt.legend()
plt.grid(True)

# Plot stimulus currents
plt.subplot(2, 1, 2)
for i, (t, stim) in enumerate(zip(times, stimulus)):
    plt.plot(t, stim, label=f"Stimulus {i+1}")
plt.xlabel('Time (s)')
plt.ylabel('Current (pA)')
plt.title('Stimulus Currents')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('icephys_responses.png')
plt.close()

# Create a plot with just one example response
if len(responses) > 0:
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times[0], responses[0])
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Example Membrane Potential Response')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(times[0], stimulus[0])
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    plt.title('Example Stimulus Current')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('icephys_single_response.png')
    plt.close()

# Create a plot with action potentials if present
# Let's check if there are any action potentials in the responses
has_spikes = False
for resp in responses:
    # Simple threshold detection for action potentials
    if np.any(resp > 0):  # Typical action potentials exceed 0 mV
        has_spikes = True
        break

if has_spikes:
    plt.figure(figsize=(10, 6))
    for i, (t, resp) in enumerate(zip(times, responses)):
        if np.any(resp > 0):
            plt.plot(t, resp, label=f"Stimulus {i+1}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Action Potentials in Response to Current Injection')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('icephys_action_potentials.png')
    plt.close()

print("Analysis completed. Plots saved to the explore directory.")