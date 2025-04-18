"""
This script focuses on extracting and analyzing action potential properties
from the intracellular electrophysiology data in the Dandiset.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
import scipy.signal as signal

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
    
    # Get a strong stimulus response that should contain action potentials
    # Higher index should have stronger stimulus (from -120pA to 260pA in 20pA increments)
    response_name = "current_clamp-response-19-ch-0"  # Try one with higher current
    stimulus_name = "stimulus-19-ch-0"
    
    if response_name in nwb.acquisition:
        response = nwb.acquisition[response_name]
        print(f"\nFound {response_name}")
        print(f"Description: {response.description}")
        
        # Get stimulus info
        if stimulus_name in nwb.stimulus:
            stimulus = nwb.stimulus[stimulus_name]
            current_pA = np.mean(stimulus.data[5000:25000]) * stimulus.conversion * 1e12  # Convert to pA
            print(f"Stimulus current: {current_pA:.2f} pA")
        
        # Load response data - full trace
        data = response.data[:]  # Load all data
        
        # Convert to actual voltage
        voltage_data = data * response.conversion
        
        # Create a time vector (in seconds)
        time = np.arange(len(voltage_data)) / response.rate
        
        # Find action potentials (voltage spikes)
        # We'll use a simple threshold approach
        threshold = -0.02  # 20 mV above resting, typical spike threshold
        resting_potential = np.mean(voltage_data[0:1000])  # Estimate resting from beginning
        print(f"Estimated resting potential: {resting_potential:.3f} V")
        
        # Detect crossings
        above_threshold = voltage_data > threshold
        crossings = np.where(np.diff(above_threshold.astype(int)) > 0)[0]
        
        if len(crossings) > 0:
            print(f"Found {len(crossings)} action potentials")
            
            # Plot the entire trace with detected spikes
            plt.figure(figsize=(12, 6))
            plt.plot(time, voltage_data, 'b')
            plt.plot(time[crossings], voltage_data[crossings], 'ro')
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold} V)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Voltage (V)')
            plt.title(f'Action Potentials in Response to Current Injection ({current_pA:.0f} pA)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('action_potentials_trace.png')
            print("Full trace with APs saved to 'action_potentials_trace.png'")
            
            # Extract a single action potential to analyze its properties
            if len(crossings) > 0:
                # Take the first spike
                spike_index = crossings[0]
                
                # Extract a window around the spike (2ms before, 5ms after)
                pre_spike_pts = int(0.002 * response.rate)  # 2ms before
                post_spike_pts = int(0.005 * response.rate)  # 5ms after
                
                # Ensure we have enough data points
                if spike_index > pre_spike_pts and spike_index + post_spike_pts < len(voltage_data):
                    spike_window = voltage_data[spike_index - pre_spike_pts:spike_index + post_spike_pts]
                    spike_time = time[spike_index - pre_spike_pts:spike_index + post_spike_pts]
                    
                    # Calculate AP properties
                    ap_amplitude = np.max(spike_window) - resting_potential
                    
                    # Find the peak and its index
                    peak_index = np.argmax(spike_window)
                    
                    # Calculate the half-width
                    # First, find the half-amplitude value
                    half_amplitude = resting_potential + ap_amplitude / 2
                    
                    # Find the crossing points
                    above_half = spike_window > half_amplitude
                    half_crossings = np.where(np.diff(above_half.astype(int)) != 0)[0]
                    
                    # Calculate half-width if we have two crossing points
                    if len(half_crossings) >= 2:
                        half_width = (spike_time[half_crossings[1]] - spike_time[half_crossings[0]]) * 1000  # in ms
                    else:
                        half_width = None
                        
                    # Calculate spike maximum upstroke (max dV/dt during rising phase)
                    dv_dt = np.diff(spike_window) * response.rate  # V/s
                    
                    # Use points before the peak for upstroke
                    max_upstroke = np.max(dv_dt[:peak_index])
                    
                    print(f"\nAction Potential Properties:")
                    print(f"Amplitude: {ap_amplitude*1000:.2f} mV")
                    if half_width:
                        print(f"Half-width: {half_width:.3f} ms")
                    else:
                        print("Half-width: Could not be calculated")
                    print(f"Maximum upstroke: {max_upstroke:.2f} V/s")
                    
                    # Plot the single AP with its properties
                    plt.figure(figsize=(8, 6))
                    plt.plot(spike_time * 1000, spike_window * 1000, 'b')  # Convert to ms and mV
                    
                    # Mark the peak
                    plt.plot(spike_time[peak_index] * 1000, spike_window[peak_index] * 1000, 'ro')
                    
                    # Mark the half-width if available
                    if half_width:
                        plt.plot([spike_time[half_crossings[0]] * 1000, spike_time[half_crossings[1]] * 1000], 
                                [half_amplitude * 1000, half_amplitude * 1000], 'g-', linewidth=2)
                        plt.annotate(f'Half-width: {half_width:.2f} ms', 
                                    xy=(spike_time[half_crossings[0]] * 1000, half_amplitude * 1000),
                                    xytext=(spike_time[half_crossings[0]] * 1000 - 0.2, half_amplitude * 1000 - 10),
                                    arrowprops=dict(arrowstyle='->'))
                    
                    plt.xlabel('Time (ms)')
                    plt.ylabel('Voltage (mV)')
                    plt.title('Single Action Potential Waveform Analysis')
                    plt.grid(True, alpha=0.3)
                    plt.savefig('action_potential_waveform.png')
                    print("Single AP waveform saved to 'action_potential_waveform.png'")
                    
            # Now create a frequency-current (F-I) curve
            # Let's find several response traces with different stimulus intensities
            response_series = []
            stimulus_intensities = []
            
            for i in range(2, 30):  # Look at a range of stimulus intensities
                response_key = f"current_clamp-response-{i:02d}-ch-0"
                stimulus_key = f"stimulus-{i:02d}-ch-0"
                
                if response_key in nwb.acquisition and stimulus_key in nwb.stimulus:
                    response_data = nwb.acquisition[response_key].data[:] * nwb.acquisition[response_key].conversion
                    stimulus_data = nwb.stimulus[stimulus_key].data[:] * nwb.stimulus[stimulus_key].conversion
                    
                    # Calculate mean stimulus current in pA
                    current = np.mean(stimulus_data[5000:25000]) * 1e12  # Convert to pA
                    
                    # Count spikes (threshold crossing)
                    above_threshold = response_data > threshold
                    spike_count = np.sum(np.diff(above_threshold.astype(int)) > 0)
                    
                    # Calculate firing frequency in Hz (spikes per second)
                    # Assuming 1s of stimulation from 0.5s to 1.5s
                    frequency = spike_count  # Since we're looking at a 1-second window
                    
                    response_series.append(frequency)
                    stimulus_intensities.append(current)
            
            if response_series:
                # Sort by stimulus intensity
                sorted_indices = np.argsort(stimulus_intensities)
                sorted_intensities = [stimulus_intensities[i] for i in sorted_indices]
                sorted_frequencies = [response_series[i] for i in sorted_indices]
                
                # Plot F-I curve
                plt.figure(figsize=(8, 6))
                plt.plot(sorted_intensities, sorted_frequencies, 'b-o')
                plt.xlabel('Injected Current (pA)')
                plt.ylabel('Firing Frequency (Hz)')
                plt.title('Frequency-Current (F-I) Curve')
                plt.grid(True, alpha=0.3)
                plt.savefig('fi_curve.png')
                print("F-I curve saved to 'fi_curve.png'")
                
        else:
            print("No action potentials detected with current threshold")
        
    else:
        print(f"Could not find response {response_name}")

except Exception as e:
    print(f"Error: {e}")
    
print("Script completed.")