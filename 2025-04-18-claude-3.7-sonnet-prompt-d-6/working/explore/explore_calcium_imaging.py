"""
This script explores calcium imaging data from a sample NWB file in the dataset.
It focuses on loading the data, visualizing fluorescence signals, and examining ROIs.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# Save figures to files instead of displaying them (to avoid hanging)
plt.ioff()

# Load NWB file with calcium imaging data
url = "https://api.dandiarchive.org/api/assets/7ac3357c-06e4-4f04-a869-2f3a71524456/download/"
print(f"Loading calcium imaging NWB file from URL: {url}")

try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    print("NWB file loaded successfully!")
    print(f"Session description: {nwb.session_description}")
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Experiment date: {nwb.session_start_time}")
    
    # Access the ophys module
    if 'ophys' in nwb.processing:
        ophys = nwb.processing['ophys']
        print("\nFound optical physiology (ophys) processing module")
        print(f"Description: {ophys.description}")
        
        # Check if we have fluorescence data
        if 'Fluorescence' in ophys.data_interfaces:
            fluorescence = ophys.data_interfaces['Fluorescence']
            print("\nFound fluorescence data")
            
            if 'RoiResponseSeries' in fluorescence.roi_response_series:
                roi_response = fluorescence.roi_response_series['RoiResponseSeries']
                print(f"ROI response data shape: {roi_response.data.shape}")
                print(f"Number of ROIs: {roi_response.data.shape[1]}")
                print(f"Number of time points: {roi_response.data.shape[0]}")
                print(f"Sampling rate: {roi_response.rate} Hz")
                
                # Plot fluorescence responses for a few ROIs
                # Get the first few minutes of data for a few ROIs
                num_rois_to_plot = 5
                # Calculate how many time points for 2 minutes at the given sampling rate
                two_min_points = int(120 * roi_response.rate)
                # Limit data to first 2 minutes and first few ROIs
                data_subset = roi_response.data[0:two_min_points, 0:num_rois_to_plot]
                
                # Create a time vector (in seconds)
                time = np.arange(data_subset.shape[0]) / roi_response.rate
                
                # Plot the fluorescence traces
                plt.figure(figsize=(12, 8))
                for i in range(num_rois_to_plot):
                    plt.plot(time, data_subset[:, i], label=f'ROI {i+1}')
                
                plt.xlabel('Time (seconds)')
                plt.ylabel('Fluorescence (a.u.)')
                plt.title('Fluorescence Traces for Selected ROIs')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig('calcium_fluorescence_traces.png')
                print("Fluorescence traces saved to 'calcium_fluorescence_traces.png'")
                
                # Get plane segmentation to plot ROIs
                if 'PlaneSegmentation' in ophys.data_interfaces:
                    plane_seg = ophys.data_interfaces['PlaneSegmentation']
                    print("\nFound plane segmentation data")
                    print(f"Number of ROIs in segmentation: {len(plane_seg.id.data)}")
                    
                    # Create a composite image showing ROI masks
                    if 'OnePhotonSeries' in nwb.acquisition:
                        onephoton_series = nwb.acquisition['OnePhotonSeries']
                        print("\nFound one-photon series data")
                        print(f"Image dimensions: {onephoton_series.dimension[:]}")
                        
                        # Get a single image frame for background
                        # Use the first frame
                        try:
                            frame = onephoton_series.data[0]
                            
                            # Create a mask image (initialize with zeros)
                            mask_image = np.zeros(frame.shape)
                            
                            # Add all ROI masks to create a composite picture
                            for i in range(min(10, len(plane_seg.id.data))):  # Limit to 10 ROIs to avoid overloading
                                # Get pixel mask for this ROI
                                # Access pixel mask correctly - using the index method instead of get
                                try:
                                    roi_pixels = plane_seg.pixel_mask[i]  # Access using index
                                    for x, y, w in roi_pixels:
                                        x, y = int(x), int(y)
                                        # Check bounds
                                        if 0 <= x < mask_image.shape[0] and 0 <= y < mask_image.shape[1]:
                                            mask_image[x, y] = 1  # Mark as ROI
                                except Exception as pixel_error:
                                    print(f"Could not access ROI {i} pixel mask: {pixel_error}")
                            
                            # Plot the composite ROI mask
                            plt.figure(figsize=(10, 8))
                            plt.imshow(frame, cmap='gray')
                            plt.imshow(mask_image, alpha=0.5, cmap='hot')
                            plt.title('One-Photon Image with ROI Masks')
                            plt.colorbar(label='ROI Mask Value')
                            plt.savefig('calcium_roi_masks.png')
                            print("ROI masks visualization saved to 'calcium_roi_masks.png'")
                            
                        except Exception as e:
                            print(f"Error creating ROI visualization: {e}")
                
                # Analyze activity for all ROIs using heatmap
                # Use more ROIs
                num_rois_for_heatmap = 20
                # Limit to first 2 minutes
                data_for_heatmap = roi_response.data[0:two_min_points, 0:num_rois_for_heatmap]
                
                plt.figure(figsize=(12, 8))
                plt.imshow(data_for_heatmap.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Fluorescence (a.u.)')
                plt.xlabel('Time (samples)')
                plt.ylabel('ROI #')
                plt.title('Calcium Activity Heatmap')
                
                # Add time ticks
                time_ticks = np.linspace(0, data_for_heatmap.shape[0], 5)
                time_labels = [f"{t/roi_response.rate:.0f}s" for t in time_ticks]
                plt.xticks(time_ticks, time_labels)
                
                plt.savefig('calcium_activity_heatmap.png')
                print("Activity heatmap saved to 'calcium_activity_heatmap.png'")
                
    else:
        print("No ophys processing module found in this file")

except Exception as e:
    print(f"Error: {e}")
    
print("Script completed.")