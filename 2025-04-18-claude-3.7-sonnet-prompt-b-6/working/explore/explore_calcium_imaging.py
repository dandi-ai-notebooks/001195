"""
This script explores the calcium imaging data from the Dandiset.
We'll examine the ROIs (regions of interest) and their fluorescence activity
from the one-photon calcium imaging dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# URL for the NWB file with calcium imaging data
url = "https://api.dandiarchive.org/api/assets/7ac3357c-06e4-4f04-a869-2f3a71524456/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print(f"Session: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject description: {nwb.subject.description}")
print(f"Experiment description: {nwb.experiment_description}")

# Get information about the imaging session
image_series = nwb.acquisition["OnePhotonSeries"]
print(f"\nFrame rate: {image_series.rate} Hz")
print(f"Image dimensions: {image_series.dimension[:]}")
print(f"Number of frames: {image_series.data.shape[0]}")
print(f"Imaging location: {image_series.imaging_plane.location}")
print(f"Indicator: {image_series.imaging_plane.indicator}")

# Get the processed fluorescence data
fluorescence = nwb.processing["ophys"].data_interfaces["Fluorescence"]
roi_response_series = fluorescence.roi_response_series["RoiResponseSeries"]
print(f"\nNumber of ROIs: {roi_response_series.data.shape[1]}")

# Let's get a summary image (first frame)
print("Loading a sample frame from the imaging dataset...")
try:
    sample_frame = image_series.data[0, :, :]
    plt.figure(figsize=(8, 6))
    plt.imshow(sample_frame, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title('Sample Frame from Calcium Imaging')
    plt.tight_layout()
    plt.savefig('calcium_sample_frame.png')
    plt.close()
    print("Sample frame saved.")
except Exception as e:
    print(f"Error loading sample frame: {e}")

# For the ROI masks, we need to construct a full image
print("Constructing ROI masks...")
try:
    # Get ROI masks
    ps = nwb.processing["ophys"].data_interfaces["PlaneSegmentation"]
    
    # Creating a blank image for all ROIs combined
    img_shape = tuple(image_series.dimension[:])
    all_rois = np.zeros(img_shape)
    
    # Loop through the ROIs and build mask
    num_rois = ps.id.data.shape[0]
    for i in range(min(num_rois, 30)):  # Limit to first 30 ROIs to avoid memory issues
        mask_idx = ps.pixel_mask.data[:][ps.pixel_mask.data[:]['roi_idx'] == i]
        for x, y, weight in mask_idx[['x', 'y', 'weight']]:
            x, y = int(x), int(y)
            if 0 <= x < img_shape[0] and 0 <= y < img_shape[1]:
                all_rois[x, y] = max(all_rois[x, y], weight)
    
    # Plot all ROIs
    plt.figure(figsize=(10, 8))
    plt.imshow(all_rois, cmap='hot')
    plt.colorbar(label='Weight')
    plt.title('Combined ROI Masks')
    plt.tight_layout()
    plt.savefig('calcium_roi_masks.png')
    plt.close()
    print("ROI masks image saved.")
except Exception as e:
    print(f"Error creating ROI masks: {e}")

# Plot some fluorescence traces
try:
    # Get the fluorescence data
    F = roi_response_series.data[:, :]
    
    # Create time vector
    t = np.arange(F.shape[0]) / roi_response_series.rate  # in seconds
    
    # Select 5 random ROIs to plot
    num_rois = F.shape[1]
    sample_rois = np.random.choice(np.arange(num_rois), size=min(5, num_rois), replace=False)
    
    plt.figure(figsize=(12, 8))
    for i, roi_idx in enumerate(sample_rois):
        # Normalize the trace for better visualization
        trace = F[:, roi_idx]
        trace_norm = (trace - np.min(trace)) / (np.max(trace) - np.min(trace) + 1e-6)
        plt.plot(t, trace_norm + i, label=f'ROI {roi_idx}')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Fluorescence + offset')
    plt.title('Sample Fluorescence Traces')
    plt.legend()
    plt.tight_layout()
    plt.savefig('calcium_fluorescence_traces.png')
    plt.close()
    print("Fluorescence traces plot saved.")
    
    # Plot the average activity over time
    mean_activity = np.mean(F, axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(t, mean_activity)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Fluorescence')
    plt.title('Average Calcium Activity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('calcium_mean_activity.png')
    plt.close()
    print("Mean activity plot saved.")
    
    # Create a heatmap of all ROI activity
    # Downsample in time to make the plot manageable
    downsample = max(1, int(F.shape[0] / 1000))  # Limit to ~1000 time points
    F_ds = F[::downsample, :]
    t_ds = t[::downsample]
    
    # Plot heatmap of activity
    plt.figure(figsize=(12, 8))
    plt.imshow(F_ds.T, aspect='auto', origin='lower', 
               extent=[t_ds[0], t_ds[-1], 0, F.shape[1]])
    plt.colorbar(label='Fluorescence')
    plt.xlabel('Time (s)')
    plt.ylabel('ROI Number')
    plt.title('Calcium Activity of All ROIs')
    plt.tight_layout()
    plt.savefig('calcium_activity_heatmap.png')
    plt.close()
    print("Activity heatmap saved.")
    
except Exception as e:
    print(f"Error plotting fluorescence data: {e}")

print("Analysis completed. Plots saved to the explore directory.")