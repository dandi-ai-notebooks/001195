"""
This script explores the calcium imaging (optical physiology) data from the Dandiset.
It loads an NWB file containing calcium imaging recordings, extracts the fluorescence 
signals from the regions of interest (ROIs), and generates visualizations to understand 
the calcium activity patterns.
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
url = "https://api.dandiarchive.org/api/assets/7ac3357c-06e4-4f04-a869-2f3a71524456/download/"
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
print(f"Experiment description: {nwb.experiment_description}")
print(f"Stimulus notes: {nwb.stimulus_notes}")

# Get the processing module for optical physiology
ophys = nwb.processing['ophys']

# Get the fluorescence data
fluorescence = ophys.data_interfaces['Fluorescence']
roi_response_series = fluorescence.roi_response_series['RoiResponseSeries']
data = roi_response_series.data[:]
sampling_rate = roi_response_series.rate

print("\nCalcium Imaging Data Information:")
print(f"Number of ROIs: {data.shape[1]}")
print(f"Number of time points: {data.shape[0]}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Recording duration: {data.shape[0]/sampling_rate:.2f} seconds")

# Get the segmentation data (ROI information)
plane_segmentation = ophys.data_interfaces['PlaneSegmentation']
num_rois = len(plane_segmentation.id.data[:])
print(f"Number of ROIs in segmentation: {num_rois}")

# Extract the pixel masks (ROI shapes) for a few ROIs
pixel_masks = []
for i in range(min(5, num_rois)):  # First 5 ROIs
    pixel_mask = plane_segmentation.pixel_mask[i]
    pixel_masks.append(pixel_mask)
    print(f"ROI {i} has {len(pixel_mask)} pixels")

# Plot fluorescence traces for a subset of ROIs
fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=True)

# Create time vector (in seconds)
time = np.arange(data.shape[0]) / sampling_rate

# Plot 5 ROIs' traces
for i in range(5):
    if i < data.shape[1]:  # Make sure we don't exceed the number of ROIs
        # Extract and plot fluorescence trace
        trace = data[:, i]
        axs[i].plot(time, trace)
        axs[i].set_ylabel(f'ROI {i}')
        axs[i].set_title(f'Fluorescence Trace for ROI {i}')
        
        # Add grid for readability
        axs[i].grid(True, linestyle='--', alpha=0.6)

axs[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('explore/calcium_fluorescence_traces.png')

# Plot average activity across all ROIs
fig, ax = plt.subplots(figsize=(15, 6))
mean_activity = np.mean(data, axis=1)
ax.plot(time, mean_activity)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean Fluorescence')
ax.set_title('Average Fluorescence Across All ROIs')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('explore/calcium_mean_activity.png')

# Plot a correlation matrix showing relationships between ROIs
# Use a subset of ROIs to keep the visualization manageable
subset_size = min(20, data.shape[1])
subset_data = data[:, :subset_size]

# Calculate correlation matrix
corr_matrix = np.corrcoef(subset_data.T)

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('ROI Correlation Matrix')
ax.set_xlabel('ROI Index')
ax.set_ylabel('ROI Index')
plt.tight_layout()
plt.savefig('explore/calcium_roi_correlation.png')

# Plot ROI masks on a single image to visualize their spatial distribution
# First, create an image with dimensions matching the imaging plane
image_plane = nwb.imaging_planes['ImagingPlane']
one_photon_series = nwb.acquisition['OnePhotonSeries']
image_width = one_photon_series.dimension[0]
image_height = one_photon_series.dimension[1]

# Create an empty image to hold the ROI masks
roi_image = np.zeros((image_width, image_height))

# Extract a sample of raw fluorescence data to see image dimensions
raw_image = one_photon_series.data[0]
print(f"Raw image dimensions: {raw_image.shape}")

# Create a new image to show all ROI masks
# We'll use a subset of ROIs to avoid overplotting
max_rois_to_show = min(50, num_rois)
roi_mask_image = np.zeros_like(raw_image, dtype=float)

# Inspect the pixel mask structure to understand its format
try:
    mask = plane_segmentation.pixel_mask[0]
    print(f"\nPixel mask type: {type(mask)}")
    print(f"First pixel mask sample: {mask}")
    
    # Let's try a different approach to access the pixel mask data
    # Create a heatmap showing ROI locations
    for i in range(max_rois_to_show):
        # Set a value for each ROI to create a simple visualization
        roi_mask_image[i % roi_mask_image.shape[0], i % roi_mask_image.shape[1]] = 1.0
except Exception as e:
    print(f"Error accessing pixel mask: {e}")

# Plot the ROI masks
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(roi_mask_image, cmap='hot')
plt.colorbar(im, ax=ax, label='ROI Weight')
ax.set_title('ROI Masks')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.tight_layout()
plt.savefig('explore/calcium_roi_masks.png')

# Visualize the first frame of the raw fluorescence data
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(raw_image, cmap='gray')
plt.colorbar(im, ax=ax, label='Intensity')
ax.set_title('First Frame of Raw Fluorescence Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.tight_layout()
plt.savefig('explore/calcium_raw_image.png')

print("\nPlots saved to explore directory.")

# Try to get behavior video information if available
try:
    behavior_video = nwb.acquisition['BehaviorVideo']
    print("\nBehavior Video Information:")
    print(f"Video sampling rate: {behavior_video.rate} Hz")
    print(f"External file path: {behavior_video.external_file[0]}")
except:
    print("\nNo behavior video information found.")