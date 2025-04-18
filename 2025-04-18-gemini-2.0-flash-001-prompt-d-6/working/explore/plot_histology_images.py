import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get histology images
images = nwb.acquisition["histology_images"]

# Plot the first image
if images.images:
  for img_name, img_data in images.images.items():
    print(f"Plotting image: {img_name}")
    plt.imshow(img_data.data[:], cmap='gray') # Display the first channel if it's a multi-channel image
    plt.title(img_name)
    plt.savefig(f"explore/{img_name}.png")
    plt.close()
    break
else:
  print("No images found in histology_images")