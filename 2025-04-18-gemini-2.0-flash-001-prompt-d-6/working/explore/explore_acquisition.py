"""
This script loads the NWB file and prints some basic information about the acquisition data.
"""

import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the acquisition data
print("Acquisition data:")
for key in nwb.acquisition:
    print(f"- {key}: {type(nwb.acquisition[key])}")
    if hasattr(nwb.acquisition[key], 'data'):
        print(f"    - data shape: {nwb.acquisition[key].data.shape}")
        print(f"    - data dtype: {nwb.acquisition[key].data.dtype}")