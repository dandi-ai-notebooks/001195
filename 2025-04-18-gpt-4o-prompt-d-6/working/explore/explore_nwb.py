# This script explores the current clamp responses in the NWB file and produces plots. It saves plots as PNG files for further examination.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access data and metadata
cc_series = nwb.acquisition["current_clamp-response-01-ch-0"]
data = cc_series.data[:]
timestamps = np.arange(len(data)) / cc_series.rate

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(timestamps, data, label="Current Clamp Response - Channel 0")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Current Clamp Response')
plt.legend()
plt.savefig('explore/cc_response_ch0.png')

# Clean up
io.close()
remote_file.close()