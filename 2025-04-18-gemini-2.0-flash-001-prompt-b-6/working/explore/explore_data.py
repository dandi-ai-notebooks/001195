import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Extract the data
data = nwb.acquisition["current_clamp-response-01-ch-0"].data[:]
timestamps = nwb.acquisition["current_clamp-response-01-ch-0"].timestamps
if timestamps is None:
    rate = nwb.acquisition["current_clamp-response-01-ch-0"].rate
    timestamps = np.arange(len(data)) / rate
else:
    timestamps = timestamps[:]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Current Clamp Response")
plt.savefig("explore/current_clamp_response.png")
plt.close()