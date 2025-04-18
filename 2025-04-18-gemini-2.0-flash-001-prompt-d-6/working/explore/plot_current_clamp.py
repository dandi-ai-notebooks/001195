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

# Plot current clamp data
data = nwb.acquisition["current_clamp-response-01-ch-0"].data[:1000]
plt.plot(data)
plt.xlabel("Time (samples)")
plt.ylabel("Voltage (mV)")
plt.title("Current Clamp Response 01 ch 0")
plt.savefig("explore/current_clamp_01_ch_0.png")
plt.close()