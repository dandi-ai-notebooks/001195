# This script loads a sample NWB file from Dandiset 001195 and explores its structure, printing metadata and plotting an example trace.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Prescribed remote file URL
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"

# Load file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Print session and subject info
print("Session description:", nwb.session_description)
print("Session start time:", nwb.session_start_time)
print("Subject ID:", getattr(nwb.subject, "subject_id", None))
print("Subject species:", getattr(nwb.subject, "species", None))
print("Subject strain:", getattr(nwb.subject, "strain", None))
print("Subject sex:", getattr(nwb.subject, "sex", None))
print("Institution:", getattr(nwb, "institution", None))

# List acquisition and stimulus keys
print("Available acquisition series:")
for k in nwb.acquisition.keys():
    print("-", k)
print("Available stimulus series:")
for k in getattr(nwb, "stimulus", {}).keys():
    print("-", k)

# Select example series for plotting
acq_key = "current_clamp-response-01-ch-0"
stim_key = "stimulus-01-ch-0"
if acq_key in nwb.acquisition and stim_key in getattr(nwb, "stimulus", {}):
    acq = nwb.acquisition[acq_key]
    stim = nwb.stimulus[stim_key]
    # Get the data as arrays
    acq_data = acq.data[:]
    stim_data = stim.data[:]
    sample_rate = acq.rate
    t = np.arange(len(acq_data)) / sample_rate

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, acq_data * acq.conversion)
    plt.ylabel(f"Voltage ({acq.unit})")
    plt.title("Current Clamp Response (trace 01, ch 0)")

    plt.subplot(2, 1, 2)
    plt.plot(t, stim_data * stim.conversion)
    plt.ylabel(f"Injected Current ({stim.unit})")
    plt.xlabel("Time (s)")
    plt.title("Stimulus (trace 01, ch 0)")

    plt.tight_layout()
    plt.savefig("explore/cc_example.png")
    print("Saved plot: explore/cc_example.png")
else:
    print("Example series for plotting not found.")