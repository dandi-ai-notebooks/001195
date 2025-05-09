# Purpose: Explore and summarize the major fields, sweep structure, and channel/series/dataset shapes in the selected NWB file for Dandiset 001195.
# This will inform the notebook about (1) the available CurrentClamp and VoltageClamp series, (2) channel identities, and (3) dataset sizes.
# Output: Prints a summary of acquisitions, stimuli, their descriptions, and main shapes.

import pynwb
import h5py
import remfile
import sys

url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"

print("Opening NWB file:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("Session description:", nwb.session_description)
print("Subject:", getattr(nwb, "subject", None))
print("Lab:", getattr(nwb, "lab", None))
print("Institution:", getattr(nwb, "institution", None))

print("\nDevices:")
for key, device in nwb.devices.items():
    print(f"  - {key}: {getattr(device, 'description', 'n/a')}")

print("\nIntracellular electrodes:")
for key, elec in nwb.icephys_electrodes.items():
    print(f"  - {key}: {getattr(elec, 'description', 'n/a')} (device: {getattr(getattr(elec, 'device', None), 'name', 'n/a')})")

print("\nAcquisition entries (CurrentClamp/VoltageClamp series):")
for key, acq in nwb.acquisition.items():
    t = type(acq).__name__
    if hasattr(acq, 'data'):
        shape = getattr(acq.data, 'shape', None)
        print(f"  - {key} ({t}), shape={shape}, desc=\"{getattr(acq, 'description', None)}\", unit={getattr(acq, 'unit', None)}")
    else:
        print(f"  - {key} ({t}), no 'data' attribute, attrs: {dir(acq)}")

print("\nStimulus entries:")
for key, stim in nwb.stimulus.items():
    t = type(stim).__name__
    shape = getattr(stim.data, 'shape', None)
    print(f"  - {key} ({t}), shape={shape}, desc=\"{getattr(stim, 'description', None)}\", unit={getattr(stim, 'unit', None)}")

# Record datasets present for sweeps/channels, to inform plotting
print("\nAcquisition dataset keys:")
for key in nwb.acquisition:
    print(key)
print("Stimulus dataset keys:")
for key in nwb.stimulus:
    print(key)

print("\nDone.")