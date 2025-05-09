# Purpose: Loads and plots example sweeps for CurrentClamp and VoltageClamp series (with corresponding stimulus) from the selected NWB file.
# Plots response and input current/voltage for each mode. Used to judge quality/usability for the notebook.
# All plots saved as PNGs in the explore/ directory.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot an example CurrentClamp response (01-ch-0) and its stimulus (01-ch-0)
cc_resp = nwb.acquisition['current_clamp-response-01-ch-0']
cc_stim = nwb.stimulus['stimulus-01-ch-0']
fs_cc = cc_resp.rate
t_cc = np.arange(cc_resp.data.shape[0]) / fs_cc

plt.figure(figsize=(8, 4))
plt.subplot(2,1,1)
plt.plot(t_cc, cc_resp.data[:]*cc_resp.conversion, lw=1)
plt.ylabel('Voltage (V)')
plt.title('CurrentClamp Response (current_clamp-response-01-ch-0)')
plt.subplot(2,1,2)
plt.plot(t_cc, cc_stim.data[:]*cc_stim.conversion, color='orange', lw=1)
plt.ylabel('Current (A)')
plt.xlabel('Time (s)')
plt.suptitle('CurrentClamp: Response & Stimulus')
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig('explore/current_clamp_response01.png')
plt.close()

# Plot an example VoltageClamp response (22-ch-0) and its stimulus (22-ch-0)
vc_resp = nwb.acquisition['voltage_clamp-response-22-ch-0']
vc_stim = nwb.stimulus['stimulus-22-ch-0']
fs_vc = vc_resp.rate
t_vc = np.arange(vc_resp.data.shape[0]) / fs_vc

plt.figure(figsize=(8, 4))
plt.subplot(2,1,1)
plt.plot(t_vc, vc_resp.data[:]*vc_resp.conversion, lw=1)
plt.ylabel('Current (A)')
plt.title('VoltageClamp Response (voltage_clamp-response-22-ch-0)')
plt.subplot(2,1,2)
plt.plot(t_vc, vc_stim.data[:]*vc_stim.conversion, color='green', lw=1)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.suptitle('VoltageClamp: Response & Stimulus')
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig('explore/voltage_clamp_response22.png')
plt.close()

print("Done writing plots.")