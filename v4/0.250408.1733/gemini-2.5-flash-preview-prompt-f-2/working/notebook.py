# %% [markdown]
# # Exploring Dandiset 001195: Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset includes supporting data for the publication "Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State" by Christopher R. Lee, Gillian A. Matthews, et al. The dataset contains in vivo calcium imaging data with supporting behavioral videos and ex vivo patch-clamp electrophysiology recordings, focusing on the role of dorsal raphe dopamine projections in loneliness-like states.
#
# You can find more information about this Dandiset on the DANDI Archive: https://dandiarchive.org/dandiset/001195/0.250408.1733

# %% [markdown]
# ## What this notebook covers
#
# This notebook demonstrates how to access and explore the data within Dandiset 001195 using the DANDI Python client and PyNWB. Specifically, it shows how to:
#
# 1. Connect to the DANDI Archive and access the Dandiset.
# 2. List and explore the assets (files) within the Dandiset.
# 3. Load a specific NWB file and inspect its metadata.
# 4. Access and visualize electrophysiology data (current clamp series) from the NWB file.

# %% [markdown]
# ## Required packages
#
# This notebook requires the following packages to be installed in your Python environment:
#
# * dandi
# * pynwb
# * remfile
# * h5py
# * numpy
# * matplotlib

# %% [markdown]
# ## Loading the Dandiset
#
# We can connect to the DANDI Archive and access the Dandiset using the `dandi` Python client.

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001195", "0.250408.1733")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB file and exploring its contents
#
# The primary data format in this Dandiset is NWB (Neurodata Without Borders). We can load individual NWB files using PyNWB and `remfile` for streaming access.
#
# We will load the file: `sub-23/sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb`.
#
# The URL for this asset is: `https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/`

# %%
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Show some basic metadata from the NWB file
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject species: {nwb.subject.species}")

# %% [markdown]
# ### Structure of the NWB file
#
# This NWB file contains intracellular electrophysiology data. Key data are stored in the `acquisition` and `stimulus` groups.
#
# *   **acquisition**: Contains the recorded physiological data, such as voltage responses.
# *   **stimulus**: Contains the applied stimuli, such as current injections.
#
# We can see the available data streams within the `acquisition` and `stimulus` groups by inspecting the keys:

# %%
print("Acquisition keys:")
for key in nwb.acquisition.keys():
    print(f"- {key}")

print("\nStimulus keys:")
for key in nwb.stimulus.keys():
    print(f"- {key}")

# %% [markdown]
# ### Exploring the NWB file on Neurosift
#
# You can explore this NWB file interactively on Neurosift using the following link:
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/&dandisetId=001195&dandisetVersion=draft

# %% [markdown]
# ## Visualizing Electrophysiology Data
#
# Let's visualize some of the recorded current clamp series and their corresponding stimuli.

# %% [markdown]
# ### Current Clamp Series 01
#
# This series shows the voltage response to a square pulse current stimulus.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Get data
response_01 = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus_01 = nwb.stimulus["stimulus-01-ch-0"]

# Get time vector
sampling_rate_01 = response_01.rate
t_01 = np.arange(response_01.data.shape[0]) / sampling_rate_01 + response_01.starting_time

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_01, response_01.data[:] * response_01.conversion, label='Response')
plt.ylabel(f'Voltage ({response_01.unit})')
plt.title('Current Clamp Series 01 - Channel 0')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_01, stimulus_01.data[:] * stimulus_01.conversion, label='Stimulus', color='orange')
plt.xlabel(f'Time ({response_01.starting_time_unit})')
plt.ylabel(f'Current ({stimulus_01.unit})')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Current Clamp Series 05
#
# This series shows the voltage response to a different square pulse current stimulus.

# %%
# Get data
response_05 = nwb.acquisition["current_clamp-response-05-ch-0"]
stimulus_05 = nwb.stimulus["stimulus-05-ch-0"]

# Get time vector
sampling_rate_05 = response_05.rate
t_05 = np.arange(response_05.data.shape[0]) / sampling_rate_05 + response_05.starting_time

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_05, response_05.data[:] * response_05.conversion, label='Response')
plt.ylabel(f'Voltage ({response_05.unit})')
plt.title('Current Clamp Series 05 - Channel 0')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_05, stimulus_05.data[:] * stimulus_05.conversion, label='Stimulus', color='orange')
plt.xlabel(f'Time ({response_05.starting_time_unit})')
plt.ylabel(f'Current ({stimulus_05.unit})')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic introduction to accessing and visualizing electrophysiology data from Dandiset 001195. We demonstrated how to load the Dandiset, access an NWB file, explore its structure, and plot current clamp recordings.
#
# Possible future directions for analysis include:
#
# *   Exploring other NWB files in the Dandiset, including calcium imaging and behavioral video data.
# *   Performing more detailed analysis of the electrophysiology recordings, such as calculating passive membrane properties or analyzing spiking behavior (note: spike detection and sorting are computationally intensive and not suitable for a simple notebook).
# *   Analyzing the behavioral video data in conjunction with the physiological recordings.
# *   Investigating the relationship between the different data modalities within the Dandiset.
#
# Remember to refer to the Dandiset metadata and the original publication for more context on the experimental design and data collected.

# %%
# Close the NWB file
io.close()