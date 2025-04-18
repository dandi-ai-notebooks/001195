# %% [markdown]
# # Exploring Dandiset 001195: Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State
# 
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Users should be cautious when interpreting the code or results.
# 
# ## Overview
# This Dandiset comprises data for the study of dopamine projections related to loneliness-like states. The data includes in vivo calcium imaging with behavioral videos and ex vivo patch-clamp electrophysiology recordings.
# 
# **Link to Dandiset:** [Dandiset 001195](https://dandiarchive.org/dandiset/001195)
# 
# ## Contents of the Notebook
# This notebook will cover:
# - Loading the Dandiset using the DANDI API
# - Exploring and visualizing data from NWB files
# - Providing insightful visualizations and guidance on data analysis
# 
# ## Required Packages
# The following packages are required to run this notebook:
# - dandi
# - pynwb
# - h5py
# - remfile
# - matplotlib

# %% [markdown]
# ## Loading the Dandiset
# Using the DANDI API, we can access the metadata and assets of the Dandiset.

# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001195")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['citation']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading NWB File
# Let's load one of the NWB files and show some basic metadata. We'll use a sample file `sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb`.

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print NWB metadata
print(f"NWB Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")

# %% [markdown]
# ## Visualizing Data from NWB File
# We will visualize a subset of data from a specific CurrentClampSeries to avoid overwhelming output.
# 
# Below, we demonstrate how to access and plot a small segment of the data.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Accessing a sample data segment
data_series = nwb.acquisition["current_clamp-response-01-ch-0"].data
sample_data = data_series[:100]  # Load smaller subset of data for plotting

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(sample_data, label='Voltage Response (First 100 samples)')
plt.title("Sample Voltage Response from CurrentClampSeries")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (Volts)")
plt.legend()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# In this notebook, we've explored the Dandiset 001195 and demonstrated how to load and visualize some data from an NWB file. Future analysis could involve detailed examination of different datasets, advanced visualization techniques, or integration of additional datasets to provide comprehensive insights into the biophysical properties of neurons studied herein.