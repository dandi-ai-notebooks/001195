# %% [markdown]
# # Exploring Dandiset 001195: Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State

# %% [markdown]
# **Disclaimer**: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
# This Dandiset contains data supporting the publication titled "Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State".
# [Dandiset 001195](https://dandiarchive.org/dandiset/001195)

# ## What this Notebook Covers
# - Loading the Dandiset
# - Visualizing Current Clamp Responses
# - Analyzing Stimulus Series

# %% [markdown]
# ## Required Packages
# Ensure the following packages are installed: `pynwb`, `h5py`, `matplotlib`, `remfile`, `dandi`.

# %% 
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001195")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading NWB File

# %%
# Load NWB file
url = "https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print session information
print(f"Session description: {nwb.session_description}")
print(f"Experimenter: {nwb.experimenter}")

# %% [markdown]
# ## Visualization

# Plot Current Clamp Response
current_clamp_series = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus_series = nwb.stimulus["stimulus-01-ch-0"]

# Time axis for the plot
data_length = current_clamp_series.data.shape[0]
time = np.arange(data_length) / current_clamp_series.rate

# Ensure the datasets are converted to NumPy arrays
current_clamp_data = current_clamp_series.data[:]
stimulus_data = stimulus_series.data[:]

# Plot current clamp response
plt.figure(figsize=(12, 6))
plt.plot(time, current_clamp_data, label="Response", color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Current (V)')
plt.title('Current Clamp Response')
plt.legend()
plt.show()

# Plot stimulus series
plt.figure(figsize=(12, 6))
plt.plot(time, stimulus_data, label="Stimulus", color='green')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Stimulus Series')
plt.legend()
plt.show()

io.close()
remote_file.close()

# %% [markdown]
# ## Summary
# The current clamp response and stimulus series were visualized to examine the dynamics of the NWB data. Further analyses can expand on different datasets or analysis techniques within the Dandiset to explore different facets of the research question.