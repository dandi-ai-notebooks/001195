# %% [markdown]
# # Exploring Dandiset 001195: Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State
#
# **Dandiset Version:** 0.250408.1733
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset, titled "Separable Dorsal Raphe Dopamine Projections Mediate the Facets of Loneliness-like State," contains supporting data for the publication by Lee, Matthews, et al. (2025). The research investigates how different dopamine projections from the dorsal raphe nucleus (DRN) influence aspects of loneliness-like states in mice.
#
# The dataset includes:
# - In vivo calcium imaging data
# - Supporting behavioral videos
# - Ex vivo patch-clamp electrophysiology recordings
#
# **Key areas of study:** Dorsal Raphe Nucleus (DRN), Central Amygdala (CeA), Bed Nucleus of the Stria Terminalis (BNST), Posterior Basolateral Amygdala (BLP).
#
# **Experimental manipulations/observations:** Optogenetics, social isolation, social motivation.
#
# **Link to the Dandiset:** [https://dandiarchive.org/dandiset/001195/0.250408.1733](https://dandiarchive.org/dandiset/001195/0.250408.1733)
#
# ## What this notebook covers
#
# This notebook will guide you through:
# 1. Loading the Dandiset information using the DANDI API.
# 2. Listing some of the assets (files) within the Dandiset.
# 3. Selecting an NWB (Neurodata Without Borders) file from the Dandiset.
# 4. Loading and inspecting metadata from the selected NWB file.
# 5. Loading and visualizing some electrophysiology data (current clamp series) from the NWB file.
#
# ## Required Packages
#
# To run this notebook, you will need the following Python packages. It is assumed they are already installed:
# - `dandi` (for interacting with the DANDI Archive)
# - `pynwb` (for working with NWB files)
# - `h5py` (for HDF5 file support, used by PyNWB)
# - `remfile` (for accessing remote files)
# - `numpy` (for numerical operations)
# - `matplotlib` (for plotting)
# - `seaborn` (for enhanced visualizations)
#
# No `pip install` commands are included in this notebook.

# %% [markdown]
# ## 1. Load Dandiset Information with DANDI API

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import 환경 # This is a placeholder, will be replaced by actual environment variables or configuration
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "001195"
dandiset_version = "0.250408.1733" # Latest version as of notebook creation
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {dandiset.get_metadata().url}") # Using .get_metadata().url for the correct URL
print(f"Dandiset description: {metadata.get('description', 'No description available.')}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## 2. Load and Inspect an NWB File
#
# We will now select one of the NWB files from the Dandiset to explore its contents.
#
# We will use the following file:
# **Path:** `sub-23/sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb`
# **Asset ID:** `a243dde4-c270-42a9-8550-025f5ffcd5a7`
#
# The direct download URL for this asset is:
# `https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/`
#
# We will use `remfile` and `pynwb` to load this NWB file directly from its URL.

# %%
# URL of the selected NWB file
nwb_asset_id = "a243dde4-c270-42a9-8550-025f5ffcd5a7"
nwb_url = f"https://api.dandiarchive.org/api/assets/{nwb_asset_id}/download/"
print(f"Loading NWB file from: {nwb_url}")

# Load the NWB file
# The following lines are based on the output from `tools_cli.py nwb-file-info`
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode
nwbfile = io.read()

print("\nNWB file loaded successfully.")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session description: {nwbfile.session_description}")
print(f"Session start time: {nwbfile.session_start_time}")
print(f"Experimenter(s): {nwbfile.experimenter}")
if nwbfile.subject:
    print(f"Subject ID: {nwbfile.subject.subject_id}")
    print(f"Subject species: {nwbfile.subject.species}")
    print(f"Subject sex: {nwbfile.subject.sex}")
    print(f"Subject age: {nwbfile.subject.age}")

# %% [markdown]
# ### Neurosift Link for this NWB file
#
# You can explore this NWB file interactively on Neurosift using the following link:
#
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/&dandisetId=001195&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/a243dde4-c270-42a9-8550-025f5ffcd5a7/download/&dandisetId=001195&dandisetVersion=draft)
#
# (Note: The `dandisetVersion` in the Neurosift link might sometimes be pinpointing `draft` or a specific published version like `0.250408.1733`. The link provided aims for general accessibility).

# %% [markdown]
# ### Summarizing NWB File Contents
#
# Let's look at the structure of this NWB file. NWB files organize data into various groups. Key groups include:
# - `acquisition`: Raw acquired data, typically time series.
# - `stimulus`: Stimulus data presented during the experiment.
# - `processing`: Processed data derived from raw acquisition.
# - `intervals`: Time intervals of interest, like trials or experimental epochs.
# - `general`: General metadata about the experiment, subject, etc.
#
# We can list the contents of some of these groups.

# %%
print("Contents of the NWB file:")

print("\n--- Acquisition ---")
if nwbfile.acquisition:
    for acq_name, acq_data in nwbfile.acquisition.items():
        print(f"- {acq_name}: {type(acq_data)}")
        if hasattr(acq_data, 'description'):
             print(f"  Description: {acq_data.description}")
        if hasattr(acq_data, 'data') and hasattr(acq_data.data, 'shape'):
             print(f"  Data shape: {acq_data.data.shape}")
else:
    print("No data in acquisition group.")

print("\n--- Stimulus ---")
if nwbfile.stimulus:
    for stim_name, stim_data in nwbfile.stimulus.items():
        print(f"- {stim_name}: {type(stim_data)}")
        if hasattr(stim_data, 'description'):
            print(f"  Description: {stim_data.description}")
        if hasattr(stim_data, 'data') and hasattr(stim_data.data, 'shape'):
            print(f"  Data shape: {stim_data.data.shape}")
else:
    print("No data in stimulus group.")

print("\n--- Icephys Electrodes ---")
if nwbfile.icephys_electrodes:
    for electrode_name, electrode_data in nwbfile.icephys_electrodes.items():
        print(f"- {electrode_name}: {type(electrode_data)}")
        if hasattr(electrode_data, 'description'):
            print(f"  Description: {electrode_data.description}")
else:
    print("No icephys electrodes defined.")

print("\n--- Lab Metadata (DandiIcephysMetadata) ---")
if "DandiIcephysMetadata" in nwbfile.lab_meta_data:
    dandi_meta = nwbfile.lab_meta_data["DandiIcephysMetadata"]
    print(f"- Cell ID: {dandi_meta.cell_id}")
    print(f"- Slice ID: {dandi_meta.slice_id}")
    print(f"- Targeted Layer: {dandi_meta.targeted_layer}")
    print(f"- Inferred Layer: {dandi_meta.inferred_layer}")
else:
    print("No DandiIcephysMetadata found.")

# %% [markdown]
# ## 3. Load and Visualize Data from the NWB file
#
# This NWB file contains intracellular electrophysiology data, specifically current clamp and voltage clamp series.
#
# Let's focus on a `CurrentClampSeries` from the acquisition data. The `tools_cli.py nwb-file-info` output showed many `current_clamp-response-XX-ch-Y` series. We'll pick one, for example, `current_clamp-response-01-ch-0`.
#
# The description for these series is: "Response to: episodic stimulation, 1s steps, 20pA increments, -120pA to 260pA". This indicates a series of current injection steps.

# %%
# Select a current clamp series to analyze
# Based on the nwb-file-info output, these are present.
series_name_response = "current_clamp-response-01-ch-0"
series_name_stimulus = "stimulus-01-ch-0" # Corresponding stimulus

if series_name_response in nwbfile.acquisition and series_name_stimulus in nwbfile.stimulus:
    response_series = nwbfile.acquisition[series_name_response]
    stimulus_series = nwbfile.stimulus[series_name_stimulus]

    print(f"Analyzing response: {series_name_response}")
    print(f"Description: {response_series.description}")
    print(f"Unit: {response_series.unit}")
    print(f"Data shape: {response_series.data.shape}")
    print(f"Sampling rate: {response_series.rate} Hz")
    print(f"Duration: {response_series.data.shape[0] / response_series.rate} seconds")

    print(f"\nAnalyzing stimulus: {series_name_stimulus}")
    print(f"Description: {stimulus_series.description}")
    print(f"Unit: {stimulus_series.unit}")
    print(f"Data shape: {stimulus_series.data.shape}")

    # Load a subset of data to avoid excessive memory usage/download time
    num_points_to_load = response_series.data.shape[0] # Load all points for this short series
    
    # Data is stored as raw values and needs conversion
    # data = (raw_data * conversion) + offset
    response_data_raw = response_series.data[:num_points_to_load]
    response_data_converted = (response_data_raw * response_series.conversion) + response_series.offset

    stimulus_data_raw = stimulus_series.data[:num_points_to_load]
    stimulus_data_converted = (stimulus_data_raw * stimulus_series.conversion) + stimulus_series.offset
    
    # Create a time vector
    time_vector = np.arange(num_points_to_load) / response_series.rate

    # Plot the data
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot response
    axs[0].plot(time_vector, response_data_converted)
    axs[0].set_title(f"Response: {series_name_response}")
    axs[0].set_ylabel(f"Voltage ({response_series.unit})")
    axs[0].grid(True)

    # Plot stimulus
    axs[1].plot(time_vector, stimulus_data_converted)
    axs[1].set_title(f"Stimulus: {series_name_stimulus}")
    axs[1].set_ylabel(f"Current ({stimulus_series.unit})")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

else:
    print(f"Could not find {series_name_response} in acquisition or {series_name_stimulus} in stimulus.")
    print("Available acquisition series:", list(nwbfile.acquisition.keys()))
    print("Available stimulus series:", list(nwbfile.stimulus.keys()))


# %% [markdown]
# The plot above shows the recorded membrane potential (top panel) in response to a current injection (bottom panel) for the first segment of an episodic stimulation protocol. The description indicates that this protocol involves 1-second steps with 20pA increments. This specific trace likely corresponds to one of these steps. The data shape (3000 points) and sampling rate (10000 Hz) suggest a duration of 0.3 seconds for this particular segment, not 1s as the overall protocol step. It's possible `current_clamp-response-01-ch-0` is one part of a longer sequence making up a full 1s step, or the description refers to the overall design rather than each individual data snippet.

# %% [markdown]
# ### Visualizing Multiple Sweeps
#
# The NWB file contains many current clamp series, likely corresponding to different current injection levels from the episodic stimulation protocol. Let's try to plot a few of these sweeps together to see how the cell responds to varying stimuli. We'll select a few consecutive response series (e.g., `current_clamp-response-01-ch-0` through `current_clamp-response-05-ch-0`).

# %%
# Select a few response series to plot together
num_sweeps_to_plot = 5 # For example, plot 5 sweeps
response_series_names = [f"current_clamp-response-{i:02d}-ch-0" for i in range(1, num_sweeps_to_plot + 1)]
stimulus_series_names = [f"stimulus-{i:02d}-ch-0" for i in range(1, num_sweeps_to_plot + 1)]

plt.figure(figsize=(12, 10))

# Plot responses
plt.subplot(2, 1, 1)
all_responses_valid = True
for i, series_name in enumerate(response_series_names):
    if series_name in nwbfile.acquisition:
        series = nwbfile.acquisition[series_name]
        data_raw = series.data[:] # Load all data for these relatively short series
        data_converted = (data_raw * series.conversion) + series.offset
        time_vector = (np.arange(len(data_converted)) / series.rate) + series.starting_time
        plt.plot(time_vector, data_converted, label=f"{series_name} (Start: {series.starting_time:.2f}s)")
    else:
        print(f"Warning: Response series {series_name} not found.")
        all_responses_valid = False
if all_responses_valid:
    plt.title(f"Overlay of First {num_sweeps_to_plot} Current Clamp Responses (Channel 0)")
    plt.ylabel(f"Voltage ({nwbfile.acquisition[response_series_names[0]].unit if response_series_names[0] in nwbfile.acquisition else 'V'})")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
else:
    plt.title(f"Could not plot all requested response sweeps.")


# Plot stimuli
plt.subplot(2, 1, 2)
all_stimuli_valid = True
for i, series_name in enumerate(stimulus_series_names):
    if series_name in nwbfile.stimulus:
        series = nwbfile.stimulus[series_name]
        data_raw = series.data[:]
        data_converted = (data_raw * series.conversion) + series.offset
        time_vector = (np.arange(len(data_converted)) / series.rate) + series.starting_time
        plt.plot(time_vector, data_converted, label=f"{series_name} (Start: {series.starting_time:.2f}s)")
    else:
        print(f"Warning: Stimulus series {series_name} not found.")
        all_stimuli_valid = False

if all_stimuli_valid:
    plt.title(f"Overlay of First {num_sweeps_to_plot} Current Clamp Stimuli (Channel 0)")
    plt.ylabel(f"Current ({nwbfile.stimulus[stimulus_series_names[0]].unit if stimulus_series_names[0] in nwbfile.stimulus else 'A'})")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
else:
    plt.title(f"Could not plot all requested stimulus sweeps.")


plt.tight_layout()
plt.show()

# %% [markdown]
# The plots display the first few electrophysiological sweeps from the experiment. The top panel shows the voltage responses, and the bottom panel shows the corresponding current stimuli. Each sweep starts at a different time, as indicated by their `starting_time` attribute, suggesting they are sequential segments of the stimulation protocol. The current steps seem to increase with each sweep, which is consistent with the protocol description ("20pA increments"). The neuron's response to these current injections can be observed in the voltage traces. For instance, larger depolarizing current steps might elicit action potentials.

# %% [markdown]
# ### Exploring `icephys_sequential_recordings` Table
#
# The NWB file also contains tables that organize these recordings. The `icephys_sequential_recordings` table groups sweeps that belong to the same experimental protocol. Let's inspect this table.

# %%
if nwbfile.icephys_sequential_recordings is not None:
    print("Icephys Sequential Recordings Table:")
    # Convert to pandas DataFrame for easier viewing if desired and if pandas is available
    try:
        import pandas as pd
        sequential_df = nwbfile.icephys_sequential_recordings.to_dataframe()
        print(sequential_df.head())

        # This table groups 'simultaneous_recordings'. Each row here represents an experimental paradigm
        # (e.g., a series of current steps).
        # 'simultaneous_recordings' column itself is a VectorIndex, pointing to rows in
        # the 'icephys_simultaneous_recordings' table.

        print("\nExample: First sequential recording details")
        first_seq_rec = sequential_df.iloc[0]
        print(f"Stimulus Type: {first_seq_rec['stimulus_type']}")

        # Get the set of simultaneous recordings associated with this sequential recording
        # The 'simultaneous_recordings' column in sequential_df contains DynamicTableRegion objects.
        # These are references to rows in the nwbfile.icephys_simultaneous_recordings table.
        # sim_rec_indices = first_seq_rec['simultaneous_recordings'] # This is a DynamicTableRegion
        # print(f"Number of simultaneous recordings in this sequence: {len(sim_rec_indices)}")
        
        # For more detailed introspection:
        # sim_recordings_df = nwbfile.icephys_simultaneous_recordings.to_dataframe()
        # referenced_sim_recs = sim_recordings_df.iloc[sim_rec_indices.data[:]]
        # print(referenced_sim_recs.head())

    except ImportError:
        print("Pandas not installed. Cannot display table as DataFrame.")
        print("Columns:", nwbfile.icephys_sequential_recordings.colnames)
        # You could iterate through rows manually if pandas isn't available, but it's more verbose.
    except Exception as e:
        print(f"Error displaying icephys_sequential_recordings table: {e}")
else:
    print("No icephys_sequential_recordings table found in this NWB file.")


# %% [markdown]
# The `icephys_sequential_recordings` table provides a high-level organization of the experimental protocol. Each row typically corresponds to a particular type of stimulus applied (e.g., "episodic stimulation"). Understanding this table can help in programmatically accessing all sweeps related to a specific experimental condition. The `tools_cli.py` output for `nwb-file-info` indicates that the `icephys_sequential_recordings` table in this file has 5 rows, and its columns are `simultaneous_recordings` and `stimulus_type`. This suggests 5 distinct "sequences" or experimental blocks.

# %% [markdown]
# ## 4. Summary and Future Directions
#
# This notebook demonstrated how to:
# 1.  Connect to the DANDI Archive and retrieve metadata for Dandiset 001195.
# 2.  List assets within the Dandiset.
# 3.  Load a specific NWB file (`sub-23/sub-23_ses-20150324T134114_slice-slice-1_cell-C1_icephys.nwb`) using its DANDI asset URL.
# 4.  Inspect basic metadata and structure of the NWB file.
# 5.  Access and visualize intracellular electrophysiology data (current clamp series and their corresponding stimuli).
# 6.  Briefly look at how recordings are organized in `icephys_sequential_recordings` table.
#
# ### Possible Future Directions:
#
# *   **Further Ephys Analysis:**
#     *   Analyze other sweeps and channels (e.g., `ch-1` which is also present).
#     *   Extract features from the responses, such as action potential threshold, firing frequency, input resistance, sag potential, etc., for different current injection levels.
#     *   Compare responses across different cells or experimental conditions if other NWB files from this Dandiset are analyzed.
#     *   Investigate data from `voltage_clamp-response-XX` series, which are also present in this file.
# *   **Exploring Other Data Types:** This Dandiset also mentions calcium imaging data and behavioral videos in its overall description. Other NWB files within this Dandiset might contain these data types, which could be explored similarly.
# *   **Relating to Experimental Context:** Connect the observed electrophysiological properties to the broader research questions of the Dandiset concerning dorsal raphe dopamine projections and loneliness-like states, if applicable to the ex vivo data. This often requires more detailed knowledge of the experimental design for each specific file.
#
# This notebook provides a starting point for exploring the rich data available in Dandiset 001195. Remember to consult the original publication and DANDI metadata for a complete understanding of the experimental design and context.
#
# ---
# *End of AI-generated notebook.*

# %%
# Final check to close NWB I/O object if it was opened
# This helps release the file handle, especially for remote files.
if 'io' in locals() and io is not None:
    try:
        io.close()
        print("\nNWBHDF5IO closed.")
    except Exception as e:
        print(f"Error closing NWBHDF5IO: {e}")

if 'h5_file' in locals() and h5_file is not None:
    try:
        h5_file.close()
        print("HDF5 file (from remfile) closed.")
    except Exception as e:
        # h5py file might be already closed if io.close() cascades, or if it was never opened properly.
        # print(f"Error closing HDF5 file: {e}")
        pass # Often this is fine, especially if io.close() handled it.

if 'remote_file' in locals() and remote_file is not None:
    try:
        remote_file.close() # remfile.File also has a close method.
        print("remfile.File closed.")
    except Exception as e:
        print(f"Error closing remfile: {e}")