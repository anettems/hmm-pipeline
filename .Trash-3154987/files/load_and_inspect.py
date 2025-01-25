import mne

# file paths
data_path = "/m/nbe/project3/hmmpipeline/control_data/sub-16C/ses-01/meg/"
raw_file = data_path + "sub-16C_ses-01_task-eo_proc-raw_meg_tsss_mc_mfilter.fif"
emptyroom_file = data_path + "emptyroom_tsss.fif"

# Loading the raw MEG data
print("Loading MEG data...")
raw = mne.io.read_raw_fif(raw_file, preload=True)
print(raw.info)  # Display information about the MEG data

# Visualizing the raw data
print("Plotting raw MEG data...")
#raw.plot()
raw.plot(duration=5, n_channels=30)

# Loading the empty room noise data (optional, if needed)
emptyroom = mne.io.read_raw_fif(emptyroom_file, preload=True)
print("Empty room data loaded.")