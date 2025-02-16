import os
import numpy as np
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics
from config import (fname, processed_dir)
from hmm_visuals import (
    plot_viterbi_path,
    visualize_state_means)

# ===========================
# 1. PARAMETERS & PATHS
# ===========================
# Change to avoid overwriting the files
job_id = '_job_source_2'
side = 'both'
K = 4  # number of HMM states

# Sampling frequency and frequency band (for spectral estimation later)
fs = 200
lfreq = 0.1
hfreq = 48

# Path to the subject list file (one subject ID per line)
subjects_file = 'subject_text_files/test.txt"'
with open(subjects_file, 'r') as f:
    subjects = [line.strip() for line in f if line.strip()]

# Specify the session to process (here, session '01')
sessions = ["01"]#, "02", "03", "04", "05"]

# ===========================
# 2. DATA LOADING & CONCATENATION
# ===========================
X_list = []         # list to store each subject's data
indices_list = []   # to store start/end indices for each subject in the concatenated data
start = 0

for subject in subjects:
    for ses in sessions:
        # Determine the processed data path based on subject id length (as in the Matlab code)
        #if len(subj) == 6:
        #    processed_path = f'/m/nbe/scratch/controlmeg/sara_tuo/processed/{subj}/ses-{session}/'
        #else:
        #    processed_path = f'/m/nbe/scratch/tbi-meg/sara_tuo/processed/{subj}/ses-{session}/'
        
        
        ch_type = 'source-ave'
        
        # Construct the filename
        #n_sensors = 'ROI_source_nht'
        #file_name = f'{subj}_npy_data_{side}_{n_sensors}_sens_{lfreq}_{hfreq}.npy'
        #file_path = os.path.join(processed_dir, ch_type, file_name)
        
        file_path = fname.data_npy(subject=subject, ch_type=ch_type, ses=ses, lfreq=lfreq, hfreq=hfreq)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the data for this subject.
        # (Assume each file contains a NumPy array of shape (T, n_features)=(time points, channels) )
        X = np.load(file_path)
        
        # Determine the number of timepoints
        T = X.shape[0]
        
        # Append the subject data to the list
        X_list.append(X)
        
        # Record the start and end indices for this subject in the concatenated array
        end = start + T
        indices_list.append([start, end])
        start = end

# Concatenate all subjects’ data along the time axis
X_concat = np.concatenate(X_list, axis=0)
indices = np.array(indices_list)

# Preprocess the data (e.g., standardisation) as in the glhmm example.
# The preprocess_data function expects (data, indices) and returns the (processed_data, _)
X_concat, _ = preproc.preprocess_data(X_concat, indices)

print(f"Loaded and concatenated data from {len(subjects)} subjects.")
print("Data shape:", X_concat.shape)
print("Indices shape:", indices.shape)

# ===========================
# 3. INITIALISE & TRAIN THE HMM
# ===========================
# Instantiate the standard Gaussian HMM.
# We do not model interactions (model_beta='no') and we model the state-specific mean ('model_mean' defaults to 'state').
hmm = glhmm.glhmm(model_beta='no', K=K, covtype='full', model_mean='state')

# Optionally, inspect hyperparameters:
print("Hyperparameters:")
print(hmm.hyperparameters)

# Set a seed for reproducibility
np.random.seed(123)

# Train the HMM.
# Note: The glhmm.train method accepts:
#     X=None, Y=<concatenated data>, indices=<2D array specifying [start, end] for each subject/session>
Gamma,Xi,FE = hmm.train(X=None, Y=X_concat, indices=indices)

# ===========================
# 4. OUTPUT EXTRACTION & VISUALISATION
# ===========================
# Extract the Viterbi path (discrete state sequence) using decode.
vpath = hmm.decode(X=None, Y=X_concat, indices=indices, viterbi=True)


# Retrieve state means using get_means() (shape: [n_features, K])
q = X_concat.shape[1] # the number of parcels/channels
K = hmm.hyperparameters["K"] # the number of states
state_means = np.zeros(shape=(q, K))
state_means = hmm.get_means()

# Retrieve the transition probability matrix
TP = hmm.P.copy()

# State covariances: Time-varying functional connectivity
state_FC = np.zeros(shape=(q, q, K))
for k in range(K):
    state_FC[:,:,k] = hmm.get_covariance_matrix(k=k) # the state covariance matrices in the shape (no. features, no. features, no. states)
    

# Fractional occupancy (FO): the fraction of time in each session that is occupied by each state
FO = utils.get_FO(Gamma, indices=indices)

# Switching rate
SR = utils.get_switching_rate(Gamma, indices) # switching rate indicates how quickly subjects switch between states

# Dwell times / State lifetimes
LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices)

# ===========================
# 5. SPECTRAL ESTIMATION PER STATE
# ===========================
# Here we perform a simple spectral estimation using Welch's method.
# For each state, we extract the timepoints (from each subject) where that state is active and compute an average power spectrum.
spectra = {}
for state in range(K):
    state_segments = []
    
    # Loop over subjects using the indices
    for i in range(len(indices)):
        start_idx, end_idx = indices[i]
        # Extract subject-specific Viterbi path and data
        subject_vpath = vpath[start_idx:end_idx]
        subject_data = X_concat[start_idx:end_idx, :]
        
        # Find indices where the state is active
        state_idx = np.where(subject_vpath == state)[0]
        if state_idx.size > 0:
            state_segments.append(subject_data[state_idx, :])
    
    if state_segments:
        # Concatenate data from all segments in this state
        state_data_all = np.concatenate(state_segments, axis=0)
        # Compute power spectrum for each feature and average over features
        psd_list = []
        for feature in range(state_data_all.shape[1]):
            f, psd = welch(state_data_all[:, feature], fs=fs, nperseg=5*fs)
            psd_list.append(psd)
        avg_psd = np.mean(psd_list, axis=0)
        spectra[state] = {'freq': f, 'psd': avg_psd}
    else:
        spectra[state] = {'freq': None, 'psd': None}

# Optionally, plot the average power spectrum for each state.
plt.figure(figsize=(10, 6))
for state in range(K):
    if spectra[state]['freq'] is not None:
        plt.plot(spectra[state]['freq'], spectra[state]['psd'], label=f'State {state+1}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Average Power Spectrum per State')
plt.legend()
plt.tight_layout()
plt.show()

# ===========================
# 6. SAVE RESULTS
# ===========================
# Save the HMM object and the key outputs to an npz file.
save_path = f'/m/nbe/scratch/controlmeg/sara_tuo/processed/group/sensors_concat_group{job_id}.npz'
np.savez(save_path,
         model=hmm,
         state_means=state_means,
         transition_probabilities=TP,
         viterbi_path=vpath,
         indices=indices,
         spectra=spectra)

print("Results saved to", save_path)
