import os
import numpy as np
import pandas as pd
from glhmm import glhmm, preproc, utils
from config import fname
from settings_hmm_beta import (lfreq, hfreq, sfreq, pc_type, sessions, lag, K, n_pca)
from compute_psd import (compute_statewise_spectra_windowed)

# ===========================
# 1. PARAMETERS & PATHS
# ===========================

job_id = '_job_source_30v9_ALL_complete'  # Change to avoid overwriting the files


# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])


# ===========================
# 2. DATA LOADING & CONCATENATION
# ===========================
X_list = []         # list to store each subject's data
indices_list = []   # to store start/end indices for each subject in the concatenated data
start = 0

print("\n------------- Data processing begins -------------\n")

for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)
    for ses in sessions:        
        file_path = fname.data_npy(subject=subject, pc_type=pc_type, ses=ses, l_freq=str(lfreq), h_freq=str(hfreq))
        
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
data = np.concatenate(X_list, axis=0)
indices = np.array(indices_list)

print(f"\nLoaded and concatenated data from {df_subjects.shape[0]} subjects.")
print("Data shape:", data.shape)
print("Indices shape:", indices.shape)

n_timepoints, n_subjects, n_channels = data.shape[0], indices.shape[0], data.shape[1]

# ===========================
# 3. DATA PREPROCESSING FOR TDE-HMM
# ===========================

# Preprocess the data
# The preprocess_data function expects (data, indices) and returns the (processed_data, _)
data_processed, indices_new, log = preproc.preprocess_data(data, 
                                                           indices, 
                                                           fs=sfreq, 
                                                           dampen_extreme_peaks=None, 
                                                           standardise=True,
                                                           filter=(lfreq, hfreq),
                                                           detrend=False,
                                                           onpower=False,
                                                           onphase=False,
                                                           #pca=n_pca,
                                                           #exact_pca=True,
                                                           ica=None,
                                                           #post_standardise=True,
                                                           downsample=None)
print("\nData preprocessed.")
print("Data shape:", data_processed.shape)
print("Indices shape:", indices_new.shape)


# Time-delay embedding

# Define the number of lags (= window size)
embeddedlags = list(range(-lag, lag + 1, 1))

print("\nLags for TDE-HMM: ", embeddedlags)

# Define the number of PCA components
#noPca = n_labels*2      # 2 x the number of regions (logic explained in Vidaurre et al. 2018)
print("PCA components for TDE-HMM: ", n_pca)

# Build the time-delay embedded data
data_tde, indices_tde, pcamodel = preproc.build_data_tde(
    data_processed,
    indices_new,
    embeddedlags,
    pca=n_pca,
    standardise_pc=True
)

# Adding this information in the preproclog to avoid errors
log["pca"] = n_pca # store number of pca components in logs
log["pcamodel"] = pcamodel # store pcamodel in logs
log["icamodel"] = None

# Convert indices to integer type
#indices_tde = indices_tde.astype(int)

print("\nData built for the time-delay embedded HMM model.")
print("Concatenated & preprocessed TDE data shape:", data_tde.shape)
print("Updated TDE time stamp indices shape:", indices_tde.shape)
    

# ===========================
# 4. INITIALISE & TRAIN THE HMM
# ===========================

print("\n------------- Initializing HMM -------------\n")

hmm = glhmm.glhmm(
    K=K,
    covtype='full',
    model_beta='no',
    preproclog=log
)

print("\nLogs of HMM model:")
print(hmm.preproclog)

# Optionally, inspect hyperparameters:
print("\nHyperparameters of HMM model:")
print(hmm.hyperparameters)


# Train the HMM.

print("\n------------- TDE-HMM model training begins -------------\n")
# Set a seed for reproducibility
np.random.seed(123)


options = {
    "gpu_acceleration": 1,  # Enable GPU when >= 1
    "gpuChunks": 1,         # Split data if needed for large datasets         
    "tol": 1e-5
}


"""

options = {
    "gpu_acceleration": 1,  # Enable GPU when >= 1
    "gpuChunks": 1,         # Split data if needed for large datasets
    "initrep": 1,           # To make the training quicker - leave by default if not just testing
    "initcyc": 1,           # To make the training quicker - leave by default if not just testing
    "cyc": 1,                # To make the training quicker - leave by default if not just testing
    "tol": 1e-5
}
"""

Gamma_tde, Xi, FE = hmm.train(X=None, Y=data_tde, indices=indices_tde, options=options)

print("\n------------- TDE-HMM model trained-------------\n")

# Saving the trained HMM model with GLHMM saving function
hmm_path = fname.tde_hmm_path
filename = "latest-tde-hmm-group-pca-45-complete.pkl"
glhmm.io.save_hmm(hmm, filename, directory=hmm_path)


# ===========================
# 4. OUTPUT EXTRACTION & VISUALISATION
# ===========================


print("\n------------- Output extraction begins -------------\n")

# Viterbi path (discrete state sequence)
vpath = hmm.decode(X=None, Y=data_tde, indices=indices_tde, viterbi=True)

# Retrieve the transition probability matrix
TP = hmm.P.copy()

Gamma = Gamma_tde

# === Check if Gamma sums to 1 at each time point ===
gamma_sums = Gamma.sum(axis=1)
bad_indices = np.where(np.abs(gamma_sums - 1) > 1e-5)[0]

if len(bad_indices) > 0:
    print(f"Warning: {len(bad_indices)} timepoints where Gamma does not sum to 1.")
    print("Example of problematic Gamma sums:", gamma_sums[bad_indices[:5]])
else:
    print("Gamma is well-behaved: all timepoints sum to 1.")

# Fractional occupancy (FO): the fraction of time in each session that is occupied by each state
FO = utils.get_FO(Gamma, indices=indices_tde)

# Switching rate
SR = utils.get_switching_rate(Gamma, indices_tde) # switching rate indicates how quickly subjects switch between states

# Dwell times / State lifetimes
LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices_tde)

# Number of active states
active_K = hmm.get_active_K()

# State covariances: Time-varying functional connectivity
q = data_tde.shape[1] # the number of parcels/channels
K = hmm.hyperparameters["K"] # the number of states

state_FC = np.zeros(shape=(q, q, K))

for k in range(K):
    state_FC[:,:,k] = hmm.get_covariance_matrix(k=k, orig_space=False) # the state covariance matrices in the shape (no. features, no. features, no. states)


# Spectra
spectra, freqs = compute_statewise_spectra_windowed(
    data_processed,
    Gamma,
    sfreq=sfreq,
    lag=lag,
    fmin=1,
    fmax=hfreq,
    bandwidth=4.0,
    win_length_sec=5,
    step_sec=2.5  # 50% overlap
)


#### MODEL DIAGNOSTICS ####

# Initial probabilities
IP = hmm.get_Pi()

# The likelihood of the model per state and time point 
llh = hmm.loglikelihood(X=None, Y=data_tde)


print("\n------------- Output extraction finished -------------\n")

# ===========================
# 5. SAVE RESULTS
# ===========================
# Save the HMM object and the key outputs to an npz file.
npz_file_path = fname.tde_hmm_ob(job_id=job_id)
np.savez(npz_file_path,
         model=hmm,
         active_states=active_K,
         transition_probabilities=TP,
         viterbi_path=vpath,
         gamma=Gamma,
         switching_rate=SR,
         fractional_occupancy=FO,
         dwell_time_mean=LTmean,
         spectra=spectra,
         spectra_freqs=freqs,
         initial_probabilities=IP,
         likelihood=llh,
         indices=indices_tde,
         q=q)

print("\n >>> Results saved to ", npz_file_path)
