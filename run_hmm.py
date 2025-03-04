import os
import numpy as np
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics
from config import fname
from settings_hmm_beta import (lfreq, hfreq, sfreq, pc_type, sessions, lag)
from hmm_visuals import (
    plot_viterbi_path,
    visualize_state_means)

# ===========================
# 1. PARAMETERS & PATHS
# ===========================

job_id = '_job_source_2'  # Change to avoid overwriting the files

K = 6  # number of HMM states

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

# ===========================
# 3. DATA PREPROCESSING FOR TDE-HMM
# ===========================

# Preprocess the data
# The preprocess_data function expects (data, indices) and returns the (processed_data, _)
data_processed, indices_new, log = preproc.preprocess_data(data, indices, pca=0.99, post_standardise=True)
print("\nData preprocessed.")
print("Data shape:", data_processed.shape)
print("Indices shape:", indices_new.shape)


# Time-delay embedding

# Define the number of lags (= window size)
lag_val =list(range(-7, 8, 1))

# Build the time-delay embedded data
data_tde, indices_tde = preproc.build_data_tde(
    data_processed,
    indices_new,
    lag_val
)

# Convert indices to integer type
indices_tde = indices_tde.astype(int)

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

# Optionally, inspect hyperparameters:
print("\nHyperparameters of HMM model:")
print(hmm.hyperparameters)


# Train the HMM.

print("\n------------- TDE-HMM model training begins -------------\n")
# Set a seed for reproducibility
np.random.seed(123)

options = {
    "gpu_acceleration": 1,  # Enable GPU when >= 1
    "gpuChunks": 1         # Split data if needed for large datasets
}

Gamma_tde, Xi, FE = hmm.train(X=None, Y=data_tde, indices=indices_tde, options=options)

print("\n------------- TDE-HMM model trained-------------\n")

# ===========================
# 4. OUTPUT EXTRACTION & VISUALISATION
# ===========================

# Gamma extraction?


# Viterbi path (discrete state sequence)
vpath = hmm.decode(X=None, Y=data_tde, indices=indices_tde, viterbi=True)


"""# Retrieve state means using get_means() (shape: [n_features, K])
q = X_concat.shape[1] # the number of parcels/channels
K = hmm.hyperparameters["K"] # the number of states
state_means = np.zeros(shape=(q, K))
state_means = hmm.get_means()

# Retrieve the transition probability matrix
TP = hmm.P.copy()

# Fractional occupancy (FO): the fraction of time in each session that is occupied by each state
FO = utils.get_FO(Gamma, indices=indices)

# Switching rate
SR = utils.get_switching_rate(Gamma, indices) # switching rate indicates how quickly subjects switch between states

# Dwell times / State lifetimes
LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices)

# Number of active states
active_K = hmm.get_active_K()

# Spectral content for each state

# Retrieve the covariance matrices for all states; the function returns an array of shape (n_variables, n_variables, n_states)
covmats = hmm.get_covariance_matrices(orig_space=True)

# State covariances: Time-varying functional connectivity
state_FC = np.zeros(shape=(q, q, K))
for k in range(K):
    state_FC[:,:,k] = hmm.get_covariance_matrix(k=k) # the state covariance matrices in the shape (no. features, no. features, no. states)


#### MODEL DIAGNOSTICS ####

# Initial probabilities
IP = hmm.get_Pi()

# Explained variance per session (how much of the variance in data Y is explained by the model)
r2 = hmm.get_r2(X=None, Y=data_tde, Gamma)  # returns R-squared (proportion of the variance explained) for each session and each variable in Y.

# The likelihood of the model per state and time point 
llh = hmm.loglikelihood(=None, Y=data_tde)


print("\n------------- Output extraction finished -------------\n")

# ===========================
# 5. SAVE RESULTS
# ===========================
# Save the HMM object and the key outputs to an npz file.
npz_file_path = fname.tde_hmm_ob(job_id=job_id)
np.savez(npz_file_path,
         model=hmm,
         state_means=state_means,
         transition_probabilities=TP,
         viterbi_path=vpath,
         indices=indices,
         spectra=spectra)

print("\n >>> Results saved to ", npz_file_path)


"""