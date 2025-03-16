import os
import numpy as np
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics, auxiliary, io
from config import fname
from settings_hmm_beta import (lfreq, hfreq, sfreq, pc_type, sessions, lag, K, n_labels, n_pca)
from hmm_visuals import (
    plot_viterbi_path,
    visualize_state_means)

# ===========================
# 1. PARAMETERS & PATHS
# ===========================

job_id = '_job_source_2'  # Change to avoid overwriting the files


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
data_processed, indices_new, log = preproc.preprocess_data(data, 
                                                           indices, 
                                                           fs=sfreq, 
                                                           dampen_extreme_peaks=True, 
                                                           standardise=True,
                                                           filter=(lfreq, hfreq),
                                                           detrend=True,
                                                           onpower=False,
                                                           onphase=False,
                                                           pca=0.9,
                                                           exact_pca=True,
                                                           ica=None,
                                                           post_standardise=True,
                                                           downsample=100)
print("\nData preprocessed.")
print("Data shape:", data_processed.shape)
print("Indices shape:", indices_new.shape)


# Time-delay embedding

# Define the number of lags (= window size)
#embeddedlags = list(range(-lag, lag + 1, 1))
embeddedlags = np.arange(lag)
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

log["pca"] = n_pca # store number of pca components in logs
log["pcamodel"] = pcamodel # store pcamodel in logs

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
    "initrep": 1,           # To make the training quicker - leave by default if not just testing
    "initcyc": 1,           # To make the training quicker - leave by default if not just testing
    "cyc": 1                # To make the training quicker - leave by default if not just testing
}

Gamma_tde, Xi, FE = hmm.train(X=None, Y=data_tde, indices=indices_tde, options=options)

print("\n------------- TDE-HMM model trained-------------\n")


hmm_path = fname.tde_hmm_path
filename = "latest-tde-hmm-pca-09.pkl"
glhmm.io.save_hmm(hmm, filename, directory=hmm_path)


# ===========================
# 4. OUTPUT EXTRACTION & VISUALISATION
# ===========================


print("\n------------- Output extraction begins -------------\n")

print(f"Data dimension of Gamma-TDE: {Gamma_tde.shape}")

# Viterbi path (discrete state sequence)
vpath = hmm.decode(X=None, Y=data_tde, indices=indices_tde, viterbi=True)
plot_viterbi_path(vpath)

# Gamma reconstruction?
"""T = auxiliary.get_T(indices)
options ={'embeddedlags': embeddedlags}
Gamma = auxiliary.padGamma(Gamma_tde,
                          T,
                          options=options)"""
Gamma=Gamma_tde

# Retrieve the transition probability matrix
TP = hmm.P.copy()


# Fractional occupancy (FO): the fraction of time in each session that is occupied by each state
FO = utils.get_FO(Gamma, indices=indices_tde)

# Switching rate
SR = utils.get_switching_rate(Gamma, indices_tde) # switching rate indicates how quickly subjects switch between states

# Dwell times / State lifetimes
LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices_tde)

# Number of active states
active_K = hmm.get_active_K()

# TO DO: Spectral content for each state

# Retrieve state means using get_means() (shape: [n_features, K])
q = data_tde.shape[1] # the number of parcels/channels
K = hmm.hyperparameters["K"] # the number of states

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
r2 = hmm.get_r2(X=None, Y=data_tde, Gamma=Gamma)  # returns R-squared (proportion of the variance explained) for each session and each variable in Y.

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
         initial_probabilities=IP,
         explained_variance=r2,
         likelihood=llh,
         indices=indices_tde)

print("\n >>> Results saved to ", npz_file_path)
