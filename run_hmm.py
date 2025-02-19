import os
import numpy as np
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from glhmm import glhmm, preproc, utils, graphics
from config import (fname, processed_dir)
from settings_hmm_beta import (lfreq, hfreq)
from hmm_visuals import (
    plot_viterbi_path,
    visualize_state_means)

# ===========================
# 1. PARAMETERS & PATHS
# ===========================

job_id = '_job_source_2'  # Change to avoid overwriting the files

K = 6  # number of HMM states

# Sampling frequency and frequency band (for spectral estimation later)
fs = 200

df_subjects = pd.read_csv("subject_text_files/test.txt", names=["subject"])

# Specify the session to process (here, session '01')
sessions = ["01"]  #, "02", "03", "04", "05"]

# ===========================
# 2. DATA LOADING & CONCATENATION
# ===========================
X_list = []         # list to store each subject's data
indices_list = []   # to store start/end indices for each subject in the concatenated data
start = 0

for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)
    for ses in sessions:        
        pc_type = 'aparc_sub'
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

print(f"Loaded and concatenated data from {df_subjects.shape[0]} subjects.")
print("Data shape:", data.shape)
print("Indices shape:", indices.shape)

# Preprocess the data (e.g., standardisation) as in the glhmm example.
# The preprocess_data function expects (data, indices) and returns the (processed_data, _)
data, _ = preproc.preprocess_data(data, indices, dampen_extreme_peaks=True)

# ===========================
# 3. INITIALISE & TRAIN THE HMM
# ===========================
# Instantiate the standard Gaussian HMM.

hmm = glhmm.glhmm(model_beta='no', K=K, covtype='full')

# Optionally, inspect hyperparameters:
print("Hyperparameters:")
print(hmm.hyperparameters)

# Set a seed for reproducibility
np.random.seed(123)

# Train the HMM.
# Note: The glhmm.train method accepts:
#     X=None, Y=<concatenated data>, indices=<2D array specifying [start, end] for each subject/session>
Gamma, Xi, FE = hmm.train(X=None, Y=data, indices=indices)
