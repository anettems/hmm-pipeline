#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================
Settings Configuration File
===========================

This configuration file defines the settings and parameters needed for the HMM analysis pipeline. 
User should customize the parameters below according to their study design and data processing requirements.

Instructions:
-------------
- Adjust the filtering, ICA, and model parameters according to your analysis needs.
- Specify task names, sessions, runs, and rejection limits based on your dataset.
- Update the state mapping if you are using different naming conventions for states.

"""

#################################
### General Settings
#################################

N_JOBS = 4  # Number of parallel jobs to use for processing


#################################
### Data Acquisition Settings
#################################

# Task and Session Information
task = "eo"  # Task name (e.g., 'eo' for eyes open)
sessions = ["01"]  # Session IDs to process (e.g., ["01", "02"])
run = "01"  # Run numbers to process (e.g., ["01", "02"]) 

# Preprocessing Settings
sfreq = 200  # Sampling frequency for resampling the data (Hz)
baseline = (-0.2, 0)  # Baseline window (in seconds) for epoching

# Rejection Limits for Cleaning Data
reject = dict(mag=4e-12, grad=3000e-13)  # Limits for rejecting bad data segments


#################################
### Filtering and ICA Settings
#################################

# Frequency Range for Filtering
bandpass_fmin, bandpass_fmax = None, 48  # Full range for preprocessing

# ICA Settings
ica_method = "fastica"  # ICA algorithm to use
ica_epoch_tmin, ica_epoch_tmax = -0.2, 3.5  # Time window for ICA artifact detection
lfreq_ica = 1  # High-pass filter cutoff for ICA preprocessing


#################################
### Source Estimation Settings
#################################

# Source Space Settings
spacing = "ico4"  # Source spacing ('ico4' for ~5120 vertices per hemisphere)
proc_scimeg = "raw_meg_tsss_mc_mfilter"  # Processing tag for MEG files


#################################
### HMM Model Parameters
#################################

# Frequency Range for HMM Analysis
lfreq = 0.1  # Low-frequency cutoff (Hz)
hfreq = 48  # High-frequency cutoff (Hz)

# Model Structure
n_labels = 450  # Number of labels in cortical parcellation (e.g., 450 = both hemispheres)
lag = 7 # Lag for time-delay embedding # 3, 7, 10, 20
n_pca = 45  # Variance threshold for PCA dimensionality reduction
K = 6  # Number of states for the HMM
job_id = "_job_source_2"  # Identifier for saving HMM results
pc_type = 'aparc_sub'  # Parcellation type for source extraction


#################################
### State Mapping
#################################

# Mapping of model states to specific labels
state_mapping = {
    'low': 'state3',
    'high': 'state1',
    'gam': 'state2',
    'bl': 'state4'
}

