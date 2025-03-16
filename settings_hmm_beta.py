#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========
Config file
===========

Configuration parameters for naming.
"""

# Task
task = "eo" 

# Session
sessions = ["01"]  # ['01']#,'02']

# Runs
runs = ["01"]  # ,'02']

# Baseline window
baseline = (-0.2, 0)

# Filter settings
bandpass_fmin, bandpass_fmax = None, 48

# ICA method
ica_method = "fastica"

# ICA time window (wrt stimulus) to find artefact from
ica_epoch_tmin, ica_epoch_tmax = -0.2, 3.5

# proc
proc_scimeg = "raw_meg_tsss_mc_mfilter"


# Model parameters:
lfreq = 0.1
lfreq_ica = 1
hfreq = 48
sfreq = 200 # Sampling frequency
n_labels = 450 # left: 226, right: 224, both: 450
lag = 15
n_pca = 0.9
K = 6  # number of HMM states
job_id = "_job_source_2"
pc_type = 'aparc_sub'

# State mapping
state_mapping = {'low': 'state3',
                 'high': 'state1',
                 'gam': 'state2',
                 'bl': 'state4'}
