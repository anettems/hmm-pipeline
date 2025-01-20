#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for running preprocessing of the data
"""

import config
from settings_hmm_beta import HMMBetaSettings
from process_raw_data.create_test_file import create_subject_text_file
from process_raw_data.covariance_01 import compute_covariance

# Run configurations
settings = HMMBetaSettings()
fname = config.run_configurations()
subject_file = create_subject_text_file()


# Update lfreq and hfreq values
# settings.update_settings(lfreq=0.5, hfreq=60)

### Preprocessing the data ###

# Call the function with custom sessions and frequency range
compute_covariance(subject_file_path=subject_file, sessions=["01"], task=settings.task, lfreq=settings.lfreq, hfreq=settings.hfreq, fname=fname)
print("Covariance_01 finished")
