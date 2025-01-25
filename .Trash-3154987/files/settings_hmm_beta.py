#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========
Config file
===========

Configuration parameters for naming.
"""


class HMMBetaSettings:
    def __init__(self):
        # Task
        self.task = "restEO"

        # Session
        self.sessions = ["01", "02"]  # Default sessions

        # Runs
        self.runs = ["01"]  # Default runs

        # Baseline window
        self.baseline = (-0.2, 0)

        # Filter settings
        self.bandpass_fmin = None
        self.bandpass_fmax = 48

        # ICA method
        self.ica_method = "fastica"

        # ICA time window (relative to stimulus) to find artefact from
        self.ica_epoch_tmin = -0.2
        self.ica_epoch_tmax = 3.5

        # Processing method
        self.proc_scimeg = "raw_meg_tsss_mc"

        # Model parameters
        self.lfreq = 0.1
        self.hfreq = 48
        self.n_labels = 450  # left: 226, right: 224, both: 450
        self.lag = 8
        self.job_id = "_job_source_2"
        self.ch_type = 'source-ave'
        self.side = "both"

        # State mapping
        self.state_mapping = {
            'low': 'state3',
            'high': 'state1',
            'gam': 'state2',
            'bl': 'state4'
        }

    def display_settings(self):
        """Display all settings in a structured format."""
        print("HMM Beta Settings:")
        print(f"  Task: {self.task}")
        print(f"  Sessions: {self.sessions}")
        print(f"  Runs: {self.runs}")
        print(f"  Baseline: {self.baseline}")
        print(f"  Bandpass Filter: {self.bandpass_fmin} - {self.bandpass_fmax} Hz")
        print(f"  ICA Method: {self.ica_method}")
        print(f"  ICA Epoch Time Window: {self.ica_epoch_tmin} to {self.ica_epoch_tmax} seconds")
        print(f"  Processing Method: {self.proc_scimeg}")
        print(f"  Model Parameters:")
        print(f"    Low Frequency: {self.lfreq} Hz")
        print(f"    High Frequency: {self.hfreq} Hz")
        print(f"    Number of Labels: {self.n_labels}")
        print(f"    Lag: {self.lag}")
        print(f"    Job ID: {self.job_id}")
        print(f"    Channel Type: {self.ch_type}")
        print(f"    Side: {self.side}")
        print(f"  State Mapping: {self.state_mapping}")

    def update_settings(self, **kwargs):
        """Update settings dynamically using keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid setting.")