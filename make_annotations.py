"""
make_annotations.py

Purpose: This script identifies and annotates bad segments (e.g. artifacts, noise) in MEG data. 
It loads MEG files, filters for the selected band, and provides 
an interactive interface for marking bad intervals. The annotations are saved as '.fif' 
files for use in later analysis steps to exclude noisy data.
¨
"""

from __future__ import print_function
import mne
import os
import pandas as pd

from config import fname
from settings_hmm_beta import (proc_scimeg, task, sessions, lfreq, hfreq)


# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])


# Loop for processing all subjects and all sessions
for i, row in df_subjects.iterrows():
    subject = row["subject"]

    for ses in sessions:
        print('Processing session: ', ses)

        # Read the info structure
        mfilter_path = fname.raw(
            subject=subject,
            ses=ses,
            task=task,
            proc=proc_scimeg,
            )
        
        
        # Filtering of raw MEG data
        raw = mne.io.read_raw_fif(mfilter_path).load_data().filter(lfreq,hfreq)
        raw.plot(block=True)
        
        # Annotation: mark artifacts or noisy intervals in the data

        interactive_annot = raw.annotations

        if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/annot/'):
            os.makedirs(fname.hmm_bids_dir(subject=subject, ses=ses) + '/annot/' )
            
        # Saving annotations in .fif files
        interactive_annot.save( fname.annot(subject=subject,ses=ses,task = task,lfreq = lfreq, hfreq = hfreq), overwrite=True )


