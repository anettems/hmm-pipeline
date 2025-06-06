from config import fname
from settings_hmm_beta import (task, sessions, lfreq, hfreq)

import pandas as pd
import mne
import os

print("\n #### Starting noise covariance calculation. #### \n")

# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])

for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)
    for ses in sessions:         
        print('Processing session: ', ses)
        # Get emptyroom data for the noise covariance
        emptyroom_fname = fname.emptyroom_sci(subject=subject,ses=ses)

        if os.path.exists(emptyroom_fname):
            raw=mne.io.read_raw_fif(emptyroom_fname).load_data().filter(lfreq,hfreq)
            
            # Calculate noise covariance on baseline time period
            # Check rank estimation!!!!
            noise_cov = mne.compute_raw_covariance(raw,rank="auto",picks='meg')
            noise_cov.plot(raw.info)
            # Save noise covariance matrix

            # Create folder if it does not exist
            if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/noise_cov/'):  
                os.makedirs(fname.hmm_bids_dir(subject=subject,ses=ses) + '/noise_cov/')

            mne.write_cov(fname.noise_cov(subject=subject, ses=ses, task=task, lfreq=lfreq, hfreq=hfreq), noise_cov, overwrite=True)

print("\n #### Covariance calculations finalized. #### \n")