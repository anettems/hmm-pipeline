import mne
import pandas as pd
import os

from config import (fname)
from settings_hmm_beta import (proc_scimeg, task, sessions, lfreq_ica, hfreq, ica_method)

lfreq = lfreq_ica

# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])


# Make run ICA function
def run_ica(method, raw_or_epochs, n_components = 15, fit_params=None):

    ica = mne.preprocessing.ICA(n_components=n_components, method=method, 
                                fit_params=fit_params,
              random_state=97, max_iter=400)
    ica.fit(raw_or_epochs, picks = ['meg'])
    
    return ica


for i_ses, session in enumerate(sessions):
    print("SESSION: ", session)
    for i_sub, row in df_subjects.iterrows():
        subject = row["subject"]
        print(subject)
        
        mfilter_path = fname.raw(
            subject=subject,
            ses=session,
            task=task,
            proc=proc_scimeg,
        )
        print(mfilter_path)

        if os.path.exists(mfilter_path):  # and not os.path.exists(op_path):

            print(subject)

            # Load the raw data
            raw = mne.io.read_raw_fif(mfilter_path).load_data()

            raw.filter(lfreq,hfreq).resample(sfreq=200)
            
            ica=run_ica(ica_method, raw, n_components=50)
            ica.plot_components()
            ica.plot_sources(raw, block=True)

            if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=session) + '/ica/'):
                os.makedirs(fname.hmm_bids_dir(subject=subject, ses=session) + '/ica/' )
            
            ica.save( fname.ica(subject=subject,ses=session,task = task,lfreq = lfreq, hfreq = hfreq), overwrite = True )
            print("ICA saved")