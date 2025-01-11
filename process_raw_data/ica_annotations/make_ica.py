from config import fname

import mne
import pandas as pd
import os


# Read the peak channel csv
df_subjects = pd.read_csv("subject_text_files/test.txt", names=["subject"])
sessions = ["01"]#, "03", "04"]#, "05"]
lfreq = 1
hfreq = 40

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
        
        #if subject != 'sub-15':
        mfilter_path = fname.raw(
            subject=subject,
            ses=session,
            task="rest",
            proc="raw_meg_tsss_mc_mfilter",
        )

        if os.path.exists(mfilter_path):  # and not os.path.exists(op_path):

            print(subject)

            # Load the raw data
            raw = mne.io.read_raw_fif(mfilter_path).load_data()

            raw.filter(lfreq,hfreq).resample(sfreq=200)
            
            ica=run_ica('fastica', raw, n_components=50)
            ica.plot_components()
            ica.plot_sources(raw, block=True)

            if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=session) + '/ica/'):
                os.makedirs(fname.hmm_bids_dir(subject=subject, ses=session) + '/ica/' )
            
            ica.save( fname.ica(subject=subject,ses=session,task = 'rest',lfreq = lfreq, hfreq = hfreq), overwrite = True )

