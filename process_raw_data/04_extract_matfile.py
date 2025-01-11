import mne
import pandas as pd
import os

from settings_hmm_beta import proc_scimeg, task
from config import (fname, subjects_dir, spacing)
from scipy.io import savemat



# Read the peak channel csv
df_subjects = pd.read_csv("subject_text_files/test.txt", names=["subject"])
sessions = ["01"]#, "02", "03", "04", "05"]

proc = proc_scimeg
lfreq = 13
hfreq = 30


# Get labels for FreeSurfer 'aparc_sub' cortical parcellation
labels_parc = mne.read_labels_from_annot(subject = 'fsaverage_sara', parc = 'aparc_sub', subjects_dir=subjects_dir)
lh_parc = [ lab for lab in labels_parc if lab.hemi == 'lh']

# Get source spave
fname_src = fname.src(hmm_bids_dir=fname.hmm_bids_dir(subject='fsaverage_sara', ses='01'), subject='fsaverage_sara', spacing=spacing)
src = mne.read_source_spaces(fname_src)


for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)

    for ses in sessions:
        print('Processing session: ', ses)

        if not os.path.exists(fname.megprocessed_dir(subject=subject, ses=ses + '/source-ave/')):
            os.makedirs(fname.megprocessed_dir(subject=subject, ses=ses + '/source-ave/'))

        # Load the fsaverage source estimate
        #stc_fname_ave = fname.stc_average_nht(subject=subject, ses = ses, task= task)
        stc_fname_ave = fname.stc(subject=subject, ses = ses, task= task, lfreq=lfreq, hfreq=hfreq)
        stc_ave = mne.read_source_estimate(stc_fname_ave)

        
        label_ts_fsaverage = mne.extract_label_time_course(
            stc_ave,labels_parc,src, mode="pca_flip", allow_empty=True
        ) 
        # Save data mat
         # Make the data to modulo 2 => otherwise gives error later
        if label_ts_fsaverage.shape[1] % 2 == 1:
            label_ts_fsaverage = label_ts_fsaverage[:,0:-1]
        
        
        savemat(fname.data_mat(
                    subject=subject,
                    ch_type="source-ave",
                    ses=ses,
                    l_freq=str(lfreq),
                    h_freq=str(hfreq),
                    ),
                    dict(x=label_ts_fsaverage.T),
                )
