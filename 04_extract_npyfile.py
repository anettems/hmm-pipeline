import mne
import pandas as pd
import os

from settings_hmm_beta import (proc_scimeg, task, lfreq, hfreq, sessions, pc_type)
from config import (fname, subjects_dir, spacing)
import numpy as np


# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])

proc = proc_scimeg

# Get labels for FreeSurfer 'aparc_sub' cortical parcellation
labels_parc = mne.read_labels_from_annot(subject = 'fsaverage_sara', parc = pc_type, subjects_dir=subjects_dir)
lh_parc = [ lab for lab in labels_parc if lab.hemi == 'lh']

# Get source space from the file corresponding to the template subject
fname_src = fname.src(hmm_bids_dir=fname.hmm_bids_dir(subject='fsaverage_sara', ses='01'), subject='fsaverage_sara', spacing=spacing)
src = mne.read_source_spaces(fname_src)


for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)

    for ses in sessions:
        print('Processing session: ', ses)
        
        directory = os.path.join(fname.megprocessed_dir(subject=subject, ses=ses), pc_type)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load the source estimate file for the given subject and session
        #stc_fname_ave = fname.stc_average_nht(subject=subject, ses = ses, task= task)
        stc_fname_ave = fname.stc(subject=subject, ses = ses, task= task, lfreq=lfreq, hfreq=hfreq)
        stc_ave = mne.read_source_estimate(stc_fname_ave)

        # Extract time courses from the source estimate for each label
        label_ts_fsaverage = mne.extract_label_time_course(
            stc_ave,labels_parc,src, mode="pca_flip", allow_empty=True
        ) 
        
        # Ensure the data has an even number of samples to prevent errors
        if label_ts_fsaverage.shape[1] % 2 == 1:
            label_ts_fsaverage = label_ts_fsaverage[:,0:-1]
            
        # Save the extracted time courses of source estimates as a .npy file
        npy_file_path = fname.data_npy(
            subject=subject,
            pc_type=pc_type,
            ses=ses,
            l_freq=str(lfreq),
            h_freq=str(hfreq),
        )
        np.save(npy_file_path, label_ts_fsaverage.T)
        
     