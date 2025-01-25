"""
Create source space and compute forward solution

Author: mli
"""
from __future__ import print_function

import os
import pandas as pd
import mne
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from config import (fname, subjects_dir_ave, spacing, N_JOBS)
from settings_hmm_beta import task

if spacing=='ico4':
    ntri=5120
    ico=4
elif spacing=='ico5':
    print('ico5')
    ntri=20484 #check or 20480
    ico=5


# Read the peak channel csv
df_subjects = pd.read_csv("subject_text_files/test.txt", names=["subject"])
sessions = ["01"]#, "03", "04", "05"]
lfreq = 0.1
hfreq = 48


for i, row in df_subjects.iterrows():
    subject = row["subject"]

    # Saves bem model 
    for ses in sessions:
        print('Processing session: ', ses)
 
        # Create folder if it does not exist
        if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/forward/'):  
            os.makedirs(fname.hmm_bids_dir(subject=subject,ses=ses) + '/forward/')
        
        megdir=fname.hmm_bids_dir(subject=subject, ses=ses)
        
        if os.path.exists(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing)):
            subject_src = mne.read_source_spaces(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing))
            bem = mne.read_bem_solution(fname.bem_sol(hmm_bids_dir=megdir, subject=subject,ntri=ntri))
        else:
            # Create source space in individual subject 
            subject_src = mne.setup_source_space(subject=subject, spacing=spacing,
                                        subjects_dir=subjects_dir_ave,
                                        n_jobs=N_JOBS, add_dist=True)
            mne.write_source_spaces(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing), subject_src, overwrite=True)

            # Create BEM model
            bem_model = mne.make_bem_model(subject=subject, ico=ico, subjects_dir=subjects_dir_ave,
                                conductivity=(0.3,))
            if bem_model[0]['ntri'] == ntri:            
                bem = mne.make_bem_solution(bem_model)
                mne.write_bem_solution(fname.bem_sol(hmm_bids_dir=megdir, subject=subject,ntri=bem_model[0]['ntri']),bem,overwrite=True)
            else:
                raise ValueError('ntri should be %d' % (ntri))

        #Create the forward model, coregistration required (trans_file)
        # Use first run in one of the tasks for forward model
        mfilter_path = fname.raw(
            subject=subject,
            ses=ses,
            task=task,
            proc="raw_meg_tsss_mc_mfilter",
        )

        info = mne.io.read_info(mfilter_path)

        fwd = mne.make_forward_solution(
            info,
            trans=fname.trans(subject= subject, ses=ses, task=task),
            src=subject_src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=0,
            n_jobs=N_JOBS
            )
        mne.write_forward_solution(fname.fwd(subject=subject,ses=ses,task=task, spacing=spacing), fwd,
                        overwrite=True)
