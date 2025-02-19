"""
Create source space for template subject.

Must be run once before 04_extract_npyfile
"""

from __future__ import print_function

import os
import mne
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from config import (fname, subjects_dir_ave, spacing, N_JOBS)
from settings_hmm_beta import sessions

if spacing=='ico4':
    ntri=5120
    ico=4
elif spacing=='ico5':
    print('ico5')
    ntri=20484 #check or 20480
    ico=5


# Create the source space for the template subject
subject = 'fsaverage_sara' 

for ses in sessions:
    # Create folder if it does not exist
    if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/forward/'):  
        os.makedirs(fname.hmm_bids_dir(subject=subject,ses=ses) + '/forward/')
    
    megdir=fname.hmm_bids_dir(subject=subject, ses=ses)
    
    if os.path.exists(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing)):
        subject_src = mne.read_source_spaces(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing))
        bem = mne.read_bem_solution(fname.bem_sol(hmm_bids_dir=megdir, subject=subject,ntri=ntri))
    else:
        subject_src = mne.setup_source_space(subject=subject, spacing=spacing,
                                    subjects_dir=subjects_dir_ave,
                                    n_jobs=N_JOBS, add_dist=True)
        mne.write_source_spaces(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing), subject_src, overwrite=True)
        print("Source space for template subject created.")