"""
===========================
Create source space for template subject
===========================

This script generates the source space for the template subject and saves it to disk. 
- It must be executed once before running '04_extract_npyfile.py'.
- The generated source space serves as a standardized template, ensuring that data from all subjects can 
  be aligned and compared within the same anatomical framework.
- Without this pre-generated source space, '04_extract_npyfile.py' would not be able to extract regional 
  time courses properly.
"""


from __future__ import print_function

import os
import mne
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from config import (fname, subjects_dir_ave)
from settings_hmm_beta import (spacing, N_JOBS)

print("\n #### Starting source space creation for the template subject. #### \n")

if spacing=='ico4':
    ntri=5120
    ico=4
elif spacing=='ico5':
    print('ico5')
    ntri=20484 #check or 20480
    ico=5


# Create the source space for the template subject
subject = 'fsaverage_sara' 
ses = "01"

# Create folder if it does not exist
if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/forward/'):  
    os.makedirs(fname.hmm_bids_dir(subject=subject,ses=ses) + '/forward/')
    print("Forward directory created for template subject.")

megdir=fname.hmm_bids_dir(subject=subject, ses=ses)

if os.path.exists(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing)):
    print("Source space file already found for template subject. Reading to verify.")
    subject_src = mne.read_source_spaces(fname.src(hmm_bids_dir=megdir, subject=subject,spacing=spacing))
    print("Source space read and verified.")
    
else:
    subject_src = mne.setup_source_space(subject=subject, 
                                         spacing=spacing,
                                         subjects_dir=subjects_dir_ave,
                                         n_jobs=N_JOBS, 
                                         add_dist=True)
    mne.write_source_spaces(fname.src(hmm_bids_dir=megdir,
                                      subject=subject,
                                      spacing=spacing), subject_src, overwrite=True)
    
    print("Source space for template subject created.")