"""
====================
Configuration Script
====================

This configuration file defines all the file paths and directory structures required for the study. 
Users must customize the file paths to match the locations of their specific study data. 

Instructions:
-------------
- Modify the variables `study_path`, `data_path`, and `processed_dir` to point to your dataset's root folder, 
  where your raw and processed data are stored.
- Ensure that the paths to MRI data directories (`subjects_dir` and `subjects_dir_ave`) are correctly set.
- The `subjects_file` should point to a text file listing all subjects to be processed.

This configuration file acts as a centralized manager for all file paths used throughout the pipeline, 
allowing efficient data handling and consistency across different processing steps.

"""


import os

from fnames import FileNames

user = os.environ["USER"]

###############################################################################
# Folders


# Study path and processed path 
study_path = "/m/nbe/scratch/hmmpipeline/data/"
processed_path = "/m/nbe/scratch/hmmpipeline/data/"

# scimeg datapath
data_path = os.path.join(study_path, "control_data/")

# Path for the processed data
processed_dir = os.path.join(processed_path, "processed_data/")

# Individual MRI directory
subjects_dir = os.path.join(study_path, "MRI/")

# Directory that contains the scaled MRI data of the template subject
subjects_dir_ave = os.path.join(processed_dir, "MRI/")


###############################################################################

# Txt file which contains all subjects
subjects_file = os.path.join(study_path, "subject_text_files/test.txt")

###############################################################################
# Templates for filenames
fname = FileNames()

# Some directories<<<<<<<<<<<<<<<<<<<<<<<<<
fname.add("data_path", data_path)
fname.add("subjects_txt", subjects_file)

fname.add("megbids_dir", "{data_path}/{subject}/ses-{ses}/meg")
fname.add("processed_dir", processed_dir)
fname.add("megprocessed_dir", "{processed_dir}/{subject}/ses-{ses}/")

fname.add("hmm_bids_dir", "{processed_dir}/{subject}/ses-{ses}/meg")




# Sensor-level files
# Scimeg data unprocessed - IS THIS USED?
fname.add(
    "raw_sci", "{megbids_dir}/{subject}_ses-{ses}_task-{task}_run-{run}_proc-{proc}_meg.fif"
)
# Maxfiltered data in the scimeg folders
# Used in: make_annotations.py, make_ica.py  & 02_forward.py & 03_inverse.py
fname.add(
    "raw_sci_mf", "{megbids_dir}/{subject}_ses-{ses}_task-{task}_proc-{proc}.fif"
)
# Maxfiltered raw files in HMM folders
# Originally used in: make_annotations.py & make_ica.py & 02_forward.py - IS THIS NEEDED?
fname.add(
    "raw", "{hmm_bids_dir}/{subject}_ses-{ses}_task-{task}_proc-{proc}.fif"
)


# Annotations (Used in: make_annotations.py)
fname.add(
    'annot', "{hmm_bids_dir}/annot/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-annot.fif"
)

# Ica decompositions (Used in: make_ica.py)
fname.add(
    "ica", "{hmm_bids_dir}/ica/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-ica.fif"
)


# Emptyroom file for the noise covariance
# Used in: 01_covariance.py - DOES THIS NEED TO BE AT PROCESSED FOLDER? megbids_dir instead?
fname.add(
    "emptyroom_sci", "{hmm_bids_dir}/emptyroom_tsss.fif"
)


# Source level files
# Noise covariance (Used in: 01_covariance.py)
fname.add(
    "noise_cov", "{hmm_bids_dir}/noise_cov/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-cov.fif"
)

# Source space (Used in: create_sourcespace.py, 02_forward.py)
fname.add(
    'src', '{hmm_bids_dir}/forward/{subject}-{spacing}-src.fif'
)

# Bem solution (Used in: create_sourcespace.py, 02_forward.py)
fname.add(
    'bem_sol', '{hmm_bids_dir}/forward/{subject}-{ntri}-bem-sol.fif'
)

#Forward model (Used in: create_sourcespace.py, 02_forward.py)
fname.add(
    'fwd', '{hmm_bids_dir}/forward/{subject}_ses-{ses}_task-{task}_{spacing}-fwd.fif'
)


# Trans-file (coregistration) (Used in: create_sourcespace.py, 02_forward.py)
fname.add('trans',
    '{hmm_bids_dir}/{subject}_ses-{ses}_{task}-coreg-trans.fif'
)

# inverse solution
fname.add(
    'inv', '{hmm_bids_dir}/inverse/{subject}_ses-{ses}_task-{task}-{spacing}_lfreq-{lfreq}-hfreq-{hfreq}-inv.fif'
)

# Source estimate
fname.add(
    'stc', '{hmm_bids_dir}/stcs/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-stc'
)


# Save .npy file
fname.add(
    "data_npy",
    "{megprocessed_dir}/{pc_type}/{subject}_lfreq-{l_freq}_hfreq-{h_freq}_npy.npy",
)


# CSVs
fname.add(
    "character_csv", "{processed_dir}/{group_or_single}/{character}/{filename}.csv"
)


# Source estimate characteristics
fname.add(
    'stc_char', '{hmm_bids_dir}/stcs_char/{subject}_ses-{ses}_task-{task}_char-{char}_stc'
)


# TDE-HMM object and the key outputs 
fname.add("tde_hmm_ob", "{processed_dir}/group/tde-hmm/sensors_concat_group{job_id}.npz")

# TDE-HMM object folder
fname.add("tde_hmm_path", "{processed_dir}/group/tde-hmm/")
