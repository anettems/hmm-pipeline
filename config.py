"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os

from fnames import FileNames

user = os.environ["USER"]


# Rejection limits
reject = dict(mag=4e-12, grad=3000e-13)  # , eog=150e-6)

# Default baseline window
default_baseline = (-0.2, 0)

# Source spacing
spacing = "ico4"

N_JOBS = 4


###############################################################################
# Folders


# Study path and processed path 
study_path = "/m/nbe/scratch/hmmpipeline/data/"

# scimeg datapath
data_path = os.path.join(study_path, "control_data/")

# path for the processed data
processed_dir = os.path.join(study_path, "processed_data/")

# Individual MRI directory
subjects_dir = os.path.join(study_path, "MRI/")

# Scaled MRI directory
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
# Scimeg data unprocessed
fname.add(
    "raw_sci", "{megbids_dir}/{subject}_ses-{ses}_task-{task}_run-{run}_proc-{proc}_meg.fif"
)
# Maxfiltered data in the scimeg folders
fname.add(
    "raw_sci_mf", "{megbids_dir}/{subject}_ses-{ses}_task-{task}_proc-{proc}.fif"
)
# Raw files in HMM folders
fname.add(
    "raw", "{hmm_bids_dir}/{subject}_ses-{ses}_task-{task}_proc-{proc}.fif"
)


# Emptyroom file for the noise covariance
fname.add(
    "emptyroom_sci", "{hmm_bids_dir}/emptyroom_tsss.fif"
)

# ica decompositions
fname.add(
    "ica", "{hmm_bids_dir}/ica/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-ica.fif"
)


# Source level files
# Noise covariance
fname.add(
    "noise_cov", "{hmm_bids_dir}/noise_cov/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-cov.fif"
)

# Source space
# Individual source space
fname.add(
    'src_ind', '{hmm_bids_dir}/forward/{subject}-{spacing}-individual-src.fif'
)
# fsaverage source space
fname.add(
    'src', '{hmm_bids_dir}/forward/{subject}-{spacing}-src.fif'
)

# Bem solution
fname.add(
    'bem_sol_ind', '{hmm_bids_dir}/forward/{subject}-{ntri}-individual-bem-sol.fif'
)
fname.add(
    'bem_sol', '{hmm_bids_dir}/forward/{subject}-{ntri}-bem-sol.fif'
)

#Forward model
fname.add(
    'fwd', '{hmm_bids_dir}/forward/{subject}_ses-{ses}_task-{task}_{spacing}-fwd.fif'
)



# Trans-file (coregistration)
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


# Annotations
fname.add(
    'annot', "{hmm_bids_dir}/annot/{subject}_ses-{ses}_task-{task}_lfreq-{lfreq}-hfreq-{hfreq}-annot.fif"
)

# hmm-dual
fname.add(
    "hmm_dual", "{processed_dir}/{group_or_single}/{sensor_type}/{subject}/dual_hmm_ses_{session}{job_id}.mat"
)
fname.add(
    "FO_dual", "{processed_dir}/{group_or_single}/{sensor_type}/{subject}/FO{job_id}_ses_{session}.mat"
)
fname.add(
    "SLT_dual", "{processed_dir}/{group_or_single}/{sensor_type}/{subject}/SLT_parameters{job_id}_ses_{session}.mat"
)
fname.add(
    "spetra_true", "{processed_dir}/{group_or_single}/{sensor_type}/{subject}/true_spectra_{session}{job_id}.mat"
)