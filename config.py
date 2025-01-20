"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os

from fnames import FileNames

# Global variables for configuration
user = os.environ["USER"]


# Rejection limits
reject = dict(mag=4e-12, grad=3000e-13)  # , eog=150e-6)

# Default baseline window
default_baseline = (-0.2, 0)

# Source spacing
spacing = "ico4"

N_JOBS = 4

# Study path and processed path (different that studypath)
study_path = "/m/nbe/project3/hmmpipeline/"  # Biomag scimeg
processed_path = "/m/nbe/project3/hmmpipeline/"  # Biomag HMM-MAR

###############################################################################
# Folders (TODO: decide on folder structure)

# scimeg datapath
data_path = os.path.join(study_path, "control_data/")

# Individual MRI directory
subjects_dir = os.path.join(study_path, 'MRI/')

# path for the processed data (in HMM-MAR)
processed_dir = os.path.join(processed_path, "processed_data/")

# Scaled MRÍ directory
subjects_dir_ave = os.path.join(processed_dir, 'MRI/')

def run_configurations():
    
    
    ###############################################################################
    # Subject-codes for freesurfing
    
    subject_info = {"sub-16C": "sub-16C/"}
    # or: read all subjects from folder-names
    # or: place subjects in task-specific settings
    
    ###############################################################################
    # Templates for filenames
    fname = FileNames()
    
    # Some directories<<<<<<<<<<<<<<<<<<<<<<<<<
    fname.add("data_path", data_path)
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
    
    
    # Save .mat file
    fname.add(
        "ch_names_mat",
        "{megprocessed_dir}/{ch_type}/{subject}_ch_names_{n_sensors}_sens_{lr_side}.mat",
    )
    # fname.add('data_mat','{megprocessed_dir}/{ch_type}/{subject}_peak_ch_data_{lr_side}.mat' )
    fname.add(
        "data_full_freq",
        "{megprocessed_dir}/{ch_type}/{subject}_mat_data_{lr_side}_{n_sensors}_sens_{l_freq}_{h_freq}_ff.mat",
    )
    fname.add(
        "data_mat",
        "{megprocessed_dir}/{ch_type}/{subject}_lfreq-{l_freq}_hfreq-{h_freq}_mat.mat",
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
    
    return fname


    
if __name__ == "__main__":
    print("Running config.py as a standalone script:")
    fname = run_configurations()