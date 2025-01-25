import pandas as pd
import mne
import os


def compute_covariance(subject_file_path, sessions=None, task="restEO", lfreq=0.1, hfreq=48, fname=None):
    """
    Compute noise covariance for each subject in the provided file.

    Args:
        subject_file_path (str): Path to the text file containing subject IDs.
        sessions (list): List of session identifiers. Default is ["01"].
        task (str): Passed from settings_hmm_beta. Default is "restEO".
        lfreq (float): Low cutoff frequency for the bandpass filter. Default is 0.1.
        hfreq (float): High cutoff frequency for the bandpass filter. Default is 48.
        fname (FileNames): File name object storing paths.

    Returns:
        None
    """
    if sessions is None:
        sessions = ["01"]

    # Read the subject file
    df_subjects = pd.read_csv(subject_file_path, names=["subject"])

    for i, row in df_subjects.iterrows():
        
        subject = row["subject"]
        
        for ses in sessions:                
            # Get emptyroom data for the noise covariance
            emptyroom_fname = fname.emptyroom_sci(subject=subject,ses=ses)

            if os.path.exists(emptyroom_fname):
                raw=mne.io.read_raw_fif(emptyroom_fname).load_data().filter(lfreq,hfreq)
                
                # Calculate noise covariance on baseline time period
                # Check rank estimation!!!!
                noise_cov = mne.compute_raw_covariance(raw,rank="auto",picks='meg')#, method='auto') 
                noise_cov.plot(raw.info)
                # Save noise covariance matrix

                # Create folder if it does not exist
                if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/noise_cov/'):  
                    os.makedirs(fname.hmm_bids_dir(subject=subject,ses=ses) + '/noise_cov/')
                
                # Save noise covariance matrix
                cov_path = fname.noise_cov(
                    subject=subject, ses=ses, task=task, lfreq=lfreq, hfreq=hfreq
                )
                mne.write_cov(cov_path, noise_cov, overwrite=True)

                print(f"Noise covariance saved to: {cov_path}")
            else:
                print("emptyroom_fname path does not exist")


if __name__ == "__main__":
    # Example usage
    subject_file = "subject_text_files/test.txt"
    compute_covariance(subject_file)
