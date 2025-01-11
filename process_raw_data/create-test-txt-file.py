"""
### create-test-txt-file ###

This script automates the creation of a folder and a text file for preprocessing MEG data.

- The script extracts subject IDs from a specified directory (`data_dir`) where each subject's data is stored in subfolders.
- It creates a target folder (`subject_text_files`) in the specified path (`base_path`) if it doesn't already exist.
- A text file (`test.txt`) is created in the target folder, containing a list of subject IDs extracted from `data_dir`.
- The output can be used for later scripts that require a list of subjects for processing.

Author: Anette Sarivuo
Date: 11.11.2025
"""



import os

# Path to the data directory with folders per subject
data_dir = "/m/nbe/project3/hmmpipeline/control_data/" 
# data_dir will be used for assigning subject IDs to the text file for later pre-processing
# Directory in data_dir must contain folders with subject IDs which start with "sub-" per each subject

# Defining the target directory for the text file
base_path = "/m/nbe/project3/hmmpipeline/codes_st/scratch/process_raw_data/"
folder_name = "subject_text_files"
target_folder = os.path.join(base_path, folder_name)

# Creating the folder if it doesn't exist yet
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"Folder created at: {target_folder}")
else:
    print(f"Folder already exists at: {target_folder}")

# Defining the name and path for the text file
output_file = "test.txt"
file_path = os.path.join(target_folder, output_file)

# Extracting subject IDs from data_dir for test.txt
subjects = [name for name in os.listdir(data_dir) if name.startswith("sub-")]

# Write to text file
with open(file_path, "w") as f:
    for subject in subjects:
        f.write(f"{subject}\n")

print(f"Created {output_file} with {len(subjects)} subjects.")
