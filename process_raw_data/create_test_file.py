"""

This script automates the creation of a folder and a text file for preprocessing MEG data.

- The script extracts subject IDs from a specified directory (`data_dir`) where each subject's data is stored in subfolders.
- It creates a target folder (`subject_text_files`) in the specified path (`base_path`) if it doesn't already exist.
- A text file (`test.txt`) is created in the target folder, containing a list of subject IDs extracted from `data_dir`.
- The output can be used for later scripts that require a list of subjects for processing.

"""



import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config 

def create_subject_text_file():
    """
    Creates a text file with subject IDs extracted from data_dir.
    """
    # Get the directory of the current script
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Path to the data directory with folders per subject
    data_dir = config.data_path

    # Defining the target directory for the text file
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

    print(f"Created {output_file} with {len(subjects)} subjects at {file_path}.")
    
    # Return the path
    return file_path


if __name__ == "__main__":
    create_subject_text_file()