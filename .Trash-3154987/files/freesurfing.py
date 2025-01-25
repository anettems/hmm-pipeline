"""



"""
import os
import mne


from config import (subjects_dir)



def ensure_dir(directory):
  if not os.path.exists(directory):
      os.makedirs(directory)


# Set up mri-images for analysis with mne
for subject in {'sub-16C': 'sub-16C/'}:
    ensure_dir(subjects_dir +  subject + '/bem/')
    
    ## BEM
    mne.bem.make_watershed_bem(subject, subjects_dir, overwrite=True)
    # Alternatively, use:
    #os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir + ' --overwrite')
    
    ## High-resolution surfaces
    os.system('mne make_scalp_surfaces -s ' + subject + ' -d ' + subjects_dir +' --overwrite --force')
    #mne.bem.make_scalp_surfaces(subject, subjects_dir) #works only with version 0.24
    
    ## Set up COR-file   
    os.system('mne_setup_mri --subject ' + subject + ' --overwrite')
    