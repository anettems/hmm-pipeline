# -*- coding: utf-8 -*-
"""
Coordinate frame alignment.
Works with new mne (0.20.0) and new mayavi (4.7.1), still freezes sometimes and needs restarting

In terminal, before starting e.g. spyder
ml freesurfer
export FREESURFER_HOME=/work/modules/Ubuntu/14.04/amd64/t314/freesurfer/6.0.0

"""
import mne

from config import subjects_dir

subject = 'fsaverage_sara' # 'sub-16C'

## Coregistration 
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)

# 1. Load digitalization source (head shape source) from any tsss file
# 2. Edit Fiducaials (left side panel)
# 3. Save fiducials (default)
# 4. Fit fiducials
# 5. Fit ICP (head shape)
# 6. Omit outliers and fit ICP again
# 7. When happy with co-registation, save into the forward folder (sub-01_ses-01_coreg-trans.fif/sub-01_ses-01_fsaverage-coreg-trans.fif)
# 7. When happy with co-registation, save into the forward folder (sub-01_ses-01_source-coreg-trans.fif/sub-01_ses-01_fsaverage-source-coreg-trans.fif)
# 7. When happy with co-registation, save into the forward folder (sub-01_ses-01_source-coreg-trans.fif/sub-01_ses-01_rightCKC3Hz-coreg-trans.fif)
# 8. Run separately for each HPI digitalization (ses-01, ses-02)


## Check coregistration

# info = mne.io.read_info(test_data)
# ## surfaces: 'head-dense' = high resolution, 'head' = low reso, 'brain' = pial 
# mne.viz.plot_alignment(info, coreg, subject=subject, dig=True, meg=['helmet', 'sensors'], subjects_dir=subjects_dir, surfaces='head-dense')