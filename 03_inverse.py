#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Calculate MNE inverse
"""
from __future__ import print_function
import mne
import os
import pandas as pd
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from functions import reject_bad_segs
from config import fname
from settings_hmm_beta import (proc_scimeg, task, lfreq, lfreq_ica, hfreq, sfreq, run, sessions, spacing)
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

print("\n #### Starting 03_inverse.py #### \n")

# Read the subject txt file
df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])


for i, row in df_subjects.iterrows():
    subject = row["subject"]
    print('Processing subject: ', subject)

    for ses in sessions:
        print('Processing session: ', ses)

        # Read the forward model, one model has been created for each session
        fwd = mne.read_forward_solution(fname.fwd(subject=subject,ses=ses,task=task, run=run, spacing=spacing))
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)

        # Read the noise covariance matrix
        noise_cov = mne.read_cov(fname.noise_cov(subject=subject, ses=ses, task=task, lfreq=lfreq, hfreq=hfreq))

        # Read the info structure
        mfilter_path = fname.raw_sci_mf(
            subject=subject,
            ses=ses,
            task=task,
            proc=proc_scimeg,
            )
        
        
        raw = mne.io.read_raw_fif(mfilter_path).load_data().filter(1,40)
        
        # Load ica and apply
        ica = mne.preprocessing.read_ica( fname.ica(subject=subject,ses=ses,task=task,lfreq=lfreq_ica,hfreq=hfreq) )
        ica.apply(raw)

        # Read annotation and reject
        bad_annotation = mne.read_annotations( fname.annot(subject=subject,ses=ses,task = task,lfreq = lfreq, hfreq = hfreq) )
        raw.set_annotations(bad_annotation)

        # Then do the rest of the filtering and pick gradiometers
        raw.filter(lfreq,hfreq).resample(sfreq=sfreq).pick_types(meg='grad')
        info = raw.info

        # Reject bad segments from raw data
        raw = reject_bad_segs(raw)

        raw.plot()

        # Compute inverse operator
        pick_ori = None # 'normal' or None
        snr = 1.0 
        lambda2 = 1.0 / snr ** 2
        method = "MNE"

        # Create inverse operator in individual subject
        inverse_operator = make_inverse_operator(info = info, 
                                            forward = fwd, 
                                            noise_cov = noise_cov, 
                                            fixed=True,
                                            loose = 0, 
                                            depth = 0.8,
                                            rank = {'grad':72}) 
        if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/inverse/'):
            os.makedirs(fname.hmm_bids_dir(subject=subject, ses=ses) + '/inverse/' )
        
        mne.minimum_norm.write_inverse_operator(fname.inv(subject=subject,
                                        ses=ses,
                                        task=task,
                                        spacing = spacing,
                                        lfreq=lfreq,
                                        hfreq=hfreq), inverse_operator, overwrite=True)

        # Compute inverse
        if not os.path.exists(fname.hmm_bids_dir(subject=subject, ses=ses) + '/stcs/'):
                os.makedirs(fname.hmm_bids_dir(subject=subject, ses=ses) + '/stcs/')

        stc = apply_inverse_raw(raw, inverse_operator, lambda2, method, pick_ori=pick_ori)
        
        #stc.save(fname.stc_average_nht(subject=subject, ses = ses, task=task), overwrite=True)
        stc.save(fname.stc(subject=subject, ses = ses, task=task, lfreq=lfreq, hfreq=hfreq), overwrite=True)
        
print("\n #### Finalized 03_inverse.py #### \n")