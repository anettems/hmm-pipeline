"""
plot_hmm.py

This script loads a trained TDE-HMM model output from a .npz file and generates a set of plots
to visualize the group-level brain dynamics.

### It fetches the TDE-HMM model based on job_id specified in settings_hmm_beta.py.

### Visualized plots:
- Viterbi state sequences per subject
- Transition probability matrices
- State-specific functional connectivity (covariance matrices)
- State mean activation patterns
- Fractional occupancy and switching rate per subject
- Mean state lifetimes
- Statewise power spectra (per parcel and averaged across parcels)

### Make sure to download the needed plots manually.
"""

from hmm_visuals import (
    plot_viterbi_path,
    plot_transition_probabilities,
    plot_state_covariances,
    plot_fractional_occupancy,
    plot_switching_rate,
    plot_state_lifetimes,
    plot_statewise_spectra,
    plot_statewise_average_spectra,
    plot_state_means,
    plot_statewise_average_spectra_one_figure)
from config import fname
import numpy as np
from settings_hmm_beta import job_id

npz_file_path = fname.tde_hmm_ob(job_id=job_id)

# Load the .npz file
print("Content of the file:")
data = np.load(npz_file_path, allow_pickle=True)

print(data.files)

# Extract variables

spectra = data['spectra_min'].item()
f = spectra['f']
p = spectra['p']

print("Spectra loaded")

hmm = data['model'].item()
indices = data['indices']
q = data['q']
vpath = data['viterbi_path']
Gamma = data['gamma']
indices = data['indices']
fo = data['fractional_occupancy']
means = data['means']
covs = data['covariances']
states = data['active_states']
TP = data['transition_probabilities']
FO = data['fractional_occupancy']
SR = data['switching_rate']
LTmean = data['dwell_time_mean']
K = hmm.hyperparameters["K"]

print("Variables loaded")


plot_viterbi_path(vpath, indices)

plot_transition_probabilities(hmm)

plot_state_covariances(covs, K)

plot_state_means(means)

plot_fractional_occupancy(FO)

plot_switching_rate(SR)

plot_state_lifetimes(LTmean)

plot_statewise_spectra(f, p, K)

plot_statewise_average_spectra(f, p)

#plot_statewise_average_spectra_one_figure(f, p)

print("\n### Plotting done for file: ", npz_file_path)
