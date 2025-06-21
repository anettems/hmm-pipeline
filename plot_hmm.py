from hmm_visuals import (
    plot_viterbi_path,
    plot_transition_probabilities,
    plot_state_covariances,
    plot_fractional_occupancy,
    plot_switching_rate,
    plot_state_lifetimes,
    plot_statewise_spectra,
    plot_statewise_average_spectra,
    plot_state_means)
from config import fname
import numpy as np

job_id = '_job_source_2204v2_pca09'

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

"""plot_state_covariances(covs, K)

plot_state_means(means)"""

plot_fractional_occupancy(FO)

plot_switching_rate(SR)

plot_state_lifetimes(LTmean)
"""
plot_statewise_spectra(f, p, K)

plot_statewise_average_spectra(f, p)"""

print("\n### Plotting done for file: ", npz_file_path)
