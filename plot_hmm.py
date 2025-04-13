from hmm_visuals import (
    plot_viterbi_path,
    plot_transition_probabilities,
    plot_state_covariances,
    plot_fractional_occupancy,
    plot_switching_rate,
    plot_state_lifetimes,
    plot_statewise_spectra)
from config import fname
import numpy as np
from glhmm import glhmm

#job_id = '_job_source_29V4_ALL' # 0.9 PCA applied twice. Detrend=True, downsampling=100. All states active.
# job_id = '_job_source_30v2_ALL' # 0.8 PCA (18 parcels). Only states 1, 3 and 6 active.
#job_id = '_job_source_30v3_ALL' # 0.95 PCA (31 parcels). All states active.
#job_id = '_job_source_30v4_ALL' # 0.98 PCA (41 parcels). All states active.
#job_id = '_job_source_30v6_ALL' # 0.99 PCA (48 parcels) applied in preproc. All states active.
#job_id = '_job_source_30v7_ALL' # PCA 45 applied in embedding. All states active.
#job_id = '_job_source_30v8_ALL' # PCA 45 applied in embedding. All states active.
#job_id = '_job_source_30v9_ALL_complete' # PCA 45. 6 active states. Downsampled 100Hz.
#job_id = '_job_source_31v1_ALL_complete-200hz'# PCA 45, no downsampling. TDE -7:7
#job_id = '_job_source_0504v1' # PCA 0.9, no downsampling, TDE -15:15
#job_id = '_job_source_0604v1-testing-pca'  
# job_id = '_job_source_1304v1-testing-spectra' # sub-16C only. Spectral analysis done.
job_id = '_job_source_1304v2-testing-cov' # sub-16C only. spectral, means and cov orig space done.

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

print("Variables loaded")

"""

plot_viterbi_path(vpath, indices)

plot_transition_probabilities(hmm)

plot_state_covariances(covs, states)

plot_fractional_occupancy(FO)

plot_switching_rate(SR)

plot_state_lifetimes(LTmean)

plot_statewise_spectra(f, p, states)


print("\n### Plotting done for file: ", npz_file_path)
"""