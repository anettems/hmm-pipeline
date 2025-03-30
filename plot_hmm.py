from hmm_visuals import (
    plot_viterbi_path,
    plot_transition_probabilities,
    plot_state_covariances,
    plot_fractional_occupancy,
    plot_switching_rate,
    plot_state_lifetimes,
    plot_statewise_spectra,
    plot_spectra_heatmap)
from config import fname
import numpy as np

#job_id = '_job_source_29V4_ALL' # 0.9 PCA applied twice. Detrend=True, downsampling=100. All states active.
# job_id = '_job_source_30v2_ALL' # 0.8 PCA (18 parcels). Only states 1, 3 and 6 active.
#job_id = '_job_source_30v3_ALL' # 0.95 PCA (31 parcels). All states active.
#job_id = '_job_source_30v4_ALL' # 0.98 PCA (41 parcels). All states active.
#job_id = '_job_source_30v6_ALL' # 0.99 PCA (48 parcels) applied in preproc. All states active.
#job_id = '_job_source_30v7_ALL' # PCA 45 applied in embedding. All states active.
job_id = '_job_source_30v8_ALL' # PCA 45 applied in embedding. All states active.

#job_id = '_job_source_30v7_ALL_complete' 
npz_file_path = fname.tde_hmm_ob(job_id=job_id)

# Load the .npz file
print("Content of the file:")
data = np.load(npz_file_path, allow_pickle=True)

print(data.files)

# Extract variables
hmm = data['model'].item()          # Use .item() if it's a Python object
indices = data['indices']
q = data['q']
vpath = data['viterbi_path']
Gamma = data['gamma']
indices = data['indices']
fo = data['fractional_occupancy']
spectra = data['spectra']
freqs = data['spectra_freqs']

plot_viterbi_path(vpath, indices)

plot_transition_probabilities(hmm)

plot_state_covariances(hmm, q)

plot_fractional_occupancy(Gamma, indices)

plot_switching_rate(Gamma, indices)

plot_state_lifetimes(vpath, indices)

plot_statewise_spectra(spectra, freqs)

plot_spectra_heatmap(spectra, freqs)


print("\n### Plotting done for file: ", npz_file_path)