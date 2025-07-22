from config import fname
import numpy as np
from glhmm import utils
from settings_hmm_beta import job_id

dual_file_path = fname.hmm_dual_ob(job_id=job_id)

# Load the .npz file
print("Content of the file:")
data = np.load(dual_file_path, allow_pickle=True)

print(data.files)

dual_hmm = data['dual_estimates'].item()
data_tde = data['data']
indices_tde = data['indices']

print("Variables loaded\n")

# Decode Gamma from dual HMM: FO, SR, Vpath

Gamma_dual, Xi_dual, _ = dual_hmm.decode(X=None, Y=data_tde, indices=indices_tde, viterbi=False)
vpath_dual = dual_hmm.decode(X=None, Y=data_tde, indices=indices_tde, viterbi=True)

FO_dual = utils.get_FO(Gamma_dual, indices=indices_tde)
SR_dual = utils.get_switching_rate(Gamma_dual, indices_tde)

print("Dual HMM Gamma decoded")

# Subject-Specific Transition Probability Matrix (P) and Initial Probabilities (Pi)
TP_dual = dual_hmm.P.copy()
IP_dual = dual_hmm.get_Pi()

print("Dual Transition probabilities and Initial probabilities extracted")

# Subject-Specific State Covariance Matrices 

q = data_tde.shape[1] # the number of parcels/channels
K = dual_hmm.hyperparameters["K"] # the number of states

state_FC_dual = np.zeros(shape=(q, q, K))

print("Starting covariance computing")

for k in range(K):
    state_FC_dual[:,:,k] = dual_hmm.get_covariance_matrix(k=k, orig_space=True) # the state covariance matrices in the shape (no. features, no. features, no. states)


print("Covariances computed in original space")

# Subject-Specific State Means

means_dual = dual_hmm.get_means(orig_space=True)
print("Dual state means computed\n")

# Save .npz per subject

np.savez(dual_file_path,
         dual_estimates=dual_hmm,
         data=data_tde,
         indices=indices_tde,
         Gamma=Gamma_dual,
         vpath=vpath_dual,
         FO=FO_dual,
         SR=SR_dual,
         TP=TP_dual,
         IP=IP_dual,
         covariance=state_FC_dual,
         means=means_dual)

print("\n >>> Dual estimate saved to ", dual_file_path)
