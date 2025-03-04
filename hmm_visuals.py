import numpy as np
import matplotlib.pyplot as plt
from glhmm import utils, graphics

# ---------------------------
# VISUALISATION FUNCTIONS
# ---------------------------


def plot_viterbi_path(vpath):
    """
    Plot the Viterbi path for the concatenated subjects using the glhmm graphics module.
    
    Parameters:
        vpath: the decoded state sequence.
    """
    graphics.plot_vpath(vpath, title="Viterbi Path (Concatenated Subjects)")

def visualize_state_means(hmm):
    """
    Retrieve state means from the HMM and visualise them as a heatmap.
    
    Parameters:
        hmm: the trained HMM object.
    """
    state_means = hmm.get_means()  # shape: [n_features, K]
    K = hmm.hyperparameters["K"]
    
    plt.figure(figsize=(6, 4))
    plt.imshow(state_means, cmap="coolwarm", interpolation="none")
    plt.colorbar(label="Activation Level")
    plt.title("State Mean Activation")
    plt.xlabel("State")
    plt.ylabel("Brain region")
    plt.xticks(np.arange(K), np.arange(1, K + 1))
    plt.tight_layout()
    plt.show()

def plot_transition_probabilities(hmm):
    """
    Plot the HMM's transition probability matrix (both original and without self-transitions).
    
    Parameters:
        hmm: the trained HMM object.
    """
    TP = hmm.P.copy()
    cmap = "coolwarm"
    
    plt.figure(figsize=(7, 4))
    
    # Original Transition Probabilities
    plt.subplot(1, 2, 1)
    plt.imshow(TP, cmap=cmap, interpolation="nearest")
    plt.title("Transition Probabilities")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Transition Probabilities without Self-Transitions
    TP_noself = TP - np.diag(np.diag(TP))
    TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)
    plt.subplot(1, 2, 2)
    plt.imshow(TP_noself2, cmap=cmap, interpolation="nearest")
    plt.title("Transition Probabilities\nwithout Self-Transitions")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def plot_state_covariances(hmm, q):
    """
    Retrieve and plot the state covariance matrices (time-varying functional connectivity).
    
    Parameters:
        hmm: the trained HMM object.
        q: number of features (e.g., parcels/channels) in the data.
    """
    cmap = "coolwarm"
    K = hmm.hyperparameters["K"]
    state_FC = np.zeros((q, q, K))
    
    for k in range(K):
        state_FC[:, :, k] = hmm.get_covariance_matrix(k=k)
    
    plt.figure(figsize=(10, 8))
    for k in range(K):
        plt.subplot(2, 2, k + 1)
        plt.imshow(state_FC[:, :, k], cmap=cmap)
        plt.xlabel("Brain region")
        plt.ylabel("Brain region")
        plt.colorbar()
        plt.title("State covariance\nstate #%s" % (k + 1))
    plt.subplots_adjust(hspace=0.7, wspace=0.8)
    plt.show()

def plot_fractional_occupancy(Gamma, indices):
    """
    Compute and plot the fractional occupancy (FO), i.e., the fraction of time in each session 
    occupied by each state.
    
    Parameters:
        Gamma: state probability matrix from the HMM training.
        indices: 2D array specifying [start, end] indices for each subject/session.
    """
    FO = utils.get_FO(Gamma, indices=indices)
    graphics.plot_FO(FO, num_ticks=FO.shape[0])

def plot_switching_rate(Gamma, indices):
    """
    Compute and plot the switching rate, which indicates how quickly subjects switch between states.
    
    Parameters:
        Gamma: state probability matrix from the HMM training.
        indices: 2D array specifying [start, end] indices for each subject/session.
    """
    SR = utils.get_switching_rate(Gamma, indices)
    graphics.plot_switching_rates(SR, num_ticks=SR.shape[0])

def plot_state_lifetimes(vpath, indices):
    """
    Compute and plot the dwell times (state lifetimes) for each state.
    
    Parameters:
        vpath: the decoded Viterbi state sequence.
        indices: 2D array specifying [start, end] indices for each subject/session.
    """
    LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices)
    graphics.plot_state_lifetimes(LTmean, num_ticks=LTmean.shape[0], ylabel='Mean lifetime')


# Optionally, you can add an example "main" block to test individual functions:
if __name__ == "__main__":
    # Note: This block is only for testing purposes and will only run
    # when this script is executed directly, not when imported.
    # You need to have a trained HMM model and data loaded to use these functions.
    
    # Example (pseudocode):
    # from your_data_loading_module import X_concat, indices, Gamma, hmm
    # vpath = get_viterbi_path(hmm, X_concat, indices)
    # plot_viterbi_path(vpath)
    # visualize_state_means(hmm)
    # plot_transition_probabilities(hmm)
    # q = X_concat.shape[1]
    # plot_state_covariances(hmm, q)
    # plot_fractional_occupancy(Gamma, indices)
    # plot_switching_rate(Gamma, indices)
    # plot_state_lifetimes(vpath, indices)
    
    pass
