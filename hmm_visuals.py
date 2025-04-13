import numpy as np
import matplotlib.pyplot as plt
from glhmm import utils, graphics
from config import fname
import pandas as pd

# ---------------------------
# VISUALISATION FUNCTIONS
# ---------------------------


def plot_viterbi_path(vpath, indices):
    """
    Plot the Viterbi path for each subject separately.
    
    Parameters:
        vpath: the decoded state sequence.
        indices: (n_subjects, 2) array of start and end timepoints per subject.
    """
    df_subjects = pd.read_csv(fname.subjects_txt, names=["subject"])

    for i, row in df_subjects.iterrows():
        subject = row["subject"]
        start, end = indices[i]
        vpath_sub = vpath[start:end]
 
        graphics.plot_vpath(
            vpath_sub,
            title=f"Viterbi Path - {subject}",
            figsize=(7, 2.5),
            show_legend=True,
            xlabel='Timepoints',
            ylabel='',
        )
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
        state_FC[:, :, k] = hmm.get_covariance_matrix(k=k, orig_space=False)
    
    plt.figure(figsize=(10, 8))
    for k in range(K):
        plt.subplot(3, 2, k + 1)
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
    
    graphics.plot_FO(
        FO,
        figsize=(8, 4),
        fontsize_ticks=12,
        fontsize_labels=14,
        fontsize_title=16,
        width=0.8,
        xlabel='Subject',
        ylabel='Fractional occupancy',
        title='State Fractional Occupancies',
        show_legend=True,
        num_x_ticks=11,
        num_y_ticks=5,
        pad_y_spine=None,
        save_path=None)

def plot_switching_rate(Gamma, indices):
    """
    Compute and plot the switching rate, which indicates how quickly subjects switch between states.
    
    Parameters:
        Gamma: state probability matrix from the HMM training.
        indices: 2D array specifying [start, end] indices for each subject/session.
    """
    SR = utils.get_switching_rate(Gamma, indices)
    graphics.plot_switching_rates(
        SR,
        figsize=(8, 4),
        fontsize_ticks=12,
        fontsize_labels=14,
        fontsize_title=16,
        width=0.18,
        xlabel='Subject',
        ylabel='Switching Rate',
        title='State Switching Rates',
        show_legend=True,
        num_x_ticks=12,
        num_y_ticks=3,
        pad_y_spine=None,
        save_path=None)

def plot_state_lifetimes(vpath, indices):
    """
    Compute and plot the dwell times (state lifetimes) for each state.
    
    Parameters:
        vpath: the decoded Viterbi state sequence.
        indices: 2D array specifying [start, end] indices for each subject/session.
    """
    LTmean, LTmed, LTmax = utils.get_life_times(vpath, indices)
    graphics.plot_state_lifetimes(
        LTmean,  # Use LTmed or LTmax to plot those instead
        figsize=(8, 4),
        fontsize_ticks=12,
        fontsize_labels=14,
        fontsize_title=16,
        width=0.18,
        xlabel='Subject',
        ylabel='Lifetime',
        title='State Lifetimes (Mean)',
        show_legend=True,
        num_x_ticks=12,
        num_y_ticks=2,
        pad_y_spine=None,
        save_path=None)
    
def plot_statewise_spectra(spectra_fit, K):
    f = spectra_fit["f"]
    p = spectra_fit["p"]  # shape: (n_subjects, n_freq, n_channels, n_states)
    
    mean_p = p.mean(axis=0)  # average across subjects → (n_freq, n_channels, n_states)
    
    for k in range(K):
        plt.figure()
        plt.plot(f, mean_p[:, :, k])
        plt.title(f"State {k+1} - Power spectra (all parcels)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

