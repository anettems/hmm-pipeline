import numpy as np
import matplotlib.pyplot as plt
from glhmm import graphics
from config import fname
import pandas as pd
from scipy.stats import sem

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
    K = TP.shape[0]
    
    plt.figure(figsize=(7, 4))
    
    # Original Transition Probabilities
    plt.subplot(1, 2, 1)
    plt.imshow(TP, cmap=cmap, interpolation="nearest")
    plt.title("Transition Probabilities")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.xticks(ticks=np.arange(K), labels=np.arange(1, K+1))
    plt.yticks(ticks=np.arange(K), labels=np.arange(1, K+1))
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Transition Probabilities without Self-Transitions
    TP_noself = TP - np.diag(np.diag(TP))
    TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)
    plt.subplot(1, 2, 2)
    plt.imshow(TP_noself2, cmap=cmap, interpolation="nearest")
    plt.title("Transition Probabilities\nwithout Self-Transitions")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.xticks(ticks=np.arange(K), labels=np.arange(1, K+1))
    plt.yticks(ticks=np.arange(K), labels=np.arange(1, K+1))
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def plot_state_covariances(state_FC, K):
    """
    Plot one covariance matrix per figure, for each HMM state.
    """
    cmap = "coolwarm"

    for k in range(K):
        plt.figure(figsize=(6, 5))
        plt.imshow(state_FC[:, :, k], cmap=cmap)
        plt.xlabel("Brain region")
        plt.ylabel("Brain region")
        plt.colorbar()
        plt.title(f"State Covariance - State #{k + 1}")
        plt.tight_layout()
        plt.show()
    
    

def plot_state_means(means):
    """
    Plot HMM state means as a 2D heatmap.

    Parameters
    * means : np.ndarray
        Array of shape (n_parcels, n_states)
    """
    
    n_parcels, n_states = means.shape
    cmap = "coolwarm"

    plt.figure(figsize=(8, 10))
    im = plt.imshow(means, cmap=cmap, interpolation="none", aspect="auto")

    plt.colorbar(im, label='Activation level')
    plt.title("State mean activation")

    # Set x-ticks (states)
    plt.xticks(ticks=np.arange(n_states), labels=[f"{i+1}" for i in range(n_states)])
    plt.xlabel('State')

    # Fewer y-ticks if too many parcels
    step = 100 if n_parcels > 100 else 10
    plt.yticks(ticks=np.arange(0, n_parcels, step))
    plt.ylabel('Brain region')

    plt.tight_layout()
    plt.show()


    


def plot_fractional_occupancy(FO):
    """
    Compute and plot the fractional occupancy (FO), i.e., the fraction of time in each session 
    occupied by each state.
    """
    
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


def plot_switching_rate(SR):
    """
    Plot the switching rate, which indicates how quickly subjects switch between states.

    """
    graphics.plot_switching_rates(SR, num_x_ticks=SR.shape[0])




def plot_state_lifetimes(LTmean):
    graphics.plot_state_lifetimes(LTmean, num_x_ticks=LTmean.shape[0], ylabel='Mean lifetime')
    
    
    
    
def plot_statewise_spectra(f, p, K):
    # Ensure p has 4 dimensions: (n_subjects, n_freq, n_channels, n_states)
    if p.ndim == 3:  # likely (n_freq, n_channels, n_states)
        p = np.expand_dims(p, axis=0)
    elif p.ndim == 2:  # (n_freq, n_channels)
        p = np.expand_dims(np.expand_dims(p, axis=0), axis=-1)

    mean_p = p.mean(axis=0)  # (n_freq, n_channels, n_states)

    for k in range(K):
        plt.figure()
        plt.plot(f, mean_p[:, :, k])
        plt.title(f"State {k+1} - Power Spectra (all parcels)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        

def plot_statewise_average_spectra(f, p):
    """
    Plot one figure per state showing the mean power spectrum across parcels.
    """
    while p.ndim < 4:
        p = np.expand_dims(p, axis=0)
        
    n_subjects, n_freq, n_parcels, n_states = p.shape

    # Compute mean and SEM over subjects for each state
    mean_spectra = p.mean(axis=2)  # mean over parcels → (n_subjects, n_freq, n_states)
    mean_over_subjects = mean_spectra.mean(axis=0)  # (n_freq, n_states)
    sem_over_subjects = sem(mean_spectra, axis=0)  # (n_freq, n_states)

    # Plot one figure per state
    for k in range(n_states):
        plt.figure(figsize=(8, 5))
        plt.plot(f, mean_over_subjects[:, k], label=f"State {k+1}", color=f"C{k}")
        plt.fill_between(f,
                         mean_over_subjects[:, k] - sem_over_subjects[:, k],
                         mean_over_subjects[:, k] + sem_over_subjects[:, k],
                         alpha=0.3, color=f"C{k}")
        plt.title(f"State {k+1} - Mean Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


