import numpy as np
from mne.time_frequency import psd_array_multitaper

def compute_statewise_spectra_windowed(data_processed, Gamma, sfreq, lag, fmin=1, fmax=48,
                                       bandwidth=4.0, win_length_sec=5, step_sec=2.5):
    """
    Compute Gamma-weighted state-wise multitaper spectra using sliding windows.
    
    Parameters:
        data_processed: (n_times, n_channels), before TDE
        Gamma: (n_times_effective, n_states), from TDE-HMM
        sfreq: Sampling frequency (Hz)
        lag: Number of lag points lost at each edge
        fmin, fmax: Frequency range
        bandwidth: Multitaper bandwidth
        win_length_sec: Window size in seconds
        step_sec: Step size in seconds

    Returns:
        spectra: (n_states, n_freqs), Gamma-weighted average spectra
        freqs: (n_freqs,), frequency axis
    """
    # === Safe alignment: match data to Gamma based on length ===
    n_gamma = Gamma.shape[0]
    n_data = data_processed.shape[0]
    
    # Ensure Gamma is centered in data_processed (assuming symmetrical lag loss)
    start_idx = (n_data - n_gamma) // 2
    end_idx = start_idx + n_gamma
    data = data_processed[start_idx:end_idx]
    
    assert data.shape[0] == Gamma.shape[0], (
        f"Mismatch after alignment: data has {data.shape[0]} samples, Gamma has {Gamma.shape[0]}")


    n_times, n_channels = data.shape
    n_states = Gamma.shape[1]
    win_samples = int(win_length_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    n_freqs = None
    spectra = None

    # Accumulators
    state_spectra = np.zeros((n_states, 0))  # initialize later with correct freq size
    state_weights = np.zeros(n_states)

    for start in range(0, n_times - win_samples + 1, step_samples):
        stop = start + win_samples
        X_win = data[start:stop]
        G_win = Gamma[start:stop]

        # Multitaper PSD: shape (n_channels, n_freqs)
        psd, freqs = psd_array_multitaper(
            X_win.T, sfreq=sfreq, fmin=fmin, fmax=fmax,
            bandwidth=bandwidth, adaptive=True, normalization='full', verbose=False
        )

        if spectra is None:
            n_freqs = len(freqs)
            state_spectra = np.zeros((n_states, n_freqs))

        # Compute Gamma weights per state in this window
        window_state_weights = G_win.sum(axis=0)
        window_state_weights /= window_state_weights.sum()  # normalize

        # For each state, accumulate weighted spectrum
        for k in range(n_states):
            for ch in range(n_channels):
                state_spectra[k] += window_state_weights[k] * (np.mean(X_win[:, ch]**2)) * psd[ch]
            state_weights[k] += window_state_weights[k]

    # Normalize by total weights
    for k in range(n_states):
        if state_weights[k] > 0:
            state_spectra[k] /= state_weights[k]

    return state_spectra, freqs
