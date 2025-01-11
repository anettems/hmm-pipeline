import numpy as np
import mne
import ast

from scipy.signal import find_peaks


def pick_ROI(side):
    """
    function to pick pair of gradiometer channels from ROI list of channels

    Parameters
    ----------
    ROI: list of MEG channels of interest
    top_ch: previously chosen channel pair index (chosen manually or in an automated fashion)
        note this is a number, which corresponds to a channel pair (and its PSD vector data)

    Returns
    -------
    chans: the names of the channels in chosen top channel pair
    """
    if side == "left":
        ROI = ["MEG0213","MEG0212",
                "MEG0222","MEG0223",
                "MEG0413","MEG0412",
                "MEG0422","MEG0423",
                "MEG0633","MEG0632",
                "MEG0243","MEG0242",
                "MEG0232","MEG0233",
                "MEG0443","MEG0442",
                "MEG0432","MEG0433",
                "MEG0713","MEG0712",
                "MEG1613","MEG1612",
                "MEG1622","MEG1623",
                "MEG1813","MEG1812",
                "MEG1822","MEG1823",
                "MEG0743","MEG0742",
            ]
    else:
        ROI = ["MEG1043","MEG1042",
                "MEG1112","MEG1113",
                "MEG1123","MEG1122",
                "MEG1312","MEG1313",
                "MEG1323","MEG1322",
                "MEG0723","MEG0722",
                "MEG1142","MEG1143",
                "MEG1133","MEG1132",
                "MEG1342","MEG1343",
                "MEG1333","MEG1332",
                "MEG0733","MEG0732",
                "MEG2212","MEG2213",
                "MEG2223","MEG2222",
                "MEG2412","MEG2413",
                "MEG2423","MEG2422",]

    return ROI


def get_rows_and_columns_for_30():
    r = [0,1,0,1,0,1,
        0,1,0,1,2,3,
        2,3,2,3,2,3,
        2,3,4,5,4,5,
        4,5,4,5,4,5,]
    c = [0,0,1,1,2,
        2,3,3,4,4,
        0,0,1,1,2,
        2,3,3,4,4,
        0,0,1,1,2,
        2,3,3,4,4,]
    return r, c


#####################################################
###  Functions for plot_sensor_dual_psds.py file  ###
#####################################################


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def plot_psds(
    axes, sensor, freqs, psds_all_states, number_of_states, r, c, ch_names, colors
):
    labels = ["State 1", "State 2", "State 3", "Satate 4", "State 5", "State 6"]

    if sensor == 29:
        for state in range(0, number_of_states):
            if (True in np.isnan(np.array(psds_all_states[state]))) == False:
                axes[r[sensor], c[sensor]].plot(
                    freqs,
                    psds_all_states[state],
                    color=colors[state],
                    label=labels[state],
                )
        axes[r[sensor], c[sensor]].title.set_text(ch_names[sensor])

    else:
        for state in range(0, number_of_states):
            if (True in np.isnan(np.array(psds_all_states[state]))) == False:
                axes[r[sensor], c[sensor]].plot(
                    freqs, psds_all_states[state], color=colors[state]
                )
        axes[r[sensor], c[sensor]].title.set_text(ch_names[sensor])


def set_axis_limits(axes, r, c, maxim):
    for sensor in range(0, 30):
        axes[r[sensor], c[sensor]].set_ylim(bottom=0, top=maxim)
        axes[r[sensor], c[sensor]].set_xlim(left=2, right=48)


def AUC(freqs, psd):
    I1 = np.trapz(y=psd, x=freqs, dx=2)
    return I1

#####################################################
#####################################################

########################################################
###  Functions for make_event_amplitude_csv.py file  ###
########################################################


# Morlet wavelet decomposition
# obtain beta amplitude envelope
def beta_amplitude_envelope(data, sfreq, dwn, lower_freq, upper_freq):
    # Resampling
    ### 16.9.2019 - Jussi Nurminen says mne.filter.resample includes filters to avoid aliasing
    # down=sfreq/dwn
    # downsample to frequency specified by 'dwn'
    # out1= mne.filter.resample(data, down=down, npad='auto', n_jobs=16, pad='reflect_limited', verbose=None) # additional options: window='boxcar', npad=100,
    out1 = data

    # split data into consecutive epochs
    window = 10  # length of individual time windows (in seconds)
    ws = int(window * dwn)  # number of samples per window
    overlap = (
        1 - 0
    )  # set amount of overlap for consecutive FFT windows (second number sets amount of overlap)

    # separate data into consecutive data chunks (episode-like, because spectral_connectivity expects epochs)
    array1 = list()
    start = 0
    stop = ws
    step = int(ws * overlap)
    while stop <= out1.shape[1]:
        tmp = out1[:, start:stop]
        start += step
        stop += step
        array1.append(tmp)
    # array1=np.expand_dims(array1, axis=1) #Comment out if all sensors are made commonly

    # define frequencies of interest
    freqs = np.arange(lower_freq, upper_freq, 1.0)
    n_cycles = freqs / 2.0

    # calculate power
    # print(np.array(data).shape)
    # print(np.array(array1).shape)
    power = mne.time_frequency.tfr_array_morlet(
        array1, sfreq=dwn, freqs=freqs, n_cycles=n_cycles, output="complex", n_jobs=16
    )

    return power, freqs


def amplitude_envelope(lower_freq, higher_freq, freqs, power):

    b_idx = np.where(np.logical_and(freqs > lower_freq, freqs < higher_freq))
    amplitude = np.mean(np.abs(power[0][:, b_idx[0]]), axis=1)

    for k in range(1, len(power)):
        tmp = power[k][:, b_idx[0]]
        tmptmp = np.mean(np.abs(tmp), axis=1)
        amplitude = np.concatenate((amplitude, tmptmp), axis=1)

    return amplitude


def envelope_cutted_list(idx_nonzero_1, zeroed_envelope1):
    fi = 0
    empty_list = []
    for ind in range(1, len(idx_nonzero_1[0])):
        if (idx_nonzero_1[0][ind] - idx_nonzero_1[0][ind - 1] != 1) and (
            ind != len(idx_nonzero_1[0])
        ):
            take_indices = idx_nonzero_1[0][fi:ind]
            take_envelope = zeroed_envelope1[take_indices]
            empty_list.append(take_envelope)
            fi = ind

        elif ind == len(idx_nonzero_1[0]):
            take_indices = idx_nonzero_1[0][fi : ind + 1]
            take_envelope = zeroed_envelope1[take_indices]
            empty_list.append(take_envelope)
    return empty_list


##############################################################
###  Functions for make_event_dispersion_rate_csv.py file  ###
##############################################################


def cut_ones_into_list(vmap_multiply):
    """
    Takes vmap_multipy in and separates that into separate
    lists having zeros nd ones.
    """

    fi = 0
    list_ones = []
    list_zeros = []
    for ind in range(1, len(vmap_multiply)):
        if vmap_multiply[ind] + vmap_multiply[ind - 1] == 1:
            take_list = vmap_multiply[fi:ind]
            if take_list[0] == 0:
                list_zeros.append(take_list)
            else:
                list_ones.append(take_list)
            fi = ind
    return list_ones, list_zeros

def cut_ones_into_list_50ms(vmap_multiply):
    """
    Takes vmap_multipy in and separates that into separate
    lists having zeros nd ones. Differs from the upper
    function in a way, that if the event is shorter than 50ms,
    it is not considered as beta burst.
    """

    fi = 0
    list_ones = []
    list_zeros = []
    for ind in range(1, len(vmap_multiply)):
        if vmap_multiply[ind] + vmap_multiply[ind - 1] == 1:
            take_list = vmap_multiply[fi:ind]
            if take_list[0] == 0:
                list_zeros.append(take_list)
                fi = ind
            else:
                if len(take_list) > 10:
                    list_ones.append(take_list)
                    fi = ind
    return list_ones, list_zeros

##############################################
###  Functions for plot_TPM_label.py file  ###
##############################################


def make_string_array_to_numpy(string_array):
    # Remove the outer brackets and the 'array' part
    cleaned_string = string_array.strip("[]").replace("array(", "").replace(")", "")
    # Convert the cleaned string to a NumPy array
    array = np.array(ast.literal_eval(cleaned_string))

    return array


####################################################
###  Function to crop bad segments from Raw data ###
####################################################


def reject_bad_segs(raw):
    """ This function rejects all time spans annotated as "bad" and concatenates the rest"""

    if len(raw.annotations) >= 1:
        fs = raw.first_samp
        sf = raw.info['sfreq']

        raw_mins = [0]
        raw_maxs = [(raw.n_times/sf)-1]

        for j in range(0, len(raw.annotations)):
            
            tmax = raw.annotations.onset[j] # Onset to the max list
            tmin = raw.annotations.onset[j] + raw.annotations.duration[j] # End to the min list
            raw_mins.append(tmin - (fs/sf))
            raw_maxs.append(tmax - (fs/sf))

        # Sort the start and end lists
        raw_mins.sort()
        raw_maxs.sort()

        # Zip the lists to get tmins and maxs to crop function
        crop_sections = list(zip(raw_mins,raw_maxs))
        
        # Empty list for raw segments
        raw_segs = []

        for crop in crop_sections:
            raw_segs.append( raw.copy().crop(tmin=crop[0],tmax=crop[1]))
        
        return mne.concatenate_raws(raw_segs)

    else:
        return raw