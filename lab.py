# !/usr/bin/env python
# -----------------------------------------------------------------------
# COMPANY              : South African Radio Astronomy Observatory (SARAO)
# -----------------------------------------------------------------------
# COPYRIGHT NOTICE :
#
# The copyright, manufacturing and patent rights stemming from this document
# in any form are vested in South African Radio Astronomy Observatory (SARAO).
#
# Author: Dirk Frier Cronje
# Date: July 2024
# -----------------------------------------------------------------------
# DESCRIPTION :
#
# Running this script on its own:
# - You will need:
#   -> a laptop with dual microphones or
#   -> laptop/NUC connected to two microphones
# - Command line example:
#   -> python3 lab.py
#
# Running this script with office.py:
# - You will need two computing devices on the same network:
#   -> One connected to 2 microphones and a screen.
#   -> The second connected to a screen.
# - Command line example:
#   -> device one: python3 lab.py
#   -> device two: python3 office.py

# OTHER EXAMPLES:
# Example 1: python3 lab.py -t 10
# Example 2: python3 lab.py -n 10
# Example 3: python3 lab.py -b
# -----------------------------------------------------------------------

import math
import time
import datetime
import os
import cProfile
import pstats
import csv
import threading
import io

import pyaudio
import numpy as np
from numpy import pi, polymul
import pika
import json
import matplotlib.pyplot as plt
import h5py
from scipy.signal import bilinear
from scipy.signal import lfilter
from optparse import OptionParser
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Parameters

# CHUNK = 2**13 and N = 8 is responsesive but takes up too much cpu usage
# N = 8 is good here
# DURATION is the duration of the plot in seconds
# Set isPlotting to False to disable plotting - for debuggin purposes
# Calibration factor (C_FACTOR) to be changed when calibration takes place.
# log_file is actualy directory

CHUNK = 2**14 # 2**11
RATE = 44100
DURATION = 8
NYQUIST_RATE = RATE // 2
A_FILTER_TIME = 0
POWER = 1
A_FILTER_FREQ = 2
isPloting = True
isPrintingSNR = False
C_FACTOR = 100
MICROPHONE_SENSITIVITY = 12.1 * 10 ** (-3)
REFERENCE_PRESSURE = 20 * 10 ** (-6)
option = A_FILTER_TIME
ave_ch2 = 0
flag = 1
N = 1 # 64, 5 works on laptop

new_log_message = ""
new_size = 0
log_file = "log"
log_frequency = 2
counter_n = 0
start_time = time.monotonic()

n_streams = 1 # Change when chaning the number of microphones

colour = [""] * n_streams
thresholds = [[0] * 7 for _ in range(n_streams)]
prev_thresholds = [[0] * 7 for _ in range(n_streams)]
plot_titles = [[0] * 7 for _ in range(n_streams)]
ave = [0] * n_streams
temp = [0] * n_streams
message_body = {}


def write_to_csv(timestamp, cpu_percent):
    with open("cpu.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, cpu_percent])


def log_message_samples(log_file, dataset_name, group_name, spectrum=None):
    """
    Writes a log message in a directory called "samples" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """

    log_file = os.path.join(log_file, "samples")

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file = os.path.join(log_file, f"log_{today}_N_SAMPLES.hdf5")

    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, "w"):
            pass

    with h5py.File(log_file, "a") as f:
        if group_name not in f:
            # Create the file if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)

def log_message_continuous(log_file, dataset_name, group_name, spectrum=None):
    """
    Writes a log message in a directory called "samples" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """

    log_file = os.path.join(log_file, "continuous")

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file = os.path.join(log_file, f"log_{today}_continuous.hdf5")

    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, "w"):
            pass

    with h5py.File(log_file, "a") as f:
        if group_name not in f:
            # Create the file if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)


def log_message_time(log_file, dataset_name, group_name, spectrum=None):
    """
    Writes a log message in a directory called "time" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """

    log_file = os.path.join(log_file, "time")

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file_path = os.path.join(log_file, f"log_{today}_TIME.hdf5")

    if not os.path.isfile(log_file_path):
        # Create the file if it doesn't exist
        with h5py.File(log_file_path, "w"):
            pass

    with h5py.File(log_file_path, "a") as f:
        if group_name not in f:
            # Create the file if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)


def log_message(log_file, dataset_name, group_name, spectrum=None):
    """
    Writes a log message in a directory called "danger" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """

    log_file = os.path.join(log_file, "danger")

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file = os.path.join(log_file, f"log_{today}.hdf5")

    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, "w"):
            pass

    with h5py.File(log_file, "a") as f:
        if group_name not in f:
            # Create the group if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)


def log_message_midnight(log_file, dataset_name, group_name, spectrum=None):
    """
    Writes a log message in a directory called "midnight" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """
    log_file = os.path.join(log_file, "midnight")

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file = os.path.join(log_file, f"log_{today}.hdf5")

    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, "w"):
            pass

    with h5py.File(log_file, "a") as f:
        if group_name not in f:
            # Create the file if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)


def process_log_message(
    channel,
    spectrum,
    freq,
    log_file,
    thresholds_1D,
    prev_thresholds,
    N_samples=False,
    time_period=False,
    midnight=False,
    thresholds_flag=False,
    continuous=False
):
    """
    Prepares log messages and sets up structures of log directories based on
    which flag is set(time_period, midnight, N_samples, thresholds)

    Parameters:
        -   channel/microphone number (int): Indication of channel number for
            message title.

        -   spectrum (numPy array): Frequency components.

        -   freq (numPy array): Frequencies

        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   N_samples: If true, log the N measurements in a seperate file

        -   time_period: If true, log every measurent within the specified time
            period.

        -   midnight: If true, log every sample within the first 10 seconds of
            it being midnight.

    Returns: void
    """
    threshold_A = thresholds_1D[0]
    threshold_B = thresholds_1D[1]
    threshold_C = thresholds_1D[2]
    threshold_D = thresholds_1D[3]
    threshold_E = thresholds_1D[4]
    threshold_F = thresholds_1D[5]
    threshold_G = thresholds_1D[6]

    prev_th_A = prev_thresholds[0]
    prev_th_B = prev_thresholds[1]
    prev_th_C = prev_thresholds[2]
    prev_th_D = prev_thresholds[3]
    prev_th_E = prev_thresholds[4]
    prev_th_F = prev_thresholds[5]
    prev_th_G = prev_thresholds[6]

    spec_and_freq = np.vstack((spectrum, freq))

    channel_name = f"Channel {channel}"

    if N_samples:
        new_log_message = (
            "___lab.py:___ PEAK:"
            + str(int(np.max(spectrum)))
            + ", Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
        log_message_samples(
            log_file, new_log_message, channel_name, spectrum=spec_and_freq
        )

    if continuous:
        new_log_message = (
            "___lab.py:___ PEAK:"
            + str(int(np.max(spectrum)))
            + ", Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
        log_message_continuous(
            log_file, new_log_message, channel_name, spectrum=spec_and_freq
        )

    if time_period:
        new_log_message = (
            "___lab.py:___ PEAK:"
            + str(int(np.max(spectrum)))
            + "dBA, Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
        log_message_time(
            log_file, new_log_message, channel_name, spectrum=spec_and_freq
        )

    if midnight:
        new_log_message = (
            "___lab.py:___ PEAK:"
            + str(int(np.max(spectrum)))
            + "dBA, Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
        log_message_midnight(
            log_file, new_log_message, channel_name, spectrum=spec_and_freq
        )

    if thresholds_flag:
        if threshold_A:
            if not prev_th_A:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:82dBA<SPL<85dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_A = threshold_A
            threshold_A = 0
        elif threshold_B:
            if not prev_th_B:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:85dBA<SPL<88dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_B = threshold_B
            threshold_B = 0
        elif threshold_C:
            if not prev_th_C:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:88dBA<SPL<91dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_C = threshold_C
            threshold_C = 0
        elif threshold_D:
            if not prev_th_D:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:91dBA<SPL<94dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_D = threshold_C
            threshold_D = 0
        elif threshold_E:
            if not prev_th_E:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:94dBA<SPL<97dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_E = threshold_E
            threshold_E = 0
        elif threshold_F:
            if not prev_th_F:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:97dBA<SPL<100dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_F = threshold_F
            threshold_F = 0
        elif threshold_G:
            if not prev_th_G:
                new_log_message = (
                    "___lab.py:___ PEAK:"
                    + str(int(np.max(spectrum)))
                    + "dBA, REGION:SPL>100dBA, Date:"
                    + datetime.datetime.now().strftime("%Y-%m-%d")
                    + ", Time:"
                    + datetime.datetime.now().strftime("%H:%M:%S")
                    + '.' + datetime.datetime.now().strftime("%f")[:3]
                )
                log_message(
                    log_file, new_log_message, channel_name, spectrum=spec_and_freq
                )
            prev_th_G = threshold_G
            threshold_G = 0

    thresholds_1D[0] = threshold_A
    thresholds_1D[1] = threshold_B
    thresholds_1D[2] = threshold_C
    thresholds_1D[3] = threshold_D
    thresholds_1D[4] = threshold_E
    thresholds_1D[5] = threshold_F
    thresholds_1D[6] = threshold_G

    prev_thresholds[0] = prev_th_A
    prev_thresholds[1] = prev_th_B
    prev_thresholds[2] = prev_th_C
    prev_thresholds[3] = prev_th_D
    prev_thresholds[4] = prev_th_E
    prev_thresholds[5] = prev_th_F
    prev_thresholds[6] = prev_th_G

    return thresholds_1D, prev_thresholds

def a_filter(f_bin):
    """
    Returns the A-weight factor per frequency

    parameters:
    f_bin (numPy array): Frequencies of the sampled spectrum

    Returns:
    The weights (int) to be applied to a frequency component
    """
    a_factor = []

    for i in range(len(f_bin)):
        a = (12194**2) * (f_bin[i] ** 4)
        b = (f_bin[i] ** 2) + (20.6**2)
        c = (f_bin[i] ** 2) + (107.7**2)
        d = (f_bin[i] ** 2) + (737.9**2)
        e = (f_bin[i] ** 2) + (12194**2)
        r_a = a / (b * math.sqrt(c * d) * e)
        # 20*log10(r_a at 1000) is equal to 2.0 This is done for
        # normalisation so that 0dB starts at 1000Hz
        a_factor.append(20 * np.log10(r_a) + 2.0)
    return a_factor


def A_weighting(fs):
    """
    Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.

    parameters:
    fs (int): Sampling rate

    Returns:
    Digital IRR filter based on the A-filter coefficients

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = polymul(
        [1, 4 * pi * f4, (2 * pi * f4) ** 2], [1, 4 * pi * f1, (2 * pi * f1) ** 2]
    )
    DENs = polymul(polymul(DENs, [1, 2 * pi * f3]), [1, 2 * pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)


def rms_flat(a):
    """
    Returns the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a) ** 2))


def data_format(option, audio_data):
    """
    Allows you to select what amplitude/spectrum format to be displayed.

    Parameters:
        option (int): Indicates which format to return:
            -   0 is single-sided A-filtered spectrum (default)
            -   1 is single-sided power spectrum
            -   2 is single-sided A-filtered spectrum with the filter
                applied in the frequency domain
        audio_data (16 bit NumPy array of ints): Raw time domain data
        of the size of CHUNK

    Returns: NumPy array of spectrum format of choice
    """
    global counter2
    audio_data = audio_data - np.mean(audio_data)
    audio_data *= np.hamming(CHUNK)

    if option == 0:
        # ===================================
        # TIME-DOMAIN A-FILTER IMPLEMENTATION
        # ===================================
        # The denominator of magnitude spectrum should be the voltage
        # representation of 20 micropascals and this should be measured
        # during calibration at the moment I am doing guess work here.

        b, a = A_weighting(RATE)
        y = lfilter(b, a, audio_data)
        magnitude_spectrum = abs(np.fft.fft(y / (150))[0 : CHUNK // 2]) # 1
        data = 20 * np.log10(magnitude_spectrum)

    elif option == 1:
        # ===================================
        # POWER
        # ===================================

        magnitude_spectrum = abs(np.fft.fft(audio_data)[0 : CHUNK // 2])
        data = 10 * np.log10((magnitude_spectrum**2))

    elif option == 2:
        # ===================================
        # FREQUENCY-DOMAIN A-FILTER IMPLEMENTATION
        # ===================================
        # The denominator of magnitude spectrum should be the voltage
        # representation of 20 micropascals and this should be measured
        # during calibration at the moment I am doing guess work here.

        magnitude_spectrum = abs(np.fft.fft(audio_data / 100)[0 : CHUNK // 2])
        data = 10 * np.log10(magnitude_spectrum**2) + a_filter(freq)

    else:
        print("No valid option selected")
        data = None
    return data


def plots_init(data, option, ch_num):
    """
    Initialize a waterfall plot and frequency spectrum plot.

    Parameters:
        RATE (int): Sampling rate.
        chunk (int): Size of each audio chunk.
        duration (float): Duration of the plot in seconds.

    Returns:
        Figure object, axes object for the waterfall plot, axes object
        for the frequency-domain plot, image object.
    """

    # Initialize waterfall plot
    fig, (ax, ax_f) = plt.subplots(1, 2, figsize=(12, 5))
    data = np.zeros((RATE // CHUNK * DURATION, CHUNK // 2))

    positions = [0, 0.85, 0.88, 0.91, 0.94, 0.97, 0.99, 1] # [0, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1]
    colors = [
        "white",
        "green",
        "lightgreen",
        "yellow",
        "orange",
        "red",
        "darkred",
        "black",
    ]
    cmap = LinearSegmentedColormap.from_list(
        "custom_colormap", list(zip(positions, colors))
    )
    im = ax.imshow(
        data, aspect="auto", origin="lower", norm=LogNorm(vmin=1, vmax=100), cmap=cmap
    )
    plt.colorbar(im)

    ax.set_title("Waterfall")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Time Elapsed [s]")
    # ax.set_yticks(np.arange(0, DURATION + 0.2, 0.2))
    # ax.set_yticklabels([0, 3.24 * 4, 3.24 * 8, 3.24 * 12, round(3.24 * 20, 2), 3.24 * 25][::-1])
    # ax.set_yticklabels([0, 0.4, 0.8, 1.2, 1.6, 1.8, 2.1, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6,][::-1])
    # Set y-tick positions: only show ticks at the bottom and top
    # Define y-tick positions: bottom, middle, and top
    yticks_positions = [0, DURATION / 2, DURATION]  # Add middle value
    ax.set_yticks(yticks_positions)

    # Define y-tick labels corresponding to these positions
    yticks_labels = [f'{yticks_positions[2]:.1f}', f'{yticks_positions[1]:.1f}', f'{yticks_positions[0]:.1f}']
    ax.set_yticklabels(yticks_labels)

    plt.ion()

    # PSD plot setup
    ax_f.set_title("SPL vs Frequency")
    ax_f.set_xlabel("Frequency [Hz]")
    ax_f.set_ylim(-40, 150)

    if option == 0 or option == 2:
        fig.suptitle(f"A-filtered Spectrum (ch{ch_num})", fontsize=16)
        ax_f.set_ylabel("SPL [dBA]")
    elif option == 1:
        fig.suptitle(f"Power Spectrum (ch{ch_num})", fontsize=16)
        ax_f.set_ylabel("Power [dB]")

    (line_f1,) = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))
    (line_f2,) = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))

    return fig, ax, ax_f, im, line_f1, line_f2


def read_audio_data(stream, audio_data, index):
    audio_data[index] = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

def audio_acquisition(streams, buffers):
    global buffer_counter, buffer_ready

    n_channels = len(streams)  # Number of channels
    audio_data = np.zeros((n_channels, CHUNK), dtype=np.int16)
    
    # Create and start threads for reading audio data
    threads = []
    for i, stream in enumerate(streams):
        thread = threading.Thread(target=read_audio_data, args=(stream, audio_data, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Initialize spectra with the correct shape
    spectra = np.zeros((n_channels, CHUNK), dtype=np.float32)  # Change dtype if needed

    # Process audio data
    if N == 1:
        buffer_ready = 1
        buffer_counter = 0
        spectra = audio_data
    else:
        if buffer_counter < N:
            for i in range(n_channels):
                buffers[i][:, buffer_counter] = audio_data[i]
        elif buffer_counter >= N:
            buffer_counter = 0
            buffer_ready = 1
            for i in range(n_channels):
                spectra[i] = np.mean(buffers[i], axis=1)

    return spectra

def initialize_pyaudio(n_streams):
    # CHANGE THIS FUNCTION TO ADD MICROPHONES
    p = pyaudio.PyAudio()
    streams = []
    for i in range(n_streams):
        # You can specify the input device here when usb
        # soundcards are plugged in.
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        streams.append(stream)
    
    return streams

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Index: {i}, Name: {device_info['name']}, Max Input Channels: {device_info['maxInputChannels']}")

def stop_streams(streams):
    for stream in streams:
        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error stopping or closing stream: {e}")

def update_plots(data, spectrum, freq, im, line):

    print(time.time())


    # Populate/Update Waterfall Plot
    data = np.roll(data, -1, axis=0)
    data[-1, :] = spectrum
    im.set_data(data)
    im.set_extent([freq[0], freq[-1], 0, DURATION])

    # Populate/Update Power vs Frequency Plot
    line.set_data(freq, spectrum)

    return data

def boundary_check(spectrum, thresholds_1D):

    temp = np.max(spectrum)

    colour = "grey"
    if (temp > 82) and (temp < 85):
        # A
        thresholds_1D[0] = 1
        colour = "green"
    if (temp > 85) and (temp < 88):
        # B
        thresholds_1D[1] = 1
        colour = "lightgreen"
    if (temp > 88) and (temp < 91):
        # C
        thresholds_1D[2] = 1
        colour = "yellow"
    if temp > 91 and (temp < 94):
        # D
        thresholds_1D[3] = 1
        colour = "orange"
    if temp > 94 and (temp < 97):
        # E
        thresholds_1D[4] = 1
        colour = "red"
    if temp > 97 and (temp < 100):
        # F
        thresholds_1D[5] = 1
        colour = "darkred"
    if temp > 100:
        # G
        thresholds_1D[6] = 1
        colour = "black"

    return colour, thresholds_1D


if __name__ == "__main__":
    """
    Main function to run the simulated design for the lab nuc and sound card.
    """
    # remove this after profiling is done
    profile = cProfile.Profile()
    profile.enable()

    parser = OptionParser()
    parser.add_option(
        "-a",
        "--aFilter",
        help="Apply a-filter in time domain",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "-f",
        "--freq",
        help="Apply a-filter in frequency domain",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "-p",
        "--power",
        help="Display unfiltered power spectrum",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "-b", "--plot", help="Enable plotting", default=False, action="store_true"
    )
    parser.add_option(
        "-s", "--snr", help="Print SNR", default=False, action="store_true"
    )
    parser.add_option("-n", "--logn", help="Logs N-samples", metavar="N", type="int")
    parser.add_option(
        "-t",
        "--logt",
        help="Logs samples within a time period",
        metavar="TIME",
        type="int",
    )

    parser.add_option(
        "-c",
        "--logc",
        help="Logs samples continuously",
        default=False,
        action="store_true",
    )

    # 'options' is an object containing values for all of your options - e.g.
    # if --print takes a single string argument, then options.print will be
    # the filename supplied by the user.
    # 'args' is the list of positional arguments leftover after parsing
    (opts, args) = parser.parse_args()


try:

    # Menu Handeling
    if opts.aFilter:
        option = A_FILTER_TIME
        print("Time-implemented A-filter selected")

    if opts.freq:
        option = A_FILTER_FREQ
        print("Frequency-implemented A-filter selected")

    if opts.power:
        option = POWER
        print("Power mode selected")

    if opts.plot:
        isPloting = False
        print("Plotting disabled. Plotting is on by default")

    if opts.snr:
        isPrintingSNR = True
        print("SNR option selected. This is to be used with a tone at 1000Hz")

    if (opts.snr and opts.aFilter) or (opts.snr and opts.freq):
        print('SNR (-s) should only be displayed with the power option "-p".')
        print("Exiting program")
        exit()

    # Creating the log file if it hasn't been created already
    for i in range(n_streams):
        log_message(log_file, new_log_message, group_name=f"Channel {i+1}")

    # Networking
    # credentials = pika.PlainCredentials("nuctwo", "nuctwo")
    # connection = pika.BlockingConnection(
    #     pika.ConnectionParameters("192.168.10.195", credentials=credentials)
    # )
    connection = pika.BlockingConnection(
        pika.ConnectionParameters("localhost")
    )
    channel = connection.channel()
    channel.exchange_declare(exchange="log", exchange_type="fanout")

    # Initialize PyAudio
    streams = initialize_pyaudio(n_streams)
    stream_buffers = np.zeros((n_streams, CHUNK, N), dtype=np.int16)

    # Initialize data buffer
    data_ch1 = np.zeros(((RATE * DURATION) // CHUNK, CHUNK // 2))
    data_ch2 = np.zeros(((RATE * DURATION) // CHUNK, CHUNK // 2))
    spectra_formatted = np.zeros((n_streams, CHUNK // 2), dtype=np.float32)

    data_list = [np.zeros(((RATE * DURATION) // CHUNK, CHUNK // 2)) for _ in range(n_streams)]

    # Initialize watefall and spectrum plots
    fig = [None] * (n_streams)
    ax = [None] * (n_streams)
    ax_f = [None] * (n_streams)
    im = [None] * (n_streams)
    line_f1 = [None] * (n_streams)
    line_f2 = [None] * (n_streams)

    if isPloting:
        for ch_num in range(n_streams):
            fig[ch_num], ax[ch_num], ax_f[ch_num], im[ch_num], line_f1[ch_num], line_f2[ch_num] = plots_init(data_ch1, option, ch_num+1) # change this "1" later

    # Calculate frequency axis
    freq = np.fft.fftfreq(CHUNK, 1 / RATE)[0 : CHUNK // 2]

    counter = 0
    counter2 = 0
    buffer_counter = 0
    buffer_ready = 0

    # ch1_buffer = np.zeros((CHUNK, N))
    # ch2_buffer = np.zeros((CHUNK, N))

    while True:
        spectra = audio_acquisition(streams, stream_buffers)
        # Change this to spectra[1] when you add another mic

        if buffer_ready:
            buffer_ready = 0

            for i in range(n_streams):
                spectra_formatted[i] = data_format(option, spectra[i])
                spectra_formatted[i] = spectra_formatted[i].astype(np.float32)

                # ===================================================================
                # NETWORKING
                # ===================================================================

                # Convert NumPy array to Python list and serialize the list to JSON

                # Prepare message containing plot titles
                if option == A_FILTER_TIME or option == A_FILTER_FREQ:
                    plot_title = f"A-Filtered Spectrum (ch{i+1})"
                    ylabel = "SPL [dBA]"

                elif option == 1:
                    plot_title = f"Power Spectrum (ch{i+1})"
                    ylabel = "Power [dB]"

                message_body["nth_stream"] = i+1
                message_body[f"spectrum_ch{i+1}"] = spectra_formatted[i].tolist()
                message_body[f"plotTitle_ch{i+1}"] = plot_title
                message_body[f"ylabel_ch{i+1}"] = ylabel
      
            message_body_json = json.dumps(message_body)

            channel.basic_publish(exchange="log", routing_key="", body=message_body_json)

            # SNR calculation of a 1000Hz tone
            if isPrintingSNR:
                """
                This has to be done with a tone playing at 1000Hz from my
                mobile phone, because I tested which freq bins contain this
                spesific signal. I could expand this to automatically pick up
                a tone, i.e., a peak if need be.

                STILL NEEDS TO BE EDITED
                """

                signal = spectrum_ch1[184:187]
                # Average signal power
                p_signal = sum(signal**2) / 3
                noise = spectrum_ch1
                noise[range(184, 187)] = 0
                # Average noise power
                p_noise = sum(noise**2) / (len(noise) - 3)
                snr = 10 * np.log10(p_signal / p_noise)
                print(snr)

            # =====================================================================
            # PLOTS AND DATA LOGGING
            # =====================================================================
            
            for i in range(n_streams):
                colour[i], thresholds[i] = boundary_check(spectra_formatted[i], thresholds[i]) # change spectrum_ch1 to be a list later

            # ================================================================
            # PLOTS
            # ================================================================

            if isPloting:
                # Show history of spectrum with highest peak

                for i in range(n_streams):
                    temp[i] = np.max(spectra_formatted[i])
                    if temp[i] > ave[i]:
                        ave[i] = temp[i]
                        line_f2[i].set_data(freq, temp[i])
                        line_f2[i].set_color(colour[i]) # one colour per stream at any given time 

                if counter2 >= 15:
                    ave = [0] * n_streams
                    counter2 = 0

                for i in range(n_streams):
                    data_list[i] = update_plots(data_list[i], spectra_formatted[i], freq, im[i], line_f1[i]) # spectum/spectra needs work
                    plt.pause(0.05)
                    fig[i].canvas.flush_events()

                # data_ch2 = update_plots(data_ch2, spectrum_ch2, freq, im_ch2, line_f1_ch2)

            # =================================
            # Data Logging
            # =================================

            if opts.logn is None and opts.logt is None:
                for i in range(n_streams):
                    thresholds[i], prev_thresholds[i] = process_log_message(
                        i+1, spectra_formatted[i], freq, log_file,
                        thresholds[i], prev_thresholds[i], thresholds_flag=True # log_file needs to be changed to mach the list-like thing going on
                    )

            # Log N samples
            if opts.logn is not None:
                if opts.logn > 0:
                    if counter_n < opts.logn:
                        for i in range(n_streams):
                            thresholds[i], prev_thresholds[i] = process_log_message(
                                i+1, spectra_formatted[i], freq, log_file,
                                thresholds[i], prev_thresholds[i], N_samples=True
                            )
                        counter_n += 1
                    else:
                        print("Sample logging has stopped")
                        # exit program
                        connection.close()
                        plt.ioff()
                        stop_streams(streams)
                        # stream.stop_stream()
                        # stream.close()
                        # p.terminate()
                        exit()
                else:
                    print("Number of samples must be larger than 0")
                    # exit program
                    connection.close()
                    plt.ioff()
                    stop_streams(streams)
                    # stream.stop_stream()
                    # stream.close()
                    # p.terminate()
                    exit()

            if opts.logc:
                for i in range(n_streams):
                    thresholds[i], prev_thresholds[i] = process_log_message(
                        i+1, spectra_formatted[i], freq, log_file,
                        thresholds[i], prev_thresholds[i], continuous=True
                    )

            # Log samples within a time period
            if opts.logt is not None:
                if opts.logt > 0:
                    end_time = time.monotonic()
                    elapsed_time = end_time - start_time
                    if elapsed_time < opts.logt:
                        for i in range(n_streams):
                            thresholds[i], prev_thresholds[i] = process_log_message(
                                i+1, spectra_formatted[i], freq, log_file,
                                thresholds[i], prev_thresholds[i], time_period=True
                            )
                    else:
                        print("Time period logging has stopped")
                        # exit program
                        connection.close()
                        plt.ioff()
                        stop_streams(streams)
                        # stream.stop_stream()
                        # stream.close()
                        # p.terminate()
                        exit()
                else:
                    print("Time must be larger than 0")
                    # exit program
                    connection.close()
                    plt.ioff()
                    stop_streams(streams)
                    # stream.stop_stream()
                    # stream.close()
                    # p.terminate()
                    exit()

            # Take a sample at midnight
            now = datetime.datetime.now()
            if (0, 0) == (now.hour, now.minute) and 0 < now.second < 10:
                for i in range(n_streams):
                    thresholds[i], prev_thresholds[i] = process_log_message(i+1, spectra_formatted[i], freq, log_file,
                                        thresholds[i], prev_thresholds[i], midnight=True
                    )


                counter2 += 1
                
        # counter2 += 1
        buffer_counter += 1

except KeyboardInterrupt:
    # remove this after profiling is done:
    # profile.disable()
    # results = pstats.Stats(profile)
    # results.sort_stats("time")
    # results.print_stats()
    exit()
    print("Interrupted")
    connection.close()
    plt.ioff()
    stream.stop_stream()
    stream.close()
    p.terminate()
    exit()