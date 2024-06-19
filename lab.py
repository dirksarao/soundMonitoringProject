import pyaudio
import numpy as np
import math
import pika
import json
import matplotlib.pyplot as plt
import sys
import time
import datetime
import h5py
import os
from matplotlib.colors import LogNorm
from numpy import pi, polymul
from scipy.signal import bilinear
from scipy.signal import lfilter
from optparse import OptionParser
from matplotlib.colors import LinearSegmentedColormap
'''
Combinations of CHUNK and N that works:
1. CHUNK = 2**13, N = 8
2. CHUNK = 2**10, N = 32
'''

# Parameters
CHUNK = 2**13 #2**13 is good here
RATE = 44100
DURATION = 1  # Duration of the plot in seconds
NYQUIST_RATE = RATE//2
A_FILTER_TIME = 0
POWER = 1
A_FILTER_FREQ = 2
isPloting = False # Change to False to stop program from plotting - for debuggin purposes
isPrintingSNR = False
C_FACTOR = 100 # Calibration factor to be changed when calibration takes place.
MICROPHONE_SENSITIVITY = 12.1*10**(-3)
REFERENCE_PRESSURE = 20*10**(-6)
option = A_FILTER_TIME
ave_ch1 = 0
ave_ch2 = 0
flag = 1
N = 8 #8 is good here
threshold_A_ch1 = 1
threshold_B_ch1 = 1
threshold_C_ch1 = 1
threshold_D_ch1 = 1
threshold_A_ch2 = 1
threshold_B_ch2 = 1
threshold_C_ch2 = 1
threshold_D_ch2 = 1
new_log_message = ""
new_size = 0
log_file = 'log.hdf5'

def log_message(log_file, dataset_name, spectrum=None):
    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, 'w'):
            pass

    with h5py.File(log_file, 'a') as f:
        # Create dataset for data array if provided
        if spectrum is not None:
            f.create_dataset(dataset_name, data=spectrum, dtype=np.float32)

def print_dataset_contents(log_file):

    print("Printing updated message")

    if not os.path.isfile(log_file):
        print("File '{}' does not exist.".format(log_file))
        return

    with h5py.File(log_file, 'r') as f:
        data_arrays = [f[name] for name in f.keys() if not isinstance(f[name], h5py.Group)]

        if len(data_arrays) == 0:
            print("No data arrays found.")

        print("Dataset names:")
        for name in f.keys():
            print("-", name)

        for data_array in data_arrays:
            print("\nDataset Name:", data_array.name)
            print(data_array[:])
        
    print("Finished printing updated message")


def a_filter(f_bin):
    """
    Returns the A-weight factor per frequency to be added to power spectrum
    """
    a_factor = []
    
    for i in range(len(f_bin)):
        a = (12194**2)*(f_bin[i]**4)
        b = (f_bin[i]**2)+(20.6**2)
        c = (f_bin[i]**2)+(107.7**2)
        d = (f_bin[i]**2)+(737.9**2)
        e = (f_bin[i]**2)+(12194**2)
        r_a = a/(b*math.sqrt(c*d)*e)
        #20*log10(r_a at 1000) is equal to 2.0 This is done for normalisation so that 0dB starts at 1000Hz 
        a_factor.append(20*np.log10(r_a) + 2.0)
    return a_factor

def A_weighting(fs):
    """Design of an A-weighting filter.
    b, a = A_weighting(fs) designs a digital A-weighting filter for
    sampling frequency `fs`. Usage: y = scipy.signal.lfilter(b, a, x).
    Warning: `fs` should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.
    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = polymul([1, 4*pi * f4, (2*pi * f4)**2],
                   [1, 4*pi * f1, (2*pi * f1)**2])
    DENs = polymul(polymul(DENs, [1, 2*pi * f3]),
                                 [1, 2*pi * f2])

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, fs)

def rms_flat(a):
    """
    Returns the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

def data_format(option, audio_data):
    """
    Allows you to select what amplitude/spectrum format to be displayed.

    Parameters:
        option (int): Indicates which format to return:
            - 0 is single-sided A-filtered spectrum
            - 1 is single-sided power spectrum
            - default is single-sided power spectrum
        audio_data (16 bit NumPy array of ints): Raw time domain data of the size of CHUNK
    
    Returns: NumPy array of spectrum format of choice
    """
    global counter2
    audio_data = audio_data - np.mean(audio_data)
    audio_data *= np.hamming(CHUNK)

    if option == 0:
        # ===================================
        # TIME-DOMAIN A-FILTER IMPLEMENTATION
        # ===================================
        b,a = A_weighting(RATE)
        y = lfilter(b, a, audio_data)
        magnitude_spectrum = abs(np.fft.fft(y/(100))[0:CHUNK//2]) #The denominator should be the voltage representation of 20 micropascals and this should be measured during calibration at the moment I am doing guess work here.
        data = 20 * np.log10(magnitude_spectrum)    

    elif option == 1:
        # ===================================
        # POWER
        # ===================================
        magnitude_spectrum = abs(np.fft.fft(audio_data)[0:CHUNK//2])
        data = 10*np.log10((magnitude_spectrum**2))

    elif option == 2:
        # ===================================
        # FREQUENCY-DOMAIN A-FILTER IMPLEMENTATION
        # ===================================
        magnitude_spectrum = abs(np.fft.fft(audio_data/100)[0:CHUNK//2]) #The denominator should be the voltage representation of 20 micropascals and this should be measured during calibration at the moment I am doing guess work here.
        data = 10*np.log10(magnitude_spectrum**2) + a_filter(freq)

    else:
        print("No valid option selected")
        data = None
    return data

def plots_init(rate, chunk, duration, data, data_format):
    """
    Initialize a waterfall plot and frequency spectrum plot.

    Parameters:
        RATE (int): Sampling rate.
        chunk (int): Size of each audio chunk.
        duration (float): Duration of the plot in seconds.

    Returns:
        Figure object, axes object for the waterfall plot, axes object for the frequency-domain plot, image object.
    """

    # Initialize waterfall plot
    fig, (ax, ax_f) = plt.subplots(1,2,figsize=(12, 5))
    data = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))

    positions = [0, 0.67, 0.75, 0.83, 0.92, 1]
    colors = ['white', 'green', 'yellow', 'orange', 'red', 'black']
    cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
    im = ax.imshow(data, aspect='auto', origin='lower', norm=LogNorm(vmin=1, vmax=120), cmap=cmap) # vmin = -50, vmax = 120
    plt.colorbar(im)

    ax.set_title('Waterfall')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Time Elapsed [s]') #This is what it was: Time [s]'
    # ax.set_yticks([])  # Hide y-axis numbers
    ax.set_yticks(np.arange(0, DURATION+0.2, 0.2))
    ax.set_yticklabels([0, 1.45, 1.45*2, 1.45*3, 1.45*4, 1.45*5][::-1]) # [1.921, 1.921+1.45, 1.921+1.45*2, 1.921+1.45*3, 1.921+1.45*4]

    plt.ion()

    # PSD plot setup
    ax_f.set_title('SPL vs Frequency')
    ax_f.set_xlabel('Frequency [Hz]')
    ax_f.set_ylim(-40, 150)

    if data_format==0 or data_format==2:
        fig.suptitle('A-filtered Spectrum', fontsize=16)  # Set big title
        ax_f.set_ylabel('SPL [dBA]')
    elif data_format==1:
        fig.suptitle('Power Spectrum', fontsize=16)  # Set big title
        ax_f.set_ylabel('Power [dB]')

    x = np.arange(0, CHUNK)
    y_time = np.zeros(CHUNK)
    line_f1, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))
    line_f2, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))


    return fig, ax, ax_f, im, line_f1, line_f2

if __name__ == '__main__':
    """
    Main function to run the simulated design for the lab nuc and sound card. To be edited later.
    """
    parser = OptionParser()
    
    parser.add_option("-a", "--aFilter", help='Apply a-filter in time domain', default = False, action = 'store_true') # Time-domain implementation
    parser.add_option("-f", "--freq", help='Apply a-filter in frequency domain', default = False, action = 'store_true')# Frequency domain implementation
    parser.add_option("-p", "--power", help='Display unfiltered power spectrum', default = False, action = 'store_true')
    parser.add_option("-b", "--plot", help='Enable plotting', default = False, action = 'store_true')
    parser.add_option("-s", "--snr", help='Print SNR', default = False, action = 'store_true')

    # 'options' is an object containing values for all of your options - e.g. if --print takes a single string argument, then options.print will be the filename supplied by the user.
    # 'args' is the list of positional arguments leftover after parsing
    (opts, args) = parser.parse_args()
    
try:

    # Menu Handeling
    if opts.aFilter:
        option = A_FILTER_TIME
    
    if opts.freq:
        option = A_FILTER_FREQ

    if opts.power:
        option = POWER

    if opts.plot:
        isPloting = True

    if opts.snr:
        isPrintingSNR = True

    if (opts.snr and opts.aFilter) or (opts.snr and opts.freq):
        print('SNR (-s) should only be displayed with the power option "-p".')
        print('Exiting program')
        exit()

    #Creating the log file if it hasn't been created already
    log_message(log_file, new_log_message)

    # Networking
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='log', exchange_type='fanout')

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=RATE, input=True, frames_per_buffer=CHUNK)

    #Initialize data buffer
    data_ch1 = np.zeros(((RATE * DURATION)// CHUNK, CHUNK//2))
    data_ch2 = np.zeros(((RATE * DURATION)// CHUNK, CHUNK//2))


    #Initialize watefall and spectrum plots
    if isPloting:
        fig, ax, ax_f, im, line_f1, line_f2 = plots_init(RATE, CHUNK, DURATION, data_ch1, option) #add parameter for data format
        fig_ch2, ax_ch2, ax_f_ch2, im_ch2, line_f1_ch2, line_f2_ch2 = plots_init(RATE, CHUNK, DURATION, data_ch2, option) #add parameter for data format


    # Calculate frequency axis
    freq = np.fft.fftfreq(CHUNK, 1 / RATE)[0:CHUNK//2]

    counter = 0
    counter2 = 0
    buffer_counter = 0
    buffer_ready = 0

    ch1_buffer = np.zeros((CHUNK,N))
    ch2_buffer = np.zeros((CHUNK,N))

    # spectrum_ch1 = np.zeros((CHUNK//2))
    # spectrum_ch2 = np.zeros((CHUNK//2))

    while True:
        # Grab Audio Data
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        ch1 = audio_data[0::2]
        ch2 = audio_data[1::2]

        if buffer_counter < N:
            ch1_buffer[:,buffer_counter] = ch1
            ch2_buffer[:,buffer_counter] = ch2
        elif buffer_counter >= N:
            buffer_counter = 0
            buffer_ready = 1

            spectrum_ch1 = np.mean(ch1_buffer, axis=1)
            spectrum_ch2 = np.mean(ch2_buffer, axis=1)

        if buffer_ready:
            buffer_ready = 0
            spectrum_ch1 = data_format(option, spectrum_ch1)
            spectrum_ch1 = spectrum_ch1.astype(np.float32)

            spectrum_ch2 = data_format(option, spectrum_ch2)
            spectrum_ch2 = spectrum_ch2.astype(np.float32)

            # =====================================================================
            # NETWORKING
            # =====================================================================

            # Convert NumPy array to Python list and serialize the list to JSON

            # Prepare message containing plot titles
            if option == A_FILTER_TIME or option == A_FILTER_FREQ:
                plot_title = "A-Filtered Spectrum"
                ylabel = "SPL [dBA]"

            elif option == 1:
                plot_title = "Power Spectrum"
                ylabel = "Power [dB]"

            message_body = json.dumps({
                "spectrum_ch1":spectrum_ch1.tolist(),
                "spectrum_ch2":spectrum_ch2.tolist(),
                "plotTitle": plot_title,
                "ylabel": ylabel
            })

            channel.basic_publish(exchange='log',
                                routing_key='',
                                body=message_body
            )

            # SNR calculation of a 1000Hz tone
            if isPrintingSNR:
                '''
                This has to be done with a tone playing at 1000Hz from my mobile phone, because I tested which freq bins contain this spesific signal.
                I could expand this to automatically pick up a tone, i.e., a peak if need be.
                '''
                signal = spectrum_ch1[184:187]
                p_signal = sum(signal**2)/3 # Average signal power
                noise = spectrum_ch1
                noise[range(184,187)] = 0
                p_noise = sum(noise**2)/(len(noise)-3) # Average noise power
                snr = 10*np.log10(p_signal/p_noise)
                print(snr)
                
            # =====================================================================
            # PLOTS
            # =====================================================================

            if isPloting:

                # Show history of spectrum with highest peak
                temp_ch1 = np.max(spectrum_ch1)
                temp_ch2 = np.max(spectrum_ch2)

                if (temp_ch1 > ave_ch1):
                    ave_ch1 = temp_ch1
                    line_f2.set_data(freq, temp_ch1)
                    line_f2.set_color('green')
                    if((temp_ch1>80) and (temp_ch1<90)):
                        line_f2.set_color('yellow')
                    if((temp_ch1>90) and (temp_ch1<100)):
                        line_f2.set_color('orange')
                    if(temp_ch1>100 and (temp_ch1<110)):
                        line_f2.set_color('red')
                    if(temp_ch1>110):
                        line_f2.set_color('black')

                if (temp_ch2 > ave_ch2):
                    ave_ch2 = temp_ch2
                    line_f2_ch2.set_data(freq, temp_ch2)
                    line_f2_ch2.set_color('green')
                    if((temp_ch2>80) and (temp_ch2<90)):
                        # threshold_A = threshold_A + 1
                        line_f2_ch2.set_color('yellow')
                    if((temp_ch2>90) and (temp_ch2<100)):
                        # threshold_B = threshold_B + 1
                        line_f2_ch2.set_color('orange')
                    if(temp_ch2>100 and (temp_ch2<110)):
                        # threshold_C = threshold_C + 1
                        line_f2_ch2.set_color('red')
                    if(temp_ch2>110):
                        # threshold_D = threshold_D + 1
                        line_f2_ch2.set_color('black')
                
                if counter2 >= 15:
                    ave_ch1 = 0
                    ave_ch2 = 0
                    counter2 = 0

                # Populate/Update Waterfall Plot
                data_ch1 = np.roll(data_ch1, -1, axis=0)
                data_ch1[-1, :] = spectrum_ch1
                data_ch2 = np.roll(data_ch2, -1, axis=0)
                data_ch2[-1, :] = spectrum_ch2

                im.set_data(data_ch1)
                im.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

                im_ch2.set_data(data_ch2)
                im_ch2.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

                # Populate/Update Power vs Frequency Plot
                line_f1.set_data(freq, spectrum_ch1)
                line_f1_ch2.set_data(freq, spectrum_ch2)
  
                # =================================
                # Data Logging
                # =================================
                # This can be changed:
                #ch1
                if((temp_ch2>80) and (temp_ch2<90)):
                    threshold_A_ch1 = threshold_A_ch1 + 1
                if((temp_ch2>90) and (temp_ch2<100)):
                    threshold_B_ch1 = threshold_B_ch1 + 1
                if(temp_ch2>100 and (temp_ch2<110)):
                    threshold_C_ch1 = threshold_C_ch1 + 1
                if(temp_ch2>110):
                    threshold_D_ch1 = threshold_D_ch1 + 1

                #ch2
                if((temp_ch2>80) and (temp_ch2<90)):
                    threshold_A_ch2 = threshold_A_ch2 + 1
                if((temp_ch2>90) and (temp_ch2<100)):
                    threshold_B_ch2 = threshold_B_ch2 + 1
                if(temp_ch2>100 and (temp_ch2<110)):
                    threshold_C_ch2 = threshold_C_ch2 + 1
                if(temp_ch2>110):
                    threshold_D_ch2 = threshold_D_ch2 + 1

                #ch1
                spec_and_freq = np.vstack((spectrum_ch1, freq))
                if threshold_A_ch1>=2:
                    new_log_message = "lab.py: Peak:" + np.max(spectrum_ch1) + "dBA Region: 80dBA < SPL < 90dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_A_ch1 = 0
                elif threshold_B_ch1>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch1) + "dBA Region: 90dBA < SPL < 100dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_B_ch1 = 0
                elif threshold_C_ch1>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch1) + "dBA 100dBA < SPL < 110dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_C_ch1 = 0
                elif threshold_D_ch1>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch1) + "dBA SPL > 110dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_D_ch1 = 0

                #ch2
                spec_and_freq = np.vstack((spectrum_ch2, freq))
                if threshold_A_ch2>=2:
                    new_log_message = "lab.py: Peak:" + np.max(spectrum_ch2) + "dBA Region: 80dBA < SPL < 90dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_A_ch2 = 0
                elif threshold_B_ch2>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch2) + "dBA Region: 90dBA < SPL < 100dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_B_ch2 = 0
                elif threshold_C_ch2>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch2) + "dBA 100dBA < SPL < 110dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_C_ch2 = 0
                elif threshold_D_ch2>=2:
                    new_log_message = "lab.py:" + np.max(spectrum_ch2) + "dBA SPL > 110dBA @ " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message(log_file, new_log_message, spectrum = spec_and_freq)
                    threshold_D_ch2 = 0
                
                #for debugging
                print_dataset_contents(log_file)


                counter2 += 1 #basically keeping track of how many buffer_counters are triggered
                plt.pause(0.05)
                fig.canvas.flush_events()

        # counter2 += 1    
        buffer_counter += 1

except KeyboardInterrupt:
    print('Interrupted')
    connection.close()
    plt.ioff()
    stream.stop_stream()
    stream.close()
    p.terminate()
    exit()