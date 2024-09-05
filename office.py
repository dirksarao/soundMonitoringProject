import pyaudio
import numpy as np
import math
import os
import json
import pika
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

# Parameters
CHUNK = 2**14 # was 2**11
RATE = 44100
DURATION = 1  # Duration of the plot in seconds
NYQUIST_RATE = RATE//2
C_FACTOR = 1 # Calibration factor to be changed when calibration takes place
POWER = 0
A_FILTER = 1
titleReceived = 0
counter = 0
ave_ch1 = 0
ave_ch2 = 0
counter2 = 0
counter = 0
n_streams = 1 # Change when chaning the number of microphones

thresholds = [[0] * 7 for _ in range(n_streams)]
prev_thresholds = [[0] * 7 for _ in range(n_streams)]
ave = [0] * n_streams

#Initialize data buffer
data_ch1 = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))
data_ch2 = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))
data = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))

def boundary_check(spectrum):

    temp = np.max(spectrum)

    colour = "grey"
    if (temp > 82) and (temp < 85):
        # A
        colour = "green"
    if (temp > 85) and (temp < 88):
        # B
        colour = "lightgreen"
    if (temp > 88) and (temp < 91):
        # C
        colour = "yellow"
    if temp > 91 and (temp < 94):
        # D
        colour = "orange"
    if temp > 94 and (temp < 97):
        # E
        colour = "red"
    if temp > 97 and (temp < 100):
        # F
        colour = "darkred"
    if temp > 100:
        # G
        colour = "black"

    return colour

def plots_init(data):
    """
    Initialize a waterfall plot and frequency spectrum plot.

    Parameters:
        rate (int): Sampling rate.
        chunk (int): Size of each audio chunk.
        duration (float): Duration of the plot in seconds.

    Returns:
        Figure object, axes object for the waterfall plot, axes object for the frequency-domain plot, image object.
    """
    # Initialize waterfall plot
    fig, (ax, ax_f) = plt.subplots(1,2,figsize=(12, 5))
    data = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))

    #positions = [0, 0.67, 0.75, 0.83, 0.92, 1]
    positions = [0, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1] # [0, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1]
    colors = ['white', 'green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred', 'black']
    #colors = ['white', 'green', 'yellow', 'orange', 'red', 'black']
    cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
    im = ax.imshow(data, aspect='auto', origin='lower', norm=LogNorm(vmin=1, vmax=120), cmap=cmap)
    plt.colorbar(im)

    ax.set_title('Waterfall')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Time Elapsed [s]')
    ax.set_yticks(np.arange(0, DURATION+0.2, 0.2))
    ax.set_yticklabels([0, 12, 24, 48, 60, 72][::-1])

    plt.ion()

    # PSD plot setup
    # fig_f, ax_f = plt.subplots()
    ax_f.set_title('SPL vs Frequency')
    ax_f.set_xlabel('Frequency [Hz]')
    ax_f.set_ylabel('SPL [dBA]')
    ax_f.set_ylim(-40, 150)

    line_f1, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))
    line_f2, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))

    return fig, ax, ax_f, im, line_f1, line_f2

def callback(ch, method, properties, body):
    """
    Callback function for handeling received messages from the message broker.
    """
    global data_ch1, data_ch2, plot_title, ylabel, titleReceived, ave_ch1, ave_ch2, counter2, data, ave

    # spectrum = np.array(json.loads(body))
    message_data = json.loads(body.decode('utf-8'))
            
    for i in range(n_streams):
        plot_title = message_data[f'plotTitle_ch{i+1}']
        ylabel = message_data[f"ylabel_ch{i+1}"]
        spectrum = np.array(message_data[f'spectrum_ch{i+1}'])

        data = update_plots(data, spectrum, freq, im[i], line_f1[i])
        fig[i].suptitle(plot_title, fontsize=16)
        ax[i].set_ylabel(ylabel)

        plt.pause(0.05)
        fig[i].canvas.flush_events()

        colour = boundary_check(spectrum)

        temp = np.max(spectrum)
        if temp > ave[i]:
            ave[i] = temp
            line_f2[i].set_data(freq, temp)
            line_f2[i].set_color(colour) # one colour per stream at any given time 

    if counter2 >= 5:
        ave = [0] * n_streams
        counter2 = 0

    counter2 += 1

    ch.basic_ack(delivery_tag = method.delivery_tag)

    # remove this after profiling is done:
    #profile.disable()
    #results = pstats.Stats(profile)
    #results.sort_stats('time')
    #results.print_stats()
    #exit()

def update_plots(data, spectrum, freq, im, line):

    # Populate/Update Waterfall Plot
    data = np.roll(data, -1, axis=0)
    data[-1, :] = spectrum
    im.set_data(data)
    im.set_extent([freq[0], freq[-1], 0, DURATION])

    # Populate/Update Power vs Frequency Plot
    line.set_data(freq, spectrum)

    return data

def reject_unacknowledged_messages():
    while True:
        method_frame, _, _ = channel.basic_get(queue=audioBuffer1)
        if method_frame:
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
        else:
            break

# remove this after profiling is done
profile = cProfile.Profile()
profile.enable()

# Networking
#credentials = pika.PlainCredentials('nuc_two','nuc_two')
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.exchange_declare(exchange='log', exchange_type='fanout')

result = channel.queue_declare(queue='', exclusive=True)
audioBuffer1 = result.method.queue

channel.queue_bind(exchange='log', queue=audioBuffer1)

print(' [*] Waiting for logs. To exit press CTRL+C')

channel.queue_purge(queue=audioBuffer1)

reject_unacknowledged_messages()

channel.basic_consume(queue=audioBuffer1,
                      auto_ack=False,
                      on_message_callback=callback)

#Initialize watefall and spectrum plots

fig = [None] * (n_streams)
ax = [None] * (n_streams)
ax_f = [None] * (n_streams)
im = [None] * (n_streams)
line_f1 = [None] * (n_streams)
line_f2 = [None] * (n_streams)

for ch_num in range(n_streams):
    fig[ch_num], ax[ch_num], ax_f[ch_num], im[ch_num], line_f1[ch_num], line_f2[ch_num] = plots_init(data)

# Calculate frequency axis
freq = np.fft.fftfreq(CHUNK, 1 / RATE)[0:CHUNK//2]

try: 
    # Start consuming messages
    print('Waiting for messages. To exit, press CTRL+C')
    channel.start_consuming()
    connection.close()
except KeyboardInterrupt:
    print('Interrupted')
    connection.close()
    # Close the stream and PyAudio
    plt.ioff()
    exit()
