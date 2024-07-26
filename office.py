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
CHUNK = 2**11 # was 2**13
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

#Initialize data buffer
data_ch1 = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))
data_ch2 = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))

def plots_init(rate, chunk, duration, data):
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
    positions = [0, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1]
    colors = ['white', 'green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred', 'black']
    #colors = ['white', 'green', 'yellow', 'orange', 'red', 'black']
    cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
    im = ax.imshow(data, aspect='auto', origin='lower', norm=LogNorm(vmin=1, vmax=100), cmap=cmap)
    plt.colorbar(im)

    ax.set_title('Waterfall')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Time Elapsed [s]')
    ax.set_yticks(np.arange(0, DURATION+0.2, 0.2))
    ax.set_yticklabels([0, 1.5, 1.5*2, 1.5*3, 1.5*4, 1.5*5][::-1])

    plt.ion()

    # PSD plot setup
    # fig_f, ax_f = plt.subplots()
    ax_f.set_title('SPL vs Frequency')
    ax_f.set_xlabel('Frequency [Hz]')
    ax_f.set_ylabel('Power')
    ax_f.set_ylim(-40, 150)

    x = np.arange(0, CHUNK)
    y_time = np.zeros(CHUNK)
    line_f1, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))
    line_f2, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))

    return fig, ax, ax_f, im, line_f1, line_f2

def callback(ch, method, properties, body):
    """
    Callback function for handeling received messages from the message broker.
    """
    global data_ch1, data_ch2, plot_title, ylabel, titleReceived, ave_ch1, ave_ch2, counter2

    # spectrum = np.array(json.loads(body))
    message_data = json.loads(body)

    # Check if the message contains spectrum data or plot titles
    if 'spectrum_ch1' in message_data:
        # Deserialize JSON string to Python list then convert Python list to NumPy array
        spectrum_ch1 = np.array(message_data['spectrum_ch1'])

        # Populate/Update Waterfall Plot
        data_ch1 = np.roll(data_ch1, -1, axis=0)
        data_ch1[-1, :] = spectrum_ch1
        im.set_data(data_ch1)
        im.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

        # Populate/Update Power vs Frequency Plot
        line_f1.set_data(freq, spectrum_ch1)

        plt.pause(0.05)
        fig.canvas.flush_events()

    if 'spectrum_ch2' in message_data:
        # Deserialize JSON string to Python list then convert Python list to NumPy array
        spectrum_ch2 = np.array(message_data['spectrum_ch2'])

        # Populate/Update Waterfall Plot
        data_ch2 = np.roll(data_ch2, -1, axis=0)
        data_ch2[-1, :] = spectrum_ch2
        im_ch2.set_data(data_ch2)
        im_ch2.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

        # Populate/Update Power vs Frequency Plot
        line_f1_ch2.set_data(freq, spectrum_ch2)

        plt.pause(0.05)
        fig.canvas.flush_events()

    if (titleReceived==0):
        titleReceived = 1
        print("Received plot titles:")
        if ('plotTitle_ch1' in message_data):
            plot_title = message_data['plotTitle_ch1']
            ylabel = message_data['ylabel']

            fig.suptitle(plot_title, fontsize=16)
            ax_f.set_ylabel(ylabel)

        if ('plotTitle_ch2' in message_data):
            plot_title = message_data['plotTitle_ch2']
            ylabel = message_data['ylabel']

            fig_ch2.suptitle(plot_title, fontsize=16)
            ax_f_ch2.set_ylabel(ylabel)

    # Show history of spectrum with highest average in the last 20 cycles
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
            line_f2_ch2.set_color('yellow')
        if((temp_ch2>90) and (temp_ch2<100)):
            line_f2_ch2.set_color('orange')
        if(temp_ch2>100 and (temp_ch2<110)):
            line_f2_ch2.set_color('red')
        if(temp_ch2>110):
            line_f2_ch2.set_color('black')

    if counter2 >= 5:
        ave_ch1 = 0
        ave_ch2 = 0
        counter2 = 0

    counter2 += 1

    ch.basic_ack(delivery_tag = method.delivery_tag)

    # remove this after profiling is done:
    #profile.disable()
    #results = pstats.Stats(profile)
    #results.sort_stats('time')
    #results.print_stats()
    #exit()

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
fig, ax, ax_f, im, line_f1, line_f2 = plots_init(RATE, CHUNK, DURATION, data_ch1)
fig_ch2, ax_ch2, ax_f_ch2, im_ch2, line_f1_ch2, line_f2_ch2 = plots_init(RATE, CHUNK, DURATION, data_ch2)


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
