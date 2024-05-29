import pyaudio
import numpy as np
import math
import os
import json
import pika
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Parameters
CHUNK = 2**10
RATE = 44100
DURATION = 1  # Duration of the plot in seconds
NYQUIST_RATE = RATE//2
C_FACTOR = 1 # Calibration factor to be changed when calibration takes place
POWER = 0
A_FILTER = 1

#Initialize data buffer
data = np.zeros((RATE // CHUNK * DURATION, CHUNK//2))

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
    im = ax.imshow(data, aspect='auto', origin='lower', norm=LogNorm(vmin=1, vmax=400))
    plt.colorbar(im)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (s)')

    plt.ion()

    # PSD plot setup
    # fig_f, ax_f = plt.subplots()
    ax_f.set_title('Frequency-Domain Signal')
    ax_f.set_xlabel('Frequency (Hz)')
    ax_f.set_ylabel('Power')
    ax_f.set_ylim(0, 150)

    x = np.arange(0, CHUNK)
    y_time = np.zeros(CHUNK)
    line_f, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))

    return fig, ax, ax_f, im, line_f

def callback(ch, method, properties, body):
    """
    Callback function for handeling received messages from the message broker.
    """
    global data

    # Deserialize JSON string to Python list then convert Python list to NumPy array
    spectrum = np.array(json.loads(body))

    # Populate/Update Waterfall Plot
    data = np.roll(data, -1, axis=0)
    data[-1, :] = spectrum
    im.set_data(data)
    im.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

    # Populate/Update Power vs Frequency Plot
    line_f.set_data(freq, spectrum)

    plt.pause(0.05)
    fig.canvas.flush_events()

# Networking
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='audioBuffer')
channel.basic_consume(queue='audioBuffer',
                      auto_ack=True,
                      on_message_callback=callback)

#Initialize watefall and spectrum plots
fig, ax, ax_f, im, line_f = plots_init(RATE, CHUNK, DURATION, data)

# Calculate frequency axis
freq = np.fft.fftfreq(CHUNK, 1 / RATE)[0:CHUNK//2]

try: 
    # Start consuming messages
    print('Waiting for messages. To exit, press CTRL+C')
    channel.start_consuming()
except KeyboardInterrupt:
    print('Interrupted')
    # Close the stream and PyAudio
    plt.ioff()
    exit()
