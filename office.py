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
titleReceived = 0
counter = 0
ave = 0
counter2 = 0

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
    ax_f.set_title('SPL vs Frequency')
    ax_f.set_xlabel('Frequency (Hz)')
    ax_f.set_ylabel('Power')
    ax_f.set_ylim(0, 150)

    x = np.arange(0, CHUNK)
    y_time = np.zeros(CHUNK)
    line_f1, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))
    line_f2, = ax_f.plot(np.arange(0, NYQUIST_RATE), np.zeros(NYQUIST_RATE))

    return fig, ax, ax_f, im, line_f1, line_f2

def callback(ch, method, properties, body):
    """
    Callback function for handeling received messages from the message broker.
    """
    global data, plot_title, ylabel, titleReceived, ave, counter2

    # spectrum = np.array(json.loads(body))
    message_data = json.loads(body)

    # Check if the message contains spectrum data or plot titles
    if 'spectrum' in message_data:
        # Deserialize JSON string to Python list then convert Python list to NumPy array
        spectrum = np.array(message_data['spectrum'])

        # Populate/Update Waterfall Plot
        data = np.roll(data, -1, axis=0)
        data[-1, :] = spectrum
        im.set_data(data)
        im.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

        # Populate/Update Power vs Frequency Plot
        line_f1.set_data(freq, spectrum)

        plt.pause(0.05)
        fig.canvas.flush_events()

    if (titleReceived==0) and ('plotTitle' in message_data):
        titleReceived = 1
        print("Received plot titles:")

        plot_title = message_data['plotTitle']
        ylabel = message_data['ylabel']

        fig.suptitle(plot_title, fontsize=16)
        ax_f.set_ylabel(ylabel)

    # Show history of spectrum with highest peak
    temp = np.average(spectrum)
    if (temp > ave):
        print("temp is bigger than max")
        ave = temp
        line_f2.set_data(freq, spectrum)

    if counter2 >= 100:
        ave = 0
        counter2 = 0

    counter2 += 1

    # # Populate/Update Waterfall Plot
    # data = np.roll(data, -1, axis=0)
    # data[-1, :] = spectrum
    # im.set_data(data)
    # im.set_extent([freq[0], freq[-1], 0, DURATION])  # Update x-axis data: Only set the extent along the x-axis

    # # Populate/Update Power vs Frequency Plot
    # line_f1.set_data(freq, spectrum)

    # plt.pause(0.05)
    # fig.canvas.flush_events()

    ch.basic_ack(delivery_tag = method.delivery_tag)

def reject_unacknowledged_messages():
    while True:
        method_frame, _, _ = channel.basic_get(queue=audioBuffer1)
        if method_frame:
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
        else:
            break

# Networking
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
fig, ax, ax_f, im, line_f1, line_f2 = plots_init(RATE, CHUNK, DURATION, data)

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
