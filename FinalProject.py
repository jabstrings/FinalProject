from scipy.io import wavfile
import scipy.io
import wave
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import matplotlib.figure as figure
from scipy.signal import butter, filtfilt
import argparse
import base64
import json
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from tkinter import ttk, filedialog as fd, messagebox
from tkinter.messagebox import showinfo

from matplotlib.backend_bases import button_press_handler
#Receiving sound data
	#check for .wav
		#if yes keep going
		#if no convert to .wav
	#check for number of channels
		#if 1 channel keep going
		#if 2 channels, make two .wav files one for left channel one for right channel
	#check for metadata
		#if yes remove metadata
		#if no keep going


#Plotting graphs and analysing data
#display waveform graph
def display_wave_1channel(data, length) :
	time = np.linspace(0., length, data.shape[0])
	plt.plot(time, data, label="Single channel")
	plt.xlabel("Time [s]")
	plt.ylabel("Amplitude")
	plt.show()
def display_wave_2channel(data, length) :
	time = np.linspace(0., length, data.shape[0])
	plt.plot(time, data[:, 0], label="Left channel")
	plt.plot(time, data[:, 1], label="Right channel")
	plt.legend()
	plt.xlabel("Time [s]")
	plt.ylabel("Amplitude")
	plt.show()

#compute RT60 for low medium and high frequency
# Band-pass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Load the audio file
sample_rate, data = wavfile.read("ClapIST.wav")

# Define the time vector
t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)

# Calculate the Fourier Transform of the signal
fft_result = np.fft.fft(data)
spectrum = np.abs(fft_result)  # Get magnitude spectrum
freqs = np.fft.fftfreq(len(data), d=1/sample_rate)

# Use only positive frequencies
freqs = freqs[:len(freqs)//2]
spectrum = spectrum[:len(spectrum)//2]

# Find the target frequency closest to 1000 Hz
def find_target_frequency(freqs, target=1000):
    nearest_freq = freqs[np.abs(freqs - target).argmin()]
    return nearest_freq

# Find the target frequency
target_frequency = find_target_frequency(freqs)

# Apply a band-pass filter around the target frequency
filtered_data = bandpass_filter(data, target_frequency - 50, target_frequency + 50, sample_rate)

# Convert the filtered audio signal to decibel scale
data_in_db = 10 * np.log10(np.abs(filtered_data) + 1e-10)  # Avoid log of zero

# Find the index of the maximum value
index_of_max = np.argmax(data_in_db)
value_of_max = data_in_db[index_of_max]

# Slice the array from the maximum value
sliced_array = data_in_db[index_of_max:]
value_of_max_less_5 = value_of_max - 5

# Function to find the nearest value in the array
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Find the nearest value for max-5dB and its index
value_of_max_less_5 = find_nearest_value(sliced_array, value_of_max_less_5)
index_of_max_less_5 = np.where(data_in_db == value_of_max_less_5)[0][0]

# Find the nearest value for max-25dB and its index
value_of_max_less_25 = value_of_max - 25
value_of_max_less_25 = find_nearest_value(sliced_array, value_of_max_less_25)
index_of_max_less_25 = np.where(data_in_db == value_of_max_less_25)[0][0]

# Calculate RT60 time
rt20 = t[index_of_max_less_5] - t[index_of_max_less_25]
rt60 = 3 * rt20

# Print RT60 value
print(f'The RT60 reverb time at freq {int(target_frequency)}Hz is {round(abs(rt60), 2)} seconds')


#display RT60 graphs for low medium and high frequency
#display difference in RT60 to reduce to 0.5 seconds
#display specgram graph

#Making Gui
def select_file():
    filetypes = (
        ('audio files', '*.mp3'),
        ('audio files', '*.wav')
    )

    filename = fd.askopenfilename(
        title='Open File',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )

root = tk.Tk()
root.title("SPIDAM Sound Analysis")
root.geometry("1500x600")

# frame
my_frame = ttk.LabelFrame(root, text="File Options", padding="5 5 5 5")
my_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)


# button to load audio file
file_button = ttk.Button(my_frame, text="Open File", command=select_file)
file_button.grid(row=0, column=0, padx=5, pady=5)


# button for specgram graph
analyze_button = ttk.Button(my_frame, text="Analyze")
analyze_button.grid(row=0, column=1, padx=5, pady=5)
# button

root.mainloop()