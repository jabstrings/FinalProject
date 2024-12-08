from matplotlib import pyplot as plt

from numpy import ndarray, dtype, floating
from pydub import AudioSegment
from scipy.io import wavfile
import scipy.io.wavfile as scw
import scipy.io
import wave
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import scipy.io
import scipy.io as sc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import subprocess
from pydub import AudioSegment
from mutagen import mp3

from matplotlib.backend_bases import button_press_handler
#Receiving sound data
def processsounddata(filename):
    #check file extension
    splitExtension = filename.split('.')
    if splitExtension[1] == 'mp3':
        filename = convertfile(filename)

    #everything is now a wav file

    #check for channel number
    samplerate, data = wavfile.read(filename)
    channelNumber = {data.shape[len(data.shape) - 1]}
    if channelNumber == 2:
        #split audio
        mono_audio = filename.split_to_mono()
        file1 = mono_audio[0]
        file2 = mono_audio[1]
        file1.export("left_channel.wav", format="wav")
        file2.export("right_channel.wav", format="wav")

    #channels split, time to export
    if channelNumber == 1:
        return channelNumber, filename
    elif channelNumber == 2:
        return channelNumber, file1, file2

#convert the file if necessary
def convertfile(filename):
    #filename & path of new file
    split = filename.split('.')
    newfilename = split[0] + '.wav'

    #remove metadata first
    subprocess.run(["ffmpeg", "-y", "-i", filename, "-map_metadata", "-1", filename], check=True)
    subprocess.call(['ffmpeg', '-i', mp3, newfilename])
    filename = AudioSegment.from_wav(newfilename)

    #return it back to the filename
    return newfilename

#Plotting graphs and analysing data
#get length of the audio
def get_length(filename):
	audio_file = AudioSegment.from_file(filename)
	length = audio_file.duration_seconds
	return length

#get the data of the audio
def get_data(filename):
	sample_rate, data = wavfile.read(filename)
	return data

#get the sample rate of the audio
def get_sample_rate(filename):
	sample_rate, data = wavfile.read(filename)
	return sample_rate

#display waveform graph
def display_wave_1channel(data, length, fig, ax) :
	time = np.linspace(0., length, data.shape[0])
	ax.plot(time, data, label="Single channel")

def display_wave_2channel(data, length, fig, ax) :
	time = np.linspace(0., length, data.shape[0])
	ax.plot(time, data[:, 0], label="Left channel")
	ax.plot(time, data[:, 1], label="Right channel")

#compute RT60 for low medium and high frequency
# Band-pass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Find the target frequency closest to 1000 Hz
def find_target_frequency(freqs, target):
    nearest_freq = freqs[np.abs(freqs - target).argmin()]
    return nearest_freq

# Function to find the nearest value in the array
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#low rt60 function
def lowRT60(data, sample_rate, ax):
	# Define the time vector
	t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)

	# Calculate the Fourier Transform of the signal
	fft_result = np.fft.fft(data)
	spectrum = np.abs(fft_result)  # Get magnitude spectrum
	freqs = np.fft.fftfreq(len(data), d=1/sample_rate)

	# Use only positive frequencies
	freqs = freqs[:len(freqs)//2]
	spectrum = spectrum[:len(spectrum)//2]

	# Find the target frequency
	target = 250
	target_frequency = find_target_frequency(freqs, target)

	# Apply a band-pass filter around the target frequency
	filtered_data = bandpass_filter(data, target_frequency - 50, target_frequency + 50, sample_rate)

	# Convert the filtered audio signal to decibel scale
	data_in_db = 10 * np.log10(np.abs(filtered_data) + 1e-10)  # Avoid log of zero

	# Plot the filtered signal in decibel scale
	ax.plot(t, data_in_db, linewidth=1, alpha=0.7, color='#004bc6')

	# Find the index of the maximum value
	index_of_max = np.argmax(data_in_db)
	value_of_max = data_in_db[index_of_max]
	ax.plot(t[index_of_max], data_in_db[index_of_max], 'go')

	# Slice the array from the maximum value
	sliced_array = data_in_db[index_of_max:]
	value_of_max_less_5 = value_of_max - 5

	# Find the nearest value for max-5dB and its index
	value_of_max_less_5 = find_nearest_value(sliced_array, value_of_max_less_5)
	index_of_max_less_5 = np.where(data_in_db == value_of_max_less_5)[0][0]
	ax.plot(t[index_of_max_less_5], data_in_db[index_of_max_less_5], 'yo')

	# Find the nearest value for max-25dB and its index
	value_of_max_less_25 = value_of_max - 25
	value_of_max_less_25 = find_nearest_value(sliced_array, value_of_max_less_25)
	index_of_max_less_25 = np.where(data_in_db == value_of_max_less_25)[0][0]
	ax.plot(t[index_of_max_less_25], data_in_db[index_of_max_less_25], 'ro')

	# Calculate mid RT60 time
	rt20 = t[index_of_max_less_5] - t[index_of_max_less_25]
	lowrt60 = 3 * rt20

	# Print RT60 value
	print(f'The RT60 reverb time at freq {int(target_frequency)}Hz is {round(abs(lowrt60), 2)} seconds')

#mid rt60 function
def midRT60(data, sample_rate, ax):
	# Define the time vector
	t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)

	# Calculate the Fourier Transform of the signal
	fft_result = np.fft.fft(data)
	spectrum = np.abs(fft_result)  # Get magnitude spectrum
	freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)

	# Use only positive frequencies
	freqs = freqs[:len(freqs) // 2]
	spectrum = spectrum[:len(spectrum) // 2]

	# Find the target frequency
	target = 250
	target_frequency = find_target_frequency(freqs, target)

	# Apply a band-pass filter around the target frequency
	filtered_data = bandpass_filter(data, target_frequency - 50, target_frequency + 50, sample_rate)

	# Convert the filtered audio signal to decibel scale
	data_in_db = 10 * np.log10(np.abs(filtered_data) + 1e-10)  # Avoid log of zero

	# Plot the filtered signal in decibel scale
	ax.plot(t, data_in_db, linewidth=1, alpha=0.7, color='#ffa500')

	# Find the index of the maximum value
	index_of_max = np.argmax(data_in_db)
	value_of_max = data_in_db[index_of_max]
	ax.plot(t[index_of_max], data_in_db[index_of_max], 'go')

	# Slice the array from the maximum value
	sliced_array = data_in_db[index_of_max:]
	value_of_max_less_5 = value_of_max - 5

	# Find the nearest value for max-5dB and its index
	value_of_max_less_5 = find_nearest_value(sliced_array, value_of_max_less_5)
	index_of_max_less_5 = np.where(data_in_db == value_of_max_less_5)[0][0]
	ax.plot(t[index_of_max_less_5], data_in_db[index_of_max_less_5], 'yo')

	# Find the nearest value for max-25dB and its index
	value_of_max_less_25 = value_of_max - 25
	value_of_max_less_25 = find_nearest_value(sliced_array, value_of_max_less_25)
	index_of_max_less_25 = np.where(data_in_db == value_of_max_less_25)[0][0]
	ax.plot(t[index_of_max_less_25], data_in_db[index_of_max_less_25], 'ro')

	# Calculate mid RT60 time
	rt20 = t[index_of_max_less_5] - t[index_of_max_less_25]
	midrt60 = 3 * rt20

	# Print RT60 value
	print(f'The RT60 reverb time at freq {int(target_frequency)}Hz is {round(abs(midrt60), 2)} seconds')

#mid rt60 function
def highRT60(data, sample_rate, ax):
	# Define the time vector
	t = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)

	# Calculate the Fourier Transform of the signal
	fft_result = np.fft.fft(data)
	spectrum = np.abs(fft_result)  # Get magnitude spectrum
	freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)

	# Use only positive frequencies
	freqs = freqs[:len(freqs) // 2]
	spectrum = spectrum[:len(spectrum) // 2]

	# Find the target frequency
	target = 250
	target_frequency = find_target_frequency(freqs, target)

	# Apply a band-pass filter around the target frequency
	filtered_data = bandpass_filter(data, target_frequency - 50, target_frequency + 50, sample_rate)

	# Convert the filtered audio signal to decibel scale
	data_in_db = 10 * np.log10(np.abs(filtered_data) + 1e-10)  # Avoid log of zero

	# Plot the filtered signal in decibel scale
	ax.plot(t, data_in_db, linewidth=1, alpha=0.7, color='#009900')

	# Find the index of the maximum value
	index_of_max = np.argmax(data_in_db)
	value_of_max = data_in_db[index_of_max]
	ax.plot(t[index_of_max], data_in_db[index_of_max], 'go')

	# Slice the array from the maximum value
	sliced_array = data_in_db[index_of_max:]
	value_of_max_less_5 = value_of_max - 5

	# Find the nearest value for max-5dB and its index
	value_of_max_less_5 = find_nearest_value(sliced_array, value_of_max_less_5)
	index_of_max_less_5 = np.where(data_in_db == value_of_max_less_5)[0][0]
	ax.plot(t[index_of_max_less_5], data_in_db[index_of_max_less_5], 'yo')

	# Find the nearest value for max-25dB and its index
	value_of_max_less_25 = value_of_max - 25
	value_of_max_less_25 = find_nearest_value(sliced_array, value_of_max_less_25)
	index_of_max_less_25 = np.where(data_in_db == value_of_max_less_25)[0][0]
	ax.plot(t[index_of_max_less_25], data_in_db[index_of_max_less_25], 'ro')

	# Calculate mid RT60 time
	rt20 = t[index_of_max_less_5] - t[index_of_max_less_25]
	highrt60 = 3 * rt20

	# Print RT60 value
	print(f'The RT60 reverb time at freq {int(target_frequency)}Hz is {round(abs(highrt60), 2)} seconds')

#Analyze button command
def plotData(filename, ax):
	data = get_data(filename)
	sample_rate = get_sample_rate(filename)
	length = get_length(filename)
	lowRT60(data, sample_rate, ax)
	midRT60(data, sample_rate, ax)
	highRT60(data, sample_rate, ax)

#specgram graph
def specgram(data, sample_rate):
	spectrum, freqs, t, im = plt.specgram(data, Fs=sample_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
	cbar = plt.colorbar(im)
	plt.xlabel('Time (s)')
	plt.ylabel('Frequency (Hz)')
	cbar.set_label('Intensity (dB)')

#Making GUI
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
fig, ax = plt.subplots(figsize=(4, 3))
root.title("SPIDAM Sound Analysis")
root.geometry("1500x600")

# frame
my_frame = ttk.LabelFrame(root, text="File Options", padding="5 5 5 5")
my_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

# canvas1
canvas1 = FigureCanvasTkAgg(fig, master=my_frame)
canvas1.draw()
canvas1.get_tk_widget().grid(row=1, column=0, padx=1, pady=1)

# canvas2
canvas2 = FigureCanvasTkAgg(fig, master=my_frame)
canvas2.draw()
canvas2.get_tk_widget().grid(row=1, column=1, padx=1, pady=1)

# canvas3
canvas3 = FigureCanvasTkAgg(fig, master=my_frame)
canvas3.draw()
canvas3.get_tk_widget().grid(row=1, column=2, padx=1, pady=1)

# canvas4
canvas4 = FigureCanvasTkAgg(fig, master=my_frame)
canvas4.draw()
canvas4.get_tk_widget().grid(row=2, column=0, padx=1, pady=1)

#canvas5
canvas5 = FigureCanvasTkAgg(fig, master=my_frame)
canvas5.draw()
canvas5.get_tk_widget().grid(row=2, column=1, padx=1, pady=1)

#canvas6
canvas6 = FigureCanvasTkAgg(fig, master=my_frame)
canvas6.draw()
canvas6.get_tk_widget().grid(row=2, column=2, padx=1, pady=1)

# button to load audio file
file_button = ttk.Button(my_frame, text="Open File", command=select_file)
file_button.grid(row=0, column=0, padx=5, pady=5)

# button for graphs
analyze_button = ttk.Button(my_frame, text="Analyze", command=plotData("ClapIST.wav", ax))
analyze_button.grid(row=0, column=1, padx=5, pady=5)

root.mainloop()