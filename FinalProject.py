from scipy.io import wavfile
import scipy.io
import wave
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import matplotlib.figure as figure
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


#display RT60 graphs for low medium and high frequency
#display difference in RT60 to reduce to 0.5 seconds
#display specgram graph

#Making Gui
	#button to load audio file
	#button to combine plots into single plot
	#button for specgram graph

				#structure subject to change depending on needs