# ????
import scipy.io
import scipy.io as sc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import subprocess

# GUI Imports
import argparse
import base64
import json
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import tkinter as tk
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

def processSoundData(filename):
    #in theory this works
    extension = filename.split('.')
    if extension[1] != 'wav':
        #if not wav convert to wav
        #assuming the only other thing you're uploading is an mp3
        #creates new filename
        newfilename = extension[0] + '.wav'
        #converts to wav (in theory)
        subprocess.call(['ffmpeg', '-i', filename, newfilename])
        #assigns the name of the new file of to the filename object to make it easier
        filename = newfilename

        #convert to wav
    #else, keep going
    #check for channel number
    samplerate, data = wavfile.read(filename)
    channelNumber = {data.shape[len(data.shape) - 1]}
    if channelNumber == 2:
        #split into two files
    #channels split, here we go







#Plotting graphs and analysing data
	#display waveform graph/graphs
	#compute RT60 for low medium and high frequency
	#display RT60 graphs for low medium and high frequency
	#display difference in RT60 to reduce to 0.5 seconds
	#display specgram graph

# Making Gui
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


    # button to combine plots into single plot


            # structure subject to change depending on needs