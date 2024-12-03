# ????
import scipy.io as sc
import matplotlib.pyplot as plt

# GUI Imports
import argparse
import base64
import json
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import tkinter as tk
from tkinter import ttk, filedialog as fd, messagebox

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
	#display waveform graph/graphs
	#compute RT60 for low medium and high frequency
	#display RT60 graphs for low medium and high frequency
	#display difference in RT60 to reduce to 0.5 seconds
	#display specgram graph

# Making Gui

root = tk.Tk()
root.title("SPIDAM Sound Analysis")
root.geometry("1500x600")

# frame
my_frame = ttk.LabelFrame(root, text="File Options", padding="5 5 5 5")
my_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

#

# button to load audio file
file_button = ttk.Button(my_frame, text="Open File")
file_button.grid(row=0, column=0, padx=5, pady=5)

# button for specgram graph
analyze_button = ttk.Button(my_frame, text="Analyze")
analyze_button.grid(row=0, column=1, padx=5, pady=5)
# button

root.mainloop()


    # button to combine plots into single plot


            # structure subject to change depending on needs