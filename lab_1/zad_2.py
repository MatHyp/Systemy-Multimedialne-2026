from docx import Document
from docx.shared import Inches
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import soundfile as sf
import scipy.fftpack


data, fs = sf.read('SIN/sin_440Hz.wav', dtype='float32')  


def plotAudio(Signal,Fs,Folder,TimeMargin=[0,0.02]):

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(np.arange(0,Signal.shape[0])/Fs, Signal)
    plt.title("Sygnał w dziedzinie czasu")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.xlim(TimeMargin)
    plt.grid()

    plt.subplot(2,1,2)

    fsize=2**8
    yf = scipy.fftpack.fft(data,fsize)

    plt.plot(np.arange(0,Fs/2,Fs/fsize),)
    
    plt.title("Widmo częstotliwościowe")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [dB]")
    plt.grid()

    plt.tight_layout()

    plt.savefig(f"{Folder}/img_2.png", dpi=300)
    plt.close()

plotAudio(data,fs,"img_out/zad_2",[0,0.02])

