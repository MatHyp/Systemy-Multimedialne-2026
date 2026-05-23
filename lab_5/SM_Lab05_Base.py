import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from docx import Document
from docx.shared import Inches
from io import BytesIO
import soundfile as sf

##########################################
### Settings #############################
##########################################

Test=True
Kwant_Test=True # test Kwant function

Interpolation_kind=["",""] # what flag for switching interpolation

PlotSettings={
    "Bits":[4,8,16,24],
    "Decimation":[2,4,6,10,24],
    "InterpolationFrequency":[2000,4000, 8000,11999, 16000, 16953, 24000, 41000]
}
ListeningSettings={
    "Bits":[4,8],
    "Decimation":[4,6,10,24],
    "InterpolationFrequency":[4000, 8000,11999, 16000, 16953]
}

OutputRaportFile = ".docx" 
OutputFolder="" # place for all your new audio files will be

##########################################
### Data Set #############################
##########################################

AudioDir = r'.' # Address of folder with files (do nor delete `r``)


SinFiles=[
    {"File":"","TimeMargin":[0,0.02]},
    ] # list of dicts with file names with Sinus singals and fragment that will be displayed
SingFiles=[] # list of file names of with Singing Voice

##########################################
### Functions to  ########################
##########################################

def plotAudio(Signal,Fs,axs,fsize,TimeMargin=[0,0.02]):
    pass

def Kwant(data,bit):

    return data

def decimation(Signal,Fs,step):
    NewSignal=Signal.copy()
    NewFs=Fs
    return NewSignal,NewFs

def interpolation(Signal,Fs,NewFs,kind):
    NewSignal=Signal.copy()
    return NewSignal


##########################################
### Main Program  ########################
##########################################

if Test:
    counter=1
    if Kwant_Test:
        T_X=[
            np.round(np.linspace(0,255,255,dtype=np.uint8)),
            np.round(np.linspace(np.iinfo(np.int32).min,np.iinfo(np.int32).max,1000,dtype=np.int32)),
            np.linspace(-1,1,10000),
        ]
        Bits=[1,2,4]
        for X in T_X:
            for bit in Bits:
                kwanted=Kwant(X,bit)
                print(f"Bits {bit} == {2**bit} values, unique values {np.unique(kwanted).size}. Dtype before {X.dtype} and after {kwanted.dtype}")
                plt.figure(counter)
                plt.plot(X,kwanted)
                plt.title(f"{bit} bit")
                counter+=1
                
    else:
        file=SinFiles[0]
        Signal, Fs = sf.read(os.path.join(AudioDir,file["File"]), dtype='float32') 
        # test decimation
        dec_Signal,dec_Fs=decimation(Signal,Fs,10)
        f,axs=plt.subplots(2,1,num=counter,figsize=(5,5)) 
        counter+=1
        plotAudio(Signal=dec_Signal,Fs=dec_Fs,axs=axs,fsize=2**12,TimeMargin=file["TimeMargin"])
        f.suptitle(f"{file['File']} Decimation step 10")
        # test interpolation
        for kind in Interpolation_kind:
            Int_Fs=16000
            Int_Signal=interpolation(Signal=Signal,Fs=Fs,NewFs=Int_Fs,kind=kind)
            f,axs=plt.subplots(2,1,num=counter,figsize=(5,5)) 
            counter+=1
            plotAudio(Signal=Int_Signal,Fs=Int_Fs,axs=axs,fsize=2**12,TimeMargin=file["TimeMargin"])
            f.suptitle(f"{file['File']} Interpolation {kind}")
        
        
    plt.show()
    
else:
    # generate raport
    document = Document()
    document.add_heading('Report',0) # tworzenie nagłówków druga wartość to poziom nagłówka 
    document.add_paragraph("Autor: ")
    document.add_paragraph("Proszę wstawić mi 2 jeżeli tego nie wyedytuję")
    document.add_section()
    document.add_heading("Sprawdzanie działania napisanych funkcji na podstawie wykresów",1)
    counter = 1 
    document.add_heading("Testowanie funkcji kwantyzującej",2)
    for file in SinFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file["File"]), dtype='float32') 
        for bit in PlotSettings["Bits"]:
            kSignal=Kwant(Signal,bit)
            f,axs=plt.subplots(2,1,num=counter,figsize=(5,5)) 

            plotAudio(Signal=kSignal,Fs=Fs,axs=axs,fsize=2**12,TimeMargin=file["TimeMargin"])
            f.suptitle(f"{file['File']} Kwantyzacja {bit}-bitów")
            memfile = BytesIO() 
            f.savefig(memfile)
            document.add_picture(memfile, width=Inches(6)) # set document size
            memfile.close()
            f.clf()
    document.add_heading("Testowanie funkcji decymującej",2)        
    for file in SinFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file["File"]), dtype='float32') 
        for step in PlotSettings["Decimation"]:
            dec_Signal,dec_Fs=decimation(Signal,Fs,step)
            f,axs=plt.subplots(2,1,num=counter,figsize=(5,5)) 

            plotAudio(Signal=dec_Signal,Fs=dec_Fs,axs=axs,fsize=2**12,TimeMargin=file["TimeMargin"])
            f.suptitle(f"{file['File']} Decimation step {step}")
            memfile = BytesIO() 
            f.savefig(memfile)
            document.add_picture(memfile, width=Inches(6)) # set document size
            memfile.close()
            f.clf()
    document.add_heading("Testowanie funkcji interpolujących",2)        
    for file in SinFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file["File"]), dtype='float32') 
        for Int_Fs in PlotSettings["InterpolationFrequency"]:
            for kind in Interpolation_kind:
                Int_Signal=interpolation(Signal=Signal,Fs=Fs,NewFs=Int_Fs,kind=kind)
                f,axs=plt.subplots(2,1,num=counter,figsize=(5,5)) 

                handle=plotAudio(Signal=Int_Signal,Fs=Int_Fs,axs=axs,fsize=2**12,TimeMargin=file["TimeMargin"])
                f.suptitle(f"{file['File']} Interpolation {kind} Fs {Int_Fs}")
                memfile = BytesIO() 
                f.savefig(memfile)
                document.add_picture(memfile, width=Inches(6)) # set document size
                memfile.close()
                f.clf()
                document.add_paragraph(f"Tu proszę dostosować do obsługi wyjścia waszej funkcji plot data {handle}")
                
    document.add_heading("Podsumowanie pierwszej części zadania",2)
    document.add_paragraph("Tu proszę zebrać wszystkie obserwacje na podstawie powyższych wykresów.")
    document.add_heading("Obserwacje na podstawie odsłuchanych plików ",1)
    # generating of audio files
    for file in SingFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file), dtype='float32') 
        sfile=file.split(os.sep)[-1].split('.')
        for bit in ListeningSettings["Bits"]:
            kSignal=Kwant(Signal,bit)
            nfile=f"{sfile[0]}_kwant_{bit}.wav"
            sf.write(os.path.join(OutputFolder,nfile),data=kSignal,samplerate=Fs)
        for step in ListeningSettings["Decimation"]:
            dec_Signal,dec_Fs=decimation(Signal,Fs,step)
            
            nfile=f"{sfile[0]}_dec_{step}.wav"
            sf.write(os.path.join(OutputFolder,nfile),data=dec_Signal,samplerate=dec_Fs)
            
        for Int_Fs in ListeningSettings["InterpolationFrequency"]:
            for kind in Interpolation_kind:
                Int_Signal=interpolation(Signal=Signal,Fs=Fs,NewFs=Int_Fs,kind=kind)
                
                nfile=f"{sfile[0]}_interp_{kind}_{Int_Fs}.wav"
                sf.write(os.path.join(OutputFolder,nfile),data=Int_Signal,samplerate=Int_Fs)
        
        
    document.add_paragraph("Tu proszę zebrać wszystkie obserwacje na podstawie odsłuchów (może w formie tabelki)")
    document.add_heading("Wnioski",1)
    document.add_paragraph("Tu proszę zebrać wszystkie wnioski na podstawie eksperymentów.")
    document.save(OutputRaportFile) 