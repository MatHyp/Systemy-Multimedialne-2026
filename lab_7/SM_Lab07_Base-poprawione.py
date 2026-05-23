import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import os
from docx import Document
from docx.shared import Inches
from io import BytesIO

##########################################
### Settings #############################
##########################################

Only_Tests=False
bit_test=8

DPCM_n=3
DPCM_predictor=np.mean

OutputRaportFile = ".docx" 
OutputFolder="" # place for all your new audio files will be

##########################################
### Data Set #############################
##########################################


AudioDir = r'SING'

SingFiles = [
    'sing_high1.wav', 
    'sing_high2.wav', 
    'sing_low1.wav', 
    'sing_low2.wav', 
    'sing_medium1.wav', 
    'sing_medium2.wav'
]
##########################################
### Functions to  ########################
##########################################

def Kwant(x, bit):
    poziomy = (2 ** bit) - 1
    y = np.round(x * poziomy) / poziomy
    return y
import numpy as np

def A_law_compress(x):
    A = 87.6
    y = np.zeros(x.shape)
    
    idx = np.abs(x) < (1.0 / A)
    
    y[idx] = np.sign(x[idx]) * ((A * np.abs(x[idx])) / (1 + np.log(A)))
    
    not_idx = np.logical_not(idx)
    
    y[not_idx] = np.sign(x[not_idx]) * ((1 + np.log(A * np.abs(x[not_idx]))) / (1 + np.log(A)))    
    return y


def A_law_decompress(y):
    A = 87.6
    x_prim = np.zeros(y.shape)
    
    granica = 1.0 / (1.0 + np.log(A))
    
    idx = np.abs(y) < granica
    
    x_prim[idx] = np.sign(y[idx]) * (np.abs(y[idx]) * (1.0 + np.log(A))) / A
    
    not_idx = np.logical_not(idx)
    
    x_prim[not_idx] = np.sign(y[not_idx]) * (np.exp(np.abs(y[not_idx]) * (1.0 + np.log(A)) - 1.0)) / A
    
    return x_prim    

def mu_law_compress(x):
    mu = 255.0
    y = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    return y

def mu_law_decompress(y):
    mu = 255.0
    x_prim = np.sign(y) * (((1.0 + mu)**np.abs(y) - 1.0) / mu)
    return x_prim

def DPCM_compress(x,bit):
    y=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=Kwant(x[i]-e,bit)
        e+=y[i]
    return y

def DPCM_decompress(y):
    x_prim = np.zeros(y.shape)
    e = 0
    for i in range(0, y.shape[0]):
        x_prim[i] = y[i] + e
        e = x_prim[i] 
    return x_prim

def DPCM_compress_pred(x,bit,n,predictor=np.mean): 
    y=np.zeros(x.shape)
    xp=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=Kwant(x[i]-e,bit)
        xp[i]=y[i]+e
        idx=(np.arange(i-n,i,1,dtype=int)+1)
        idx=np.delete(idx,idx<0)
        e=predictor(xp[idx])
    return y

def DPCM_decompress_pred(y, n, predictor=np.mean):
    xp = np.zeros(y.shape) 
    e = 0
    for i in range(0, y.shape[0]):
        xp[i] = y[i] + e
        
        idx = (np.arange(i-n, i, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)
        e = predictor(xp[idx])
        
    return xp
##########################################
### Main Program  ########################
##########################################


document = Document()
if not Only_Tests:
    # generate raport
    document.add_heading('Report',0) # tworzenie nagłówków druga wartość to poziom nagłówka 
    document.add_paragraph("Autor: ")
    document.add_paragraph("Proszę wstawić mi 2 jeżeli tego nie wyedytuję")
    document.add_section()
    document.add_heading('Wykresy testujące działanie algorytmów',1)

x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)

x_alaw_comp=A_law_compress(x)
x_alaw_comp=Kwant(x_alaw_comp,bit_test)
x_alaw_decomp=A_law_decompress(x_alaw_comp)

x_mulaw_comp=mu_law_compress(x)
x_mulaw_comp=Kwant(x_mulaw_comp,bit_test)
x_mulaw_decomp=mu_law_decompress(x_mulaw_comp)

y_alaw_decomp =A_law_decompress(Kwant(A_law_compress(y),bit_test))
y_mulaw_decomp =mu_law_decompress(Kwant(mu_law_compress(y),bit_test))

dpcm_c=DPCM_compress(y,bit_test)
dpcm_dec=DPCM_decompress(dpcm_c)
dpcm_c_p=DPCM_compress_pred(y,bit_test,n=DPCM_n,predictor=DPCM_predictor)
dpcm_dec_p=DPCM_decompress_pred(dpcm_c_p,n=DPCM_n,predictor=DPCM_predictor)

f1,axs=plt.subplots(1,2,num=1,figsize=(6,6)) 
f1.suptitle(f"Test kompresji law dla {bit_test} bitów")
axs[0].plot(x,x_alaw_comp,label="a_law")
axs[0].plot(x,x_mulaw_comp,label="mu_law")
axs[0].set_title("Sygnał po kompresji")
axs[0].legend()

axs[1].plot(x,x_alaw_decomp,label="a_law")
axs[1].plot(x,x_mulaw_decomp,label="mu_law")
axs[1].set_title("Sygnał po dekompresji")
axs[1].legend()

f2,axs=plt.subplots(5,1,num=2,figsize=(8,6)) 
f2.suptitle(f"Test dekompresji dla {bit_test} bitów")
axs[0].plot(x,y,label="Sygnał bazowy")
axs[0].set_title("Sygnał bazowy")

axs[1].plot(x,y_alaw_decomp,label="Sygnał po kompresji A-law")
axs[1].legend()

axs[2].plot(x,y_mulaw_decomp,label="Sygnał po kompresji mu-law")
axs[2].legend()

axs[3].plot(x,dpcm_dec,label="Sygnał po kompresji DPCM bez predykcji")
axs[3].legend()

axs[4].plot(x,dpcm_dec_p,label="Sygnał po kompresji DPCM z predykcją")
axs[4].legend()


if Only_Tests:
    # plt.show()
    f1.savefig('test_krzywe_kompresji.png')
    f2.savefig('test_sygnal_dekompresja.png')
else:
    memfile = BytesIO() 
    f1.savefig(memfile)
    document.add_picture(memfile, width=Inches(6)) # set document size
    memfile.close()
    f1.clf()
    memfile = BytesIO() 
    f2.savefig(memfile)
    document.add_picture(memfile, width=Inches(6)) # set document size
    memfile.close()
    f2.clf()  
    document.add_section()
    document.add_heading("Obserwacje na podstawie odsłuchanych plików ",1)
    document.add_paragraph("Tu proszę odpowiedzieć własnymi słowami na zadanie 2.1")
    document.add_heading("Zadanie 2.2",2)
    document.add_paragraph("W przypadku plików 8 bitowych jakość dzwieku jest nadał na dobrym poziomie nie odczuwam spadków jakości.")
    document.add_heading("Zadanie 2.3",2)
    document.add_paragraph("Tu proszę zamieścić treść dla zadania 2.3 (może być tabelka). Uwaga jak nie jesteście w stanie rozpoznać zawrtości nie musice słuchać dla niższych wartości bitowych")
    document.add_section()
    document.add_heading("Podsumowanie i Wnioski",1)
    document.add_paragraph("Algorytmy kompresji działają dobrze, kompresja staje sie nie rozpoznawalna przy kompresji dla nizjszej liczby bitow ale najsczesniej dopiero przy 3 bit dzwiek przetsaje byc rozpoznawalny. ")
    document.save(OutputRaportFile) 
    # Audio files Generator
    for file in SingFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file), dtype='float32') 
        sfile=file.split(os.sep)[-1].split('.')
        for bit in [8,7,6,5,4,3,2]:
            y_alaw_decomp =A_law_decompress(Kwant(A_law_compress(Signal),bit))
            y_mulaw_decomp =mu_law_decompress(Kwant(mu_law_compress(Signal),bit))



            dpcm_c=DPCM_compress(Signal,bit)
            dpcm_dec=DPCM_decompress(dpcm_c)
            dpcm_c_p=DPCM_compress_pred(Signal,bit,n=DPCM_n,predictor=DPCM_predictor)
            dpcm_dec_p=DPCM_decompress_pred(dpcm_c_p,n=DPCM_n,predictor=DPCM_predictor)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_A_LAW_{bit}b.wav"),data=y_alaw_decomp,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_mu_LAW_{bit}b.wav"),data=y_mulaw_decomp,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_DPCM_bp_{bit}b.wav"),data=dpcm_dec,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_DPCM_zp_{bit}b.wav"),data=dpcm_dec_p,samplerate=Fs)

            