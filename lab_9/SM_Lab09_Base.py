import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat=r'.'                                 # katalog z plikami wideo
plik="clip_1.mp4"                       # nazwa pliku
ile=20                                 # ile klatek odtworzyć? <0 - całość
key_frame_counter=4                     # co która klatka ma być kluczowa i nie podlegać kompresji
# plot_frames=np.array([30,45])           # automatycznie wyrysuj wykresy
plot_frames=np.array([11, 15, 19])
auto_pause_frames=np.array([25])        # automatycznie za pauzuj dla klatki
subsampling="4:2:2"                     # parametry dla chroma subsampling
dzielnik=1                              # dzielnik przy zapisie różnicy
wyswietlaj_kaltki=False                  # czy program ma wyświetlać klatki
# ROI = [[400, 500, 800, 900]]                  # wyświetlane fragmenty (można podać kilka )
ROI = [[400, 500, 781, 881]]

subsamplings_list = ["4:4:4", "4:2:2", "4:4:0", "4:2:0", "4:1:1", "4:1:0"]
dzielniki_list = [1, 2, 4]

# Utworzenie folderu na wyniki (żeby nie zabałaganić katalogu)
wyniki_dir = "wyniki_jakosc"
if not os.path.exists(wyniki_dir):
    os.makedirs(wyniki_dir)

##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        # w pełni skompresowane dane
        self.Y=None 
        self.Cb=None
        self.Cr=None 
        # dane bez kompresji strumieniowej w celu przyspieszenia obliczeń
        self.semi_Y=None
        self.semi_Cb=None
        self.semi_Cr=None

def Chroma_subsampling(L, subsampling):
    # Wycinamy informacje o kolorze za pomocą kroku w adresowaniu
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return L[:, ::2]     # Bierzemy co 2 kolumnę
    elif subsampling == "4:4:0":
        return L[::2, :]     # Bierzemy co 2 wiersz
    elif subsampling == "4:2:0":
        return L[::2, ::2]   # Bierzemy co 2 wiersz i co 2 kolumnę
    elif subsampling == "4:1:1":
        return L[:, ::4]     # Bierzemy co 4 kolumnę
    elif subsampling == "4:1:0":
        return L[::2, ::4]   # Bierzemy co 2 wiersz i co 4 kolumnę
    else:
        return L # domyślnie 4:4:4

def Chroma_resampling(L, subsampling):
    # Odtwarzamy informacje o kolorze powielając zachowane piksele
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return np.repeat(L, 2, axis=1) # Powielamy kolumny 2 razy
    elif subsampling == "4:4:0":
        return np.repeat(L, 2, axis=0) # Powielamy wiersze 2 razy
    elif subsampling == "4:2:0":
        # Powielamy wiersze 2 razy, a wynik powielamy w kolumnach 2 razy
        L_res = np.repeat(L, 2, axis=0)
        return np.repeat(L_res, 2, axis=1)
    elif subsampling == "4:1:1":
        return np.repeat(L, 4, axis=1) # Powielamy kolumny 4 razy
    elif subsampling == "4:1:0":
        # Powielamy wiersze 2 razy, a wynik powielamy w kolumnach 4 razy
        L_res = np.repeat(L, 2, axis=0)
        return np.repeat(L_res, 4, axis=1)
    else:
        return L

        
def frame_image_to_class(frame,subsampling):
    Frame_class = data()
    Frame_class.Y=frame[:,:,0].astype(int)
    Frame_class.Cb=Chroma_subsampling(frame[:,:,2].astype(int),subsampling)
    Frame_class.Cr=Chroma_subsampling(frame[:,:,1].astype(int),subsampling)
    return Frame_class


def frame_layers_to_image(Y,Cr,Cb,subsampling):  
    Cb=Chroma_resampling(Cb,subsampling)
    Cr=Chroma_resampling(Cr,subsampling)
    return np.dstack([Y,Cr,Cb]).clip(0,255).astype(np.uint8)

def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    ## TO DO 
    KeyFrame.Y=Frame_class.Y
    KeyFrame.Cb=Frame_class.Cb
    KeyFrame.Cr=Frame_class.Cr
    KeyFrame.semi_Y=Frame_class.Y
    KeyFrame.semi_Cb=Frame_class.Cb
    KeyFrame.semi_Cr=Frame_class.Cr
    return KeyFrame

def decompress_KeyFrame(KeyFrame, subsampling):
    Y = KeyFrame.semi_Y
    Cb = KeyFrame.semi_Cb
    Cr = KeyFrame.semi_Cr
    
    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image

def compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik=1):
    Compress_data = data()
    
    Compress_data.Y = (Frame_class.Y - KeyFrame.Y) // dzielnik
    Compress_data.Cb = (Frame_class.Cb - KeyFrame.Cb) // dzielnik
    Compress_data.Cr = (Frame_class.Cr - KeyFrame.Cr) // dzielnik

    Compress_data.semi_Y = Compress_data.Y
    Compress_data.semi_Cb = Compress_data.Cb
    Compress_data.semi_Cr = Compress_data.Cr

    return Compress_data

def decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, dzielnik=1):
    Y = KeyFrame.Y + (Compress_data.semi_Y * dzielnik)
    Cb = KeyFrame.Cb + (Compress_data.semi_Cb * dzielnik)
    Cr = KeyFrame.Cr + (Compress_data.semi_Cr * dzielnik)
    
    return frame_layers_to_image(Y, Cr, Cb, subsampling)

def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI, filename, info_text):
    # Konwersja z YCrCb do RGB specjalnie do analizy wizualnej
    Ref_RGB = cv2.cvtColor(ReferenceFrame, cv2.COLOR_YCrCb2RGB)
    Decomp_RGB = cv2.cvtColor(DecompressedFrame, cv2.COLOR_YCrCb2RGB)
    
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(15, 5)
    
    # Wycięcie wskazanego obszaru ROI
    ref_roi = Ref_RGB[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    dec_roi = Decomp_RGB[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    
    # Oryginał
    axs[0].imshow(ref_roi)
    axs[0].set_title("Oryginał (RGB)")
    
    # Dekompresja
    axs[2].imshow(dec_roi)
    axs[2].set_title("Po dekompresji (RGB)")
    
    # Różnica
    diff = np.abs(ref_roi.astype(float) - dec_roi.astype(float)).astype(np.uint8)
    axs[1].imshow(diff)
    axs[1].set_title(f"Różnica (Max błąd: {np.max(diff)})")
    
    fig.suptitle(info_text, fontsize=16)
    
    # Zapisujemy do pliku zamiast wyświetlać!
    plt.savefig(filename)
    plt.close(fig)


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

if ile < 0:
    temp_cap = cv2.VideoCapture(os.path.join(kat, plik))
    ile = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()

# Lecimy po wszystkich kombinacjach!
for subsampling in subsamplings_list:
    for dzielnik in dzielniki_list:
        
        print(f"Testuję: subsampling={subsampling}, dzielnik={dzielnik}")
        
        # Otwieramy plik wideo od nowa dla każdej kombinacji
        cap = cv2.VideoCapture(os.path.join(kat, plik))
        compression_information = np.zeros((3, ile))
        
        # Zmieniamy nazwy plików ze znaku ':' na '-' żeby Windows nie krzyczał o błędy
        sub_safe_name = subsampling.replace(":", "-") 
        
        for i in range(ile):
            ret, frame = cap.read()
            if not ret:
                break # Koniec filmu
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            Frame_class = frame_image_to_class(frame, subsampling)
            
            if (i % key_frame_counter) == 0:
                KeyFrame = compress_KeyFrame(Frame_class)
                cY = KeyFrame.Y
                cCb = KeyFrame.Cb
                cCr = KeyFrame.Cr
                Decompresed_Frame = decompress_KeyFrame(KeyFrame, subsampling)
            else:
                # TUTAJ DODAŁEM PRZEKAZANIE DZIELNIKA!
                Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik)
                cY = Compress_data.Y
                cCb = Compress_data.Cb
                cCr = Compress_data.Cr
                Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, dzielnik)
            
            compression_information[0,i] = (frame[:,:,0].size - cY.size) / frame[:,:,0].size
            compression_information[1,i] = (frame[:,:,0].size - cCb.size) / frame[:,:,0].size
            compression_information[2,i] = (frame[:,:,0].size - cCr.size) / frame[:,:,0].size  
            
            if np.any(plot_frames == i):
                for idx, r in enumerate(ROI):
                    nazwa_pliku = os.path.join(wyniki_dir, f"roi{idx}_klatka{i}_sub_{sub_safe_name}_dzielnik_{dzielnik}.png")
                    info = f"Klatka: {i} | Subsampling: {subsampling} | Dzielnik: {dzielnik}"
                    plotDiffrence(frame, Decompresed_Frame, r, nazwa_pliku, info)
        
        # Zamykamy plik po przetworzeniu klatek
        cap.release()

print(f"\nGotowe! Wykresy zapisane w folderze: {wyniki_dir}")