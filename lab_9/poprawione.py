import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


kat = r'.'                               # katalog z plikami wideo
plik = "clip_1.mp4"                      # nazwa pliku
ile = 100                                 # ile klatek odtworzyć? 
plot_frames = np.array([])               # Puste, nie potrzebujemy już zdjęć różnic
wyswietlaj_kaltki = False                # Szybciej bez wyświetlania
ROI = [[400, 500, 781, 881]]

subsampling = "4:4:0"                    
dzielnik = 4            
plot_frames=np.array([11, 15, 19])


# PRZEŁĄCZNIK RLE
USE_RLE = False

key_frames_list = [2, 5, 8, 12, 16, 20]

wyniki_jakosc_dir = "wyniki_jakosc_2"
wyniki_pamiec_dir = "wyniki_pamiec_2"
if not os.path.exists(wyniki_jakosc_dir): os.makedirs(wyniki_jakosc_dir)
if not os.path.exists(wyniki_pamiec_dir): os.makedirs(wyniki_pamiec_dir)

##############################################################################
####     Algorytm RLE               ##########################################
##############################################################################

def rle_encoder(image):
    original_shape = np.array(image).shape
    shape_header = np.array([len(original_shape)] + list(original_shape), dtype=int)
    data = np.array(image).flatten()
    
    if len(data) == 0:
        return shape_header
    
    result = np.zeros(data.shape[0] * 2, dtype=int)    
    
    write_index = 0
    current_sym = data[0]
    count = 1
    for i in range(1, len(data)):    
        if data[i] == current_sym:
            count += 1
        else:
            result[write_index] = count
            result[write_index + 1] = current_sym
            write_index += 2
            current_sym = data[i]
            count = 1

    result[write_index] = count
    result[write_index + 1] = current_sym
    write_index += 2
    
    compressed_data = result[:write_index]    
    return np.concatenate((shape_header, compressed_data))

def rle_decoder(encoded_data):
    if len(encoded_data) == 0:
        return np.array([], dtype=int)

    num_dims = encoded_data[0]
    original_shape = tuple(encoded_data[1 : 1 + num_dims])
    rle_data = encoded_data[1 + num_dims :]
    
    if len(rle_data) == 0:
        return np.zeros(0, dtype=int).reshape(original_shape)

    total_length = np.sum(rle_data[0::2])
    result = np.zeros(total_length, dtype=int)
    write_index = 0
    
    for i in range(0, len(rle_data), 2):
        count = rle_data[i]
        sym = rle_data[i+1]
        result[write_index : write_index + count] = sym
        write_index += count
        
    return result.reshape(original_shape)

##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def __init__(self):
        self.Y = None 
        self.Cb = None
        self.Cr = None 
        self.semi_Y = None
        self.semi_Cb = None
        self.semi_Cr = None

def Chroma_subsampling(L, subsampling):
    if subsampling == "4:4:4": return L
    elif subsampling == "4:2:2": return L[:, ::2]
    elif subsampling == "4:4:0": return L[::2, :]
    elif subsampling == "4:2:0": return L[::2, ::2]
    elif subsampling == "4:1:1": return L[:, ::4]
    elif subsampling == "4:1:0": return L[::2, ::4]
    else: return L

def Chroma_resampling(L, subsampling):
    if subsampling == "4:4:4": return L
    elif subsampling == "4:2:2": return np.repeat(L, 2, axis=1)
    elif subsampling == "4:4:0": return np.repeat(L, 2, axis=0)
    elif subsampling == "4:2:0":
        L_res = np.repeat(L, 2, axis=0)
        return np.repeat(L_res, 2, axis=1)
    elif subsampling == "4:1:1": return np.repeat(L, 4, axis=1)
    elif subsampling == "4:1:0":
        L_res = np.repeat(L, 2, axis=0)
        return np.repeat(L_res, 4, axis=1)
    else: return L
        
def frame_image_to_class(frame, subsampling):
    Frame_class = data()
    Frame_class.Y = frame[:,:,0].astype(int)
    Frame_class.Cb = Chroma_subsampling(frame[:,:,2].astype(int), subsampling)
    Frame_class.Cr = Chroma_subsampling(frame[:,:,1].astype(int), subsampling)
    return Frame_class

def frame_layers_to_image(Y, Cr, Cb, subsampling):  
    Cb = Chroma_resampling(Cb, subsampling)
    Cr = Chroma_resampling(Cr, subsampling)
    return np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)

def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    KeyFrame.semi_Y = Frame_class.Y
    KeyFrame.semi_Cb = Frame_class.Cb
    KeyFrame.semi_Cr = Frame_class.Cr
    
    if USE_RLE:
        KeyFrame.Y = rle_encoder(Frame_class.Y)
        KeyFrame.Cb = rle_encoder(Frame_class.Cb)
        KeyFrame.Cr = rle_encoder(Frame_class.Cr)
    else:
        KeyFrame.Y = Frame_class.Y
        KeyFrame.Cb = Frame_class.Cb
        KeyFrame.Cr = Frame_class.Cr
    return KeyFrame

def decompress_KeyFrame(KeyFrame, subsampling):
    Y = KeyFrame.semi_Y
    Cb = KeyFrame.semi_Cb
    Cr = KeyFrame.semi_Cr
    return frame_layers_to_image(Y, Cr, Cb, subsampling)

def compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik=1):
    Compress_data = data()
    
    diff_Y = (Frame_class.Y - KeyFrame.semi_Y) // dzielnik
    diff_Cb = (Frame_class.Cb - KeyFrame.semi_Cb) // dzielnik
    diff_Cr = (Frame_class.Cr - KeyFrame.semi_Cr) // dzielnik

    Compress_data.semi_Y = diff_Y
    Compress_data.semi_Cb = diff_Cb
    Compress_data.semi_Cr = diff_Cr

    if USE_RLE:
        Compress_data.Y = rle_encoder(diff_Y)
        Compress_data.Cb = rle_encoder(diff_Cb)
        Compress_data.Cr = rle_encoder(diff_Cr)
    else:
        Compress_data.Y = diff_Y
        Compress_data.Cb = diff_Cb
        Compress_data.Cr = diff_Cr
    return Compress_data

def decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, dzielnik=1):
    Y = KeyFrame.semi_Y + (Compress_data.semi_Y * dzielnik)
    Cb = KeyFrame.semi_Cb + (Compress_data.semi_Cb * dzielnik)
    Cr = KeyFrame.semi_Cr + (Compress_data.semi_Cr * dzielnik)
    return frame_layers_to_image(Y, Cr, Cb, subsampling)

def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI, filename, info_text):
    # Oczekiwane klatki są w formacie YCrCb
    ref_roi = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    dec_roi = DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    
    # Do pierwszego wiersza (pełny kolor) przeliczamy na RGB
    Ref_RGB = cv2.cvtColor(ref_roi, cv2.COLOR_YCrCb2RGB)
    Decomp_RGB = cv2.cvtColor(dec_roi, cv2.COLOR_YCrCb2RGB)
    
    # Ustawiamy siatkę 4x3
    fig, axs = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    
    # ------------------ RZĄD 0: RGB ------------------
    axs[0, 0].imshow(Ref_RGB)
    axs[0, 0].set_title("Oryginał (RGB)")
    
    diff_RGB = np.abs(Ref_RGB.astype(float) - Decomp_RGB.astype(float)).astype(np.uint8)
    axs[0, 1].imshow(diff_RGB)
    axs[0, 1].set_title(f"Różnica RGB (Max błąd: {np.max(diff_RGB)})")
    
    axs[0, 2].imshow(Decomp_RGB)
    axs[0, 2].set_title("Po dekompresji (RGB)")
    
    # ------------------ RZĘDY 1-3: Y, Cb, Cr ------------------
    # Uwaga: w OpenCV konwersja do YCrCb daje indeksy: 0=Y, 1=Cr, 2=Cb
    channels = [("Y (Luminancja)", 0), ("Cb (Chrominancja Blue)", 2), ("Cr (Chrominancja Red)", 1)]
    
    for i, (name, idx) in enumerate(channels):
        row = i + 1
        ref_chan = ref_roi[:,:,idx]
        dec_chan = dec_roi[:,:,idx]
        diff_chan = np.abs(ref_chan.astype(float) - dec_chan.astype(float)).astype(np.uint8)
        
        axs[row, 0].imshow(ref_chan, cmap='gray')
        axs[row, 0].set_title(f"Oryginał {name}")
        
        axs[row, 1].imshow(diff_chan, cmap='gray')
        axs[row, 1].set_title(f"Różnica {name} (Max: {np.max(diff_chan)})")
        
        axs[row, 2].imshow(dec_chan, cmap='gray')
        axs[row, 2].set_title(f"Dekompresja {name}")
    
    fig.suptitle(info_text, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(filename)
    plt.close(fig)
##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

if ile < 0:
    temp_cap = cv2.VideoCapture(os.path.join(kat, plik))
    ile = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()
plik_base = os.path.splitext(plik)[0]

for key_frame_counter in key_frames_list:        
    print(f"Testuję: Plik = {plik} | Odległość klatek = {key_frame_counter} | RLE = {USE_RLE} | Subsampling = {subsampling} | Dzielnik = {dzielnik}")
    cap = cv2.VideoCapture(os.path.join(kat, plik))
    compression_information = np.zeros((3, ile))
    sub_safe_name = subsampling.replace(":", "-") 
    
    for i in range(ile):
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Frame_class = frame_image_to_class(frame, subsampling)
        
        if (i % key_frame_counter) == 0:
            KeyFrame = compress_KeyFrame(Frame_class)
            cY, cCb, cCr = KeyFrame.Y, KeyFrame.Cb, KeyFrame.Cr
            Decompresed_Frame = decompress_KeyFrame(KeyFrame, subsampling)
        else:
            Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik)
            cY, cCb, cCr = Compress_data.Y, Compress_data.Cb, Compress_data.Cr
            Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling, dzielnik)
        
        compression_information[0,i] = (frame[:,:,0].size - cY.size) / frame[:,:,0].size
        compression_information[1,i] = (frame[:,:,0].size - cCb.size) / frame[:,:,0].size
        compression_information[2,i] = (frame[:,:,0].size - cCr.size) / frame[:,:,0].size  
            
        if np.any(plot_frames == i):
            for idx, r in enumerate(ROI):
                rle_txt = "RLE_ON" if USE_RLE else "RLE_OFF"
                nazwa_obrazka = os.path.join(wyniki_jakosc_dir, f"{plik_base}_roi{idx}_klatka{i}_sub_{sub_safe_name}_dzielnik_{dzielnik}_{rle_txt}.png")
                info = f"Plik: {plik_base} | Klatka: {i} | Subsampling: {subsampling} | Dzielnik: {dzielnik} | {rle_txt}"
                plotDiffrence(frame, Decompresed_Frame, r, nazwa_obrazka, info)
                
    cap.release()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, ile), compression_information[0,:] * 100, label="Y (Luminancja)")
    plt.plot(np.arange(0, ile), compression_information[1,:] * 100, label="Cb (Chrominancja)")
    plt.plot(np.arange(0, ile), compression_information[2,:] * 100, label="Cr (Chrominancja)")
    
    rle_status = "z RLE" if USE_RLE else "bez RLE"
    tytul = f"Plik: {plik} | Sub: {subsampling} | Dzielnik: {dzielnik} | Klatki co: {key_frame_counter} | {rle_status}"
    plt.title(tytul)
    plt.xlabel("Numer klatki")
    plt.ylabel("% zysku pamięci")
    plt.legend()
    plt.grid(True)
    
    nazwa_wykresu_stan = "z_RLE" if USE_RLE else "bez_RLE"
    
    # NAPRAWIONY BŁĄD Z "_4": Teraz nazwa zależy od pliku, ustawień i RLE
    nazwa_wykresu = os.path.join(wyniki_pamiec_dir, f"wykres_{plik_base}_klatki_{key_frame_counter}_sub_{sub_safe_name}_dz_{dzielnik}_{nazwa_wykresu_stan}.png")
    plt.savefig(nazwa_wykresu)
    plt.close()
    
print(f"\nGotowe! Zdjęcia różnic w folderze: {wyniki_jakosc_dir} | Wykresy pamięci w folderze: {wyniki_pamiec_dir}")