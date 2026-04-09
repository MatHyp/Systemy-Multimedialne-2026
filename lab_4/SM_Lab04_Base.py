import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from docx import Document
from docx.shared import Inches
from io import BytesIO

##########################################
### Settings #############################
##########################################

Test=False
ColorFit_Test=False


GrayScale_bits = [1,2,4]


pallet8 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [1.0, 0.0, 0.0,],
        [1.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0,],
        [1.0, 1.0, 1.0,],
])
pallet16 =  np.array([
        [0.0, 0.0, 0.0,], 
        [0.0, 1.0, 1.0,],
        [0.0, 0.0, 1.0,],
        [1.0, 0.0, 1.0,],
        [0.0, 0.5, 0.0,], 
        [0.5, 0.5, 0.5,],
        [0.0, 1.0, 0.0,],
        [0.5, 0.0, 0.0,],
        [0.0, 0.0, 0.5,],
        [0.5, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [1.0, 0.0, 0.0,],
        [0.75, 0.75, 0.75,],
        [0.0, 0.5, 0.5,],
        [1.0, 1.0, 1.0,], 
        [1.0, 1.0, 0.0,]
])

Color_pallets=[
    pallet8,pallet16
]

M2=np.array([[0,8,2,10],
             [12,4,14,6],
             [3,11,1,9],
             [12,7,13,5]])

OutputRaportFile = "rapo.docx" 

##########################################
### Data Set #############################
##########################################

ImgDir = r'.' 

GsImages = [
    'IMG_GS/GS_0001.tif',
    'IMG_GS/GS_0002.png',
    'IMG_GS/GS_0003.png'
]

ColorImages = [
    'IMG_SMALL/SMALL_0001.tif',
    'IMG_SMALL/SMALL_0002.png',
    'IMG_SMALL/SMALL_0003.png',
    'IMG_SMALL/SMALL_0004.jpg',
    'IMG_SMALL/SMALL_0005.jpg',
    'IMG_SMALL/SMALL_0006.jpg',
    'IMG_SMALL/SMALL_0007.jpg',
    'IMG_SMALL/SMALL_0008.jpg',
    'IMG_SMALL/SMALL_0009.jpg',
    'IMG_SMALL/SMALL_0010.jpg'
]

##########################################
### Functions to  ########################
##########################################

import numpy as np

def imgToUint8(img):
    if np.issubdtype(img.dtype, np.integer):
        return img
    elif np.issubdtype(img.dtype, np.floating):
        return (img * 255).astype('uint8')
    else:
        return img

def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    elif np.issubdtype(img.dtype, np.integer):
        return img / 255.0
    else:
        return img

def colorFit(pixel,Pallet):
        #######

        pixelSubPallet = Pallet-pixel 
        distances = np.linalg.norm(pixelSubPallet,axis=1)
        minDistance = np.argmin(distances)

        return Pallet[minDistance]

def kwant_colorFit(img,Pallet):
        out_img = img.copy()
        for k in range(img.shape[1]):
                for w in range(img.shape[0]):
                        tmp = colorFit(img[w,k],Pallet)
                        if len(tmp)==1:
                            out_img[w,k]=tmp[0]
                        else:
                            out_img[w,k]=tmp[:]
        return out_img

def dith_randm(img):
        random_matrx = np.random.rand(img.shape[0], img.shape[1])
        out_img = (img >= random_matrx) * 1

        return out_img.astype(img.dtype)

def dith_ordered(img, Pallet, r=1, M=M2):
        out_img = img.copy()
        
        n = M.shape[0] // 2
        
        Mpre = (M + 1) / (2 * n)**2 - 0.5
        
        for k in range(img.shape[1]):
                for w in range(img.shape[0]):
                        C = img[w, k]
                        
                        m_val = Mpre[w % (2 * n), k % (2 * n)]
                        
                        temp_val = C + r * m_val
                        
                        tmp = colorFit(temp_val, Pallet)
                        
                        if len(tmp) == 1:
                                out_img[w, k] = tmp[0]
                        else:
                                out_img[w, k] = tmp[:]
                                
        return out_img.astype(img.dtype)

def dith_FS(img, Pallet):
        out_img = img.copy()
        
        height = out_img.shape[0]
        width = out_img.shape[1]
        
        for w in range(height):
                for k in range(width):
                        
                        oldpixel = out_img[w, k].copy()
                        
                        tmp = colorFit(oldpixel, Pallet)
                        
                        if len(tmp) == 1:
                                newpixel = tmp[0]
                        else:
                                newpixel = tmp[:]
                                
                        out_img[w, k] = newpixel
                        
                        quant_error = oldpixel - newpixel
                        
                        if k + 1 < width:
                                out_img[w, k + 1] = out_img[w, k + 1] + quant_error * 7 / 16

                        if k - 1 >= 0 and w + 1 < height:
                                out_img[w + 1, k - 1] = out_img[w + 1, k - 1] + quant_error * 3 / 16
                        if w + 1 < height:
                                out_img[w + 1, k] = out_img[w + 1, k] + quant_error * 5 / 16
                                
                        if k + 1 < width and w + 1 < height:
                                out_img[w + 1, k + 1] = out_img[w + 1, k + 1] + quant_error * 1 / 16
                                
        return out_img.astype(img.dtype)


##########################################
### Main Program  ########################
##########################################

def process_and_plot_GS(img,bit,filename,counter,figsize=(5,5)):
        if len(img.shape)>2:
                img=img[:,:,0]
        palett=np.linspace(0,1,2**bit).reshape(-1,1)
        qwant_img=kwant_colorFit(img,palett)

        order_img=dith_ordered(img,palett)
        FS_img=dith_FS(img,palett)
        if bit==1:
            rand_img=dith_randm(img)
            f,axs=plt.subplots(2,3,num=counter,figsize=figsize) 
            f.suptitle(f"{filename} Dithering 1-bit")
            axs[0,0].imshow(img,cmap="gray")
            axs[0,0].set_title("Oryginał")
            axs[0,0].set_axis_off()

            axs[1,0].remove()

            axs[0,2].imshow(rand_img,cmap="gray")
            axs[0,2].set_title("Dithering\n Losowy")
            axs[0,2].set_axis_off()

            axt=[axs[0,1],axs[1,1],axs[1,2]]
        else:
            f,axs=plt.subplots(1,4,num=counter,figsize=figsize) 
            f.suptitle(f"{filename} Dithering {bit}-bitów")
            axs[0].imshow(img,cmap="gray")
            axs[0].set_title("Oryginał")
            axs[0].set_axis_off()  
            axt=[axs[1],axs[2],axs[3]] 
            rand_img=0

        axt[0].imshow(qwant_img,cmap="gray")
        axt[0].set_title("Kwantyzacja")
        axt[0].set_axis_off()

        
        axt[1].imshow(order_img,cmap="gray")
        axt[1].set_title("Dithering\n Zorganizowany")
        axt[1].set_axis_off()

        
        axt[2].imshow(FS_img,cmap="gray")
        axt[2].set_title("Dithering\n Floyda-Steinberga")
        axt[2].set_axis_off()

        if Test:
                if bit==1:
                    print(f"Test of uniqe values counts:\n"+
                        f"Unique values in Pallet: {np.unique(palett).size}\n"+
                        f"Dithering Random: {np.unique(rand_img).size}\n"+
                        f"Dithering Ordered: {np.unique(order_img).size}\n"+
                        f"Dithering Floyd-Steinberg: {np.unique(FS_img).size}\n")
                else:
                    print(f"Test of uniqe values counts:\n"+
                        f"Unique values in Pallet: {np.unique(palett).size}\n"+
                        f"Dithering Ordered: {np.unique(order_img).size}\n"+
                        f"Dithering Floyd-Steinberg: {np.unique(FS_img).size}\n")
        return f
                    

def process_and_plot_Color(img,palett,filename,counter,figsize=(5,5)):
        if img.shape[2]>3:
                img=img[:,:,:3]
        qwant_img=kwant_colorFit(img,palett)

        order_img=dith_ordered(img,palett)
        FS_img=dith_FS(img,palett)

        f,axs=plt.subplots(1,4,num=counter,figsize=figsize) 
        f.suptitle(f"{filename} Dithering {len(palett)} kolorów")
        axs[0].imshow(img)
        axs[0].set_title("Oryginał")
        axs[0].set_axis_off()  

        axs[1].imshow(qwant_img)
        axs[1].set_title("Kwantyzacja")
        axs[1].set_axis_off()  

        axs[2].imshow(order_img)
        axs[2].set_title("Dithering\n Zorganizowany")
        axs[2].set_axis_off()  

        axs[3].imshow(FS_img)
        axs[3].set_title("Dithering\n Floyda-Steinberga")
        axs[3].set_axis_off() 

        return f




if Test:
        if ColorFit_Test:
            paleta = np.linspace(0,1,3).reshape(3,1)
            print(f"0.43 -> {colorFit(0.43,paleta)}") 
            print(f"0.66 -> {colorFit(0.66,paleta)}") 
            print(f"0.8 -> {colorFit(0.8,paleta)}") 

            print(f"[0.25,0.25,0.5] 8 kolorów -> {colorFit(np.array([0.25,0.25,0.5]),pallet8)}")
            print(f"[0.25,0.25,0.5] 16 kolorów-> {colorFit(np.array([0.25,0.25,0.5]),pallet16)}")

              
        file=GsImages[0]
        img=imgToFloat(plt.imread(os.path.join(ImgDir,file)))
        counter=1
        for bit in GrayScale_bits:
               process_and_plot_GS(img=img,bit=bit,filename=file,counter=counter)
               counter+=1

        file=ColorImages[0]
        img=imgToFloat(plt.imread(os.path.join(ImgDir,file)))
        for palett in Color_pallets:
               process_and_plot_Color(img=img,palett=palett,filename=file,counter=counter)
               counter+=1
        
        plt.show()
else:
    # generate raport
    document = Document()
    document.add_heading('Report',0) # tworzenie nagłówków druga wartość to poziom nagłówka 
    document.add_paragraph("Autor: ")
    document.add_paragraph("Mateusz Hypś")
    document.add_section()
    document.add_heading("Test ditheringu na obrazach w skali odcieni szarości",1)
    counter = 1 
    for file in GsImages:
        img=imgToFloat(plt.imread(os.path.join(ImgDir,file)))
        for bit in GrayScale_bits:
                f = process_and_plot_GS(img=img,bit=bit,filename=file,counter=counter)
                memfile = BytesIO() 
                f.savefig(memfile)
                document.add_picture(memfile, width=Inches(6)) # set document size
                memfile.close()
                f.clf()
    document.add_section()
    document.add_heading("Test ditheringu na obrazach kolorowych",1)
    for file in ColorImages:
        img=imgToFloat(plt.imread(os.path.join(ImgDir,file)))
        for palett in Color_pallets:
            f = process_and_plot_Color(img=img,palett=palett,filename=file,counter=counter)
            memfile = BytesIO() 
            f.savefig(memfile)
            document.add_picture(memfile, width=Inches(6)) # set document size
            memfile.close()
            f.clf()
    document.add_section()
    document.add_heading("Podsumowanie i wnioski",1)
    document.add_paragraph("Dithering Floyda-Steinberga zapewnia najlepszą jakość — obrazy są najbardziej zbliżone do oryginału i w większości przypadków zawierają najmniej rozmycia oraz utraty jakości. Dithering zorganizowany daje w niektórych przypadkach podobny rezultat co Floyda-Steinberga, np. dla „IMG_SMALL/SMALL_0003.png Dithering 16 kolorów” i w ditheringach bitowych, ale jednak w większości innych przypadków powoduje obniżenie jakości i dużą widoczność przejść między pikselami. Jeżeli chodzi o dithering losowy, pozostawia on dużo do życzenia — przejścia między pikselami są słabo wygładzone, widać dużo szumu. Proces kwantyzacji dobrze redukuje liczbę kolorów w większości przypadków. ")
    document.save(OutputRaportFile) 