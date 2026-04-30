import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from docx import Document
from docx.shared import Inches
from io import BytesIO



test = [1,1,1,1,1,2,2,2,3,4,5,6,6,6,6,1]


def rle_encoder(image):

    original_shape = np.array(image).shape
    shape_header = np.array([len(original_shape)] + list(original_shape), dtype=int)

    data = np.array(image).flatten()
    
    if len(data) == 0:
        return shape_header

    if len(data) == 0:
            return np.array([], dtype=int)
    
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

def byterun_count_repeats(data, start_idx):
    count = 1
    n = len(data)
    for i in range(start_idx, n - 1):
        if data[i] == data[i+1]:
            count += 1
        else:
            break
    return count

def byterun_count_differences(data, start_idx):
    count = 1
    n = len(data)
    for i in range(start_idx + 1, n):
        if i < n - 1 and data[i] == data[i+1]:
            break
        count += 1
    return count

def byterun_encoder(image):
    original_shape = np.array(image).shape
    shape_info = np.array([len(original_shape)] + list(original_shape), dtype=int)

    data = np.array(image).flatten().astype(int)

    if len(data) == 0:
        return shape_info

    result = np.zeros(data.shape[0] * 2, dtype=int)
    
    write_index = 0
    i = 0
    n = len(data)
    
    while i < n:
        if i < n - 1 and data[i] == data[i+1]:
            total_count = byterun_count_repeats(data, i)
            counter = total_count
            while counter > 128:
                result[write_index] = -127         
                result[write_index + 1] = data[i] 
                write_index += 2
                counter -= 128             

            if counter > 0:
                result[write_index] = -(counter - 1)
                result[write_index + 1] = data[i]

                write_index += 2
            i += total_count
        else:
            total_count = byterun_count_differences(data, i)
            counter = total_count
            processed = 0

            while counter > 128:
                result[write_index] = 127
                write_index += 1
                result[write_index : write_index + 128] = data[i + processed : i + processed + 128]

                write_index += 128
                processed += 128
                counter -= 128
            
            if counter > 0:
                result[write_index] = counter - 1
                write_index += 1
                result[write_index : write_index + counter] = data[i + processed : i + processed + counter]
                write_index += counter
            
            i += total_count

    compressed_data = result[:write_index]
    
    return np.concatenate((shape_info, compressed_data))

def byterun_decoder(encoded_data):
    dims = encoded_data[0]
    shape = tuple(encoded_data[1 : 1 + dims])
    data = encoded_data[1 + dims :]
    
    total_elements = np.prod(shape)
    result = np.zeros(total_elements, dtype=int)
    
    i = 0
    write_index = 0
    n = len(data)
    
    while i < n:
        flag = data[i]
        
        if flag < 0:
            count = -flag + 1
            sym = data[i+1]
            result[write_index : write_index + count] = sym
            write_index += count
            i += 2
        else: 
            count = flag + 1
            result[write_index : write_index + count] = data[i+1 : i+1+count]
            write_index += count
            i += 1 + count
            
    return result.reshape(shape)


def test_compression():
    images = ['rysunek-tech.png', 'cv.png', 'color.png']
    
    for img_name in images:
        if not os.path.exists(img_name):
            print(f"Plik {img_name} nie istnieje, pomijam.")
            continue
            
        print(f"------------- Test zdjęcia {img_name} -------------")
        
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Nie udalo sie wczytac {img_name}")
            continue
        
        img_int = img.astype(int) 
        original_size = img_int.size 

        encoded_rle = rle_encoder(img_int)
        decoded_rle = rle_decoder(encoded_rle)
        
        rle_cr = original_size / encoded_rle.size
        rle_pr = (encoded_rle.size / original_size) * 100
        rle_is_identical = np.array_equal(img_int, decoded_rle)
       
        encoded_br = byterun_encoder(img_int)
        decoded_br = byterun_decoder(encoded_br)
        
        br_cr = original_size / encoded_br.size
        br_pr = (encoded_br.size / original_size) * 100
        br_is_identical = np.array_equal(img_int, decoded_br)


        print(f"Identyczność z oryginałem (RLE): {rle_is_identical}")
        print(f"Identyczność z oryginałem (ByteRun): {br_is_identical}")
        print(f"RLE: stopeń kompresji: {rle_cr:.4f}, czyli {rle_pr:.2f} %")
        print(f"ByteRun: stopeń kompresji: {br_cr:.4f}, czyli {br_pr:.2f} %")

def test_cases():
    test_cases = [
        np.array([1,1,1,1,2,1,1,1,1,2,1,1,1,1]),
        np.array([1,2,3,1,2,3,1,2,3]),
        np.array([5,1,5,1,5,5,1,1,5,5,1,1,5]),
        np.array([-1,-1,-1,-5,-5,-3,-4,-2,1,2,2,1]),
        np.zeros((1,520)),
        np.arange(0,521,1),
        np.eye(7),
        np.dstack([np.eye(7), np.eye(7), np.eye(7)]),
        np.ones((1,1,1,1,1,1,10))
    ]

    for i, data in enumerate(test_cases):
        print(f"\n=== Test {i+1} ===")

        data_int = data.astype(int)

        # RLE
        encoded_rle = rle_encoder(data_int)
        decoded_rle = rle_decoder(encoded_rle)
        rle_ok = np.array_equal(data_int, decoded_rle)

        # ByteRun
        encoded_br = byterun_encoder(data_int)
        decoded_br = byterun_decoder(encoded_br)
        br_ok = np.array_equal(data_int, decoded_br)

        print(f"RLE poprawny: {rle_ok}")
        print(f"ByteRun poprawny: {br_ok}")

        if not rle_ok:
            print("Błąd w RLE!")
        if not br_ok:
            print("Błąd w ByteRun!")

if __name__ == "__main__":
    test_compression()
    test_cases()