import cv2
import numpy as np

def manual_haar_dwt(image):
    """
    Melakukan 1-level Haar Discrete Wavelet Transform secara manual untuk citra grayscale.
    Input:
        - image: np.ndarray grayscale (2D)
    Output:
        - cA, cH, cV, cD: Approximation, Horizontal, Vertical, Diagonal detail coefficients
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (256, 256)).astype(np.float32)

    rows, cols = image.shape
    temp = np.zeros_like(image)

    # Transformasi horizontal (baris)
    for i in range(rows):
        for j in range(0, cols, 2):
            avg = (image[i, j] + image[i, j+1]) / 2
            diff = (image[i, j] - image[i, j+1]) / 2
            temp[i, j // 2] = avg
            temp[i, (j // 2) + cols // 2] = diff

    result = np.zeros_like(temp)

    # Transformasi vertikal (kolom)
    for j in range(cols):
        for i in range(0, rows, 2):
            avg = (temp[i, j] + temp[i+1, j]) / 2
            diff = (temp[i, j] - temp[i+1, j]) / 2
            result[i // 2, j] = avg
            result[(i // 2) + rows // 2, j] = diff

    # Subband hasil
    cA = result[0:rows // 2, 0:cols // 2]
    cH = result[0:rows // 2, cols // 2:]
    cV = result[rows // 2:, 0:cols // 2]
    cD = result[rows // 2:, cols // 2:]

    return cA, cH, cV, cD


def extract_manual_wavelet_features(image):
    """
    Ekstraksi fitur wavelet Haar manual dari citra grayscale.
    Return:
        - Vektor fitur numpy (1D), berupa mean dari setiap subband.
    """
    cA, cH, cV, cD = manual_haar_dwt(image)
    features = [
        np.mean(np.abs(cA)),
        np.mean(np.abs(cH)),
        np.mean(np.abs(cV)),
        np.mean(np.abs(cD))
    ]
    return np.array(features)
