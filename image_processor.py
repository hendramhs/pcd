import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def convert_to_grayscale(image):
        # Konversi ke grayscale menggunakan rumus: 0.299*R + 0.587*G + 0.114*B
        # Ekstrak channel BGR (OpenCV menggunakan format BGR)
        b, g, r = cv2.split(image)
        
        # Aplikasikan rumus grayscale
        gray = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)
        
        # Konversi kembali ke uint8
        return np.uint8(gray)
    
    @staticmethod
    def enhance_contrast(image):
        return cv2.equalizeHist(image)
    
    @staticmethod
    def sobel_edge_detection(image, threshold_value):
        # Definisi kernel Sobel secara manual
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Aplikasikan filter menggunakan filter2D seperti pada Prewitt
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        
        # Hitung magnitude gradien
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        # Thresholding untuk mendapatkan tepi biner
        _, binary_edge = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    @staticmethod
    def prewitt_edge_detection(image, threshold_value):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        _, binary_edge = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    @staticmethod
    def canny_edge_detection(image, threshold_value):
        lower = max(0, int(threshold_value * 0.5))
        edges = cv2.Canny(image, lower, threshold_value)
        return edges
    
    @staticmethod
    def laplacian_edge_detection(image, threshold_value):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        abs_laplacian = np.uint8(255 * np.absolute(laplacian) / np.max(np.absolute(laplacian)))
        _, binary_edge = cv2.threshold(abs_laplacian, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    @staticmethod
    def extract_wavelet_features(image):
        """
        Ekstraksi fitur wavelet Haar dari citra.
        Input:
            - image: Citra input (BGR atau grayscale)
        Output:
            - Vektor fitur numpy (1D), berupa mean dari setiap subband wavelet
        """
        # Konversi ke grayscale jika belum
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize ke 256x256
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
        
        # Ekstraksi fitur (mean dari setiap subband)
        features = [
            np.mean(np.abs(cA)),
            np.mean(np.abs(cH)),
            np.mean(np.abs(cV)),
            np.mean(np.abs(cD))
        ]
        
        return np.array(features)
    
    @staticmethod
    def extract_color_moments(image):
        """
        Ekstraksi fitur color moments (mean, standard deviation, skewness) untuk setiap channel warna.
        Input:
            - image: Citra input (BGR)
        Output:
            - Vektor fitur numpy (1D) dengan 9 nilai (3 moments x 3 channels)
        """
        # Pastikan citra dalam format BGR
        if len(image.shape) != 3:
            raise ValueError("Citra harus dalam format BGR (3 channel)")
            
        # Inisialisasi array untuk menyimpan fitur
        features = []
        
        # Proses setiap channel (B, G, R)
        for channel in range(3):
            # Ambil channel
            channel_data = image[:, :, channel].astype(np.float32)
            
            # 1. Mean (Moment pertama)
            mean = np.mean(channel_data)
            
            # 2. Standard Deviation (Moment kedua)
            # Hitung variance manual
            variance = np.mean((channel_data - mean) ** 2)
            std_dev = np.sqrt(variance)
            
            # 3. Skewness (Moment ketiga)
            # Hitung skewness manual
            skewness = np.mean(((channel_data - mean) / (std_dev + 1e-6)) ** 3)
            
            # Tambahkan ke list fitur
            features.extend([mean, std_dev, skewness])
            
        return np.array(features)