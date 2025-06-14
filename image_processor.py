# Tambahkan import cv2 di bagian atas file jika belum ada
import cv2
import numpy as np

# Tambahkan import yang diperlukan di bagian atas
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
            image = ImageProcessor.convert_to_grayscale(image)
            
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
    
    # @staticmethod
    # def extract_color_moments(image):
    #     """
    #     Ekstraksi fitur color moments (mean, standard deviation, skewness) untuk setiap channel warna.
    #     Input:
    #         - image: Citra input (BGR)
    #     Output:
    #         - Vektor fitur numpy (1D) dengan 9 nilai (3 moments x 3 channels)
    #     """
    #     # Pastikan citra dalam format BGR
    #     if len(image.shape) != 3:
    #         raise ValueError("Citra harus dalam format BGR (3 channel)")
            
    #     # Inisialisasi array untuk menyimpan fitur
    #     features = []
        
    #     # Proses setiap channel (B, G, R)
    #     for channel in range(3):
    #         # Ambil channel
    #         channel_data = image[:, :, channel].astype(np.float32)
            
    #         # 1. Mean (Moment pertama)
    #         mean = np.mean(channel_data)
            
    #         # 2. Standard Deviation (Moment kedua)
    #         # Hitung variance manual
    #         variance = np.mean((channel_data - mean) ** 2)
    #         std_dev = np.sqrt(variance)
            
    #         # 3. Skewness (Moment ketiga)
    #         # Hitung skewness manual
    #         skewness = np.mean(((channel_data - mean) / (std_dev + 1e-6)) ** 3)
            
    #         # Tambahkan ke list fitur
    #         features.extend([mean, std_dev, skewness])
            
    #     return np.array(features)
    
    @staticmethod
    def wavelet_texture_analysis(image, threshold_value):
        """
        Analisis tekstur menggunakan transformasi wavelet Haar.
        Menggunakan energy dari detail coefficients untuk deteksi objek berdasarkan tekstur.
        """
        # Konversi ke grayscale jika belum
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize ke 256x256
        original_size = image.shape
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
        cA = result[0:rows // 2, 0:cols // 2]  # Approximation
        cH = result[0:rows // 2, cols // 2:]   # Horizontal detail
        cV = result[rows // 2:, 0:cols // 2]   # Vertical detail
        cD = result[rows // 2:, cols // 2:]    # Diagonal detail
        # cA = result[0:rows // 2, 0:cols // 2]  # Approximation (kiri atas)
        # cH = result[0:rows // 2, cols // 2:]   # Horizontal detail (kanan atas)
        # cV = result[rows // 2:, 0:cols // 2]   # Vertical detail (kiri bawah)
        # cD = result[rows // 2:, cols // 2:]    # Diagonal detail (kanan bawah)
        
        # # Normalisasi setiap subband ke range 0-255
        # def normalize_subband(subband):
        #     subband_min = np.min(subband)
        #     subband_max = np.max(subband)
        #     if subband_max - subband_min > 0:
        #         normalized = 255 * (subband - subband_min) / (subband_max - subband_min)
        #     else:
        #         normalized = np.zeros_like(subband)
        #     return np.uint8(normalized)
        
        # cA_norm = normalize_subband(cA)
        # cH_norm = normalize_subband(cH)
        # cV_norm = normalize_subband(cV)
        # cD_norm = normalize_subband(cD)
        # Hitung energy dari detail coefficients untuk analisis tekstur
        texture_energy = np.sqrt(cH**2 + cV**2 + cD**2)
        
        # Normalisasi ke range 0-255
        texture_energy = np.uint8(255 * texture_energy / np.max(texture_energy))
        
        # Thresholding untuk segmentasi berdasarkan tekstur
        _, texture_mask = cv2.threshold(texture_energy, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Gabungkan keempat subband menjadi satu gambar
        # Layout: cA (kiri atas), cH (kanan atas), cV (kiri bawah), cD (kanan bawah)
        # wavelet_output = np.zeros((rows, cols), dtype=np.uint8)
        # wavelet_output[0:rows // 2, 0:cols // 2] = cA_norm
        # wavelet_output[0:rows // 2, cols // 2:] = cH_norm
        # wavelet_output[rows // 2:, 0:cols // 2] = cV_norm
        # wavelet_output[rows // 2:, cols // 2:] = cD_norm

        # Resize kembali ke ukuran asli
        texture_mask = cv2.resize(texture_mask, (original_size[1], original_size[0]))
        # wavelet_output = cv2.resize(wavelet_output, (original_size[1], original_size[0]))
        
        return texture_mask
        # return wavelet_output
    
    @staticmethod
    def hsv_preprocessing(image):
        """
        Preprocessing menggunakan HSV color space untuk mengurangi noise dari tembok
        dan meningkatkan deteksi objek sampah
        """
        # Konversi ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definisikan range untuk objek sampah (non-tembok)
        # Range untuk plastik, kaleng, dan sampah organik
        lower_range1 = np.array([0, 30, 30])    # Merah, oranye, kuning
        upper_range1 = np.array([30, 255, 255])
        
        lower_range2 = np.array([35, 40, 40])   # Hijau (sampah organik)
        upper_range2 = np.array([85, 255, 255])
        
        lower_range3 = np.array([100, 50, 50])  # Biru (plastik)
        upper_range3 = np.array([130, 255, 255])
        
        # Buat mask untuk setiap range
        mask1 = cv2.inRange(hsv, lower_range1, upper_range1)
        mask2 = cv2.inRange(hsv, lower_range2, upper_range2)
        mask3 = cv2.inRange(hsv, lower_range3, upper_range3)
        
        # Gabungkan semua mask
        combined_mask = cv2.bitwise_or(mask1, mask2)
        combined_mask = cv2.bitwise_or(combined_mask, mask3)
        
        # Tambahkan mask untuk objek dengan saturasi tinggi (bukan tembok abu-abu)
        high_saturation_mask = cv2.inRange(hsv[:,:,1], 40, 255)
        combined_mask = cv2.bitwise_or(combined_mask, high_saturation_mask)
        
        # Morphological operations untuk membersihkan noise
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Terapkan mask ke citra asli
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        
        # Konversi ke grayscale
        gray_result = ImageProcessor.convert_to_grayscale(result)
        
        return gray_result, combined_mask
    
    @staticmethod
    def adaptive_threshold_preprocessing(image):
        """
        Preprocessing menggunakan adaptive thresholding untuk segmentasi objek
        """
        # Konversi ke grayscale jika belum
        if len(image.shape) == 3:
            gray = ImageProcessor.convert_to_grayscale(image)
        else:
            gray = image.copy()
        
        # Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold dengan metode Gaussian
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert untuk mendapatkan objek sebagai foreground
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
        
        # Morphological operations untuk membersihkan
        kernel = np.ones((3,3), np.uint8)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        
        return adaptive_thresh
    
    @staticmethod
    def enhanced_sobel_edge_detection(image, threshold_value, use_hsv=True, use_adaptive=True):
        """
        Enhanced Sobel edge detection dengan preprocessing HSV dan adaptive thresholding
        """
        processed_image = image.copy()
        
        # Step 1: HSV preprocessing jika diminta
        if use_hsv and len(image.shape) == 3:
            processed_image, hsv_mask = ImageProcessor.hsv_preprocessing(processed_image)
        elif len(image.shape) == 3:
            processed_image = ImageProcessor.convert_to_grayscale(processed_image)
        
        # Step 2: Adaptive thresholding preprocessing jika diminta
        if use_adaptive:
            adaptive_mask = ImageProcessor.adaptive_threshold_preprocessing(processed_image)
            # Gabungkan dengan hasil HSV jika ada
            if use_hsv and len(image.shape) == 3:
                processed_image = cv2.bitwise_and(processed_image, processed_image, mask=adaptive_mask)
            else:
                processed_image = cv2.bitwise_and(processed_image, adaptive_mask)
        
        # Step 3: Sobel edge detection
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        grad_x = cv2.filter2D(processed_image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(processed_image, cv2.CV_64F, kernel_y)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        _, binary_edge = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY)
        
        return binary_edge
    
    @staticmethod
    def enhanced_canny_edge_detection(image, threshold_value, use_hsv=True, use_adaptive=True):
        """
        Enhanced Canny edge detection dengan preprocessing HSV dan adaptive thresholding
        """
        processed_image = image.copy()
        
        # Step 1: HSV preprocessing jika diminta
        if use_hsv and len(image.shape) == 3:
            processed_image, hsv_mask = ImageProcessor.hsv_preprocessing(processed_image)
        elif len(image.shape) == 3:
            processed_image = ImageProcessor.convert_to_grayscale(processed_image)
        
        # Step 2: Adaptive thresholding preprocessing jika diminta
        if use_adaptive:
            adaptive_mask = ImageProcessor.adaptive_threshold_preprocessing(processed_image)
            processed_image = cv2.bitwise_and(processed_image, processed_image, mask=adaptive_mask)
        
        # Step 3: Canny edge detection
        lower = max(0, int(threshold_value * 0.5))
        edges = cv2.Canny(processed_image, lower, threshold_value)
        
        return edges
    
    @staticmethod
    def save_extraction_steps(original_image, method, threshold_value, enhance_contrast=False):
        """
        Simpan semua tahapan ekstraksi dalam satu gambar gabungan seperti contoh Canny
        """
        # Buat timestamp untuk folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"ekstraksi_{method.lower()}_{timestamp}"
        output_dir = os.path.join("citraHasil", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Tahap 1: Original
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Tahap 2: Grayscale
        grayscale = ImageProcessor.convert_to_grayscale(original_image)
        
        # Tahap 3: Enhanced (jika diperlukan)
        if enhance_contrast:
            enhanced = ImageProcessor.enhance_contrast(grayscale)
        else:
            enhanced = grayscale.copy()
        
        # Tahap 4: HSV Preprocessing (jika menggunakan enhanced methods)
        hsv_processed = None
        if method in ['Sobel', 'Canny']:
            hsv_processed, hsv_mask = ImageProcessor.hsv_preprocessing(original_image)
        
        # Tahap 5: Adaptive Threshold (jika menggunakan enhanced methods)
        adaptive_thresh = None
        if method in ['Sobel', 'Canny']:
            adaptive_thresh = ImageProcessor.adaptive_threshold_preprocessing(enhanced)
        
        # Tahap 6: Edge Detection
        if method == 'Sobel':
            if hsv_processed is not None:
                edge_result = ImageProcessor.enhanced_sobel_edge_detection(
                    original_image, threshold_value, use_hsv=True, use_adaptive=True
                )
            else:
                edge_result = ImageProcessor.sobel_edge_detection(enhanced, threshold_value)
        elif method == 'Canny':
            if hsv_processed is not None:
                edge_result = ImageProcessor.enhanced_canny_edge_detection(
                    original_image, threshold_value, use_hsv=True, use_adaptive=True
                )
            else:
                edge_result = ImageProcessor.canny_edge_detection(enhanced, threshold_value)
        elif method == 'Prewitt':
            if len(original_image.shape) == 3:
                preprocessed, _ = ImageProcessor.hsv_preprocessing(original_image)
                edge_result = ImageProcessor.prewitt_edge_detection(preprocessed, threshold_value)
            else:
                edge_result = ImageProcessor.prewitt_edge_detection(enhanced, threshold_value)
        elif method == 'Laplacian':
            if len(original_image.shape) == 3:
                preprocessed, _ = ImageProcessor.hsv_preprocessing(original_image)
                edge_result = ImageProcessor.laplacian_edge_detection(preprocessed, threshold_value)
            else:
                edge_result = ImageProcessor.laplacian_edge_detection(enhanced, threshold_value)
        elif method == 'Wavelet':
            if len(original_image.shape) == 3:
                preprocessed, _ = ImageProcessor.hsv_preprocessing(original_image)
                edge_result = ImageProcessor.wavelet_texture_analysis(preprocessed, threshold_value)
            else:
                edge_result = ImageProcessor.wavelet_texture_analysis(enhanced, threshold_value)
        
        # Tahap 7: Contour Detection
        contours, _ = cv2.findContours(edge_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Buat layout gambar gabungan
        if method in ['Sobel', 'Canny'] and hsv_processed is not None:
            # Layout 3x3 untuk enhanced methods
            fig = plt.figure(figsize=(15, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig)
            
            # Baris 1
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_rgb)
            ax1.set_title('Original', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(grayscale, cmap='gray')
            ax2.set_title('Grayscale', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(enhanced, cmap='gray')
            title3 = 'Enhanced' if enhance_contrast else 'Grayscale'
            ax3.set_title(title3, fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # Baris 2
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(hsv_processed, cmap='gray')
            ax4.set_title('HSV Preprocessing', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(adaptive_thresh, cmap='gray')
            ax5.set_title('Adaptive Threshold', fontsize=12, fontweight='bold')
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(edge_result, cmap='gray')
            ax6.set_title(f'{method} Edge Detection', fontsize=12, fontweight='bold')
            ax6.axis('off')
            
            # Baris 3 - Contour (span 3 kolom)
            ax7 = fig.add_subplot(gs[2, :])
            ax7.imshow(contour_image)
            ax7.set_title('Contour Detection', fontsize=12, fontweight='bold')
            ax7.axis('off')
            
        else:
            # Layout 2x3 untuk methods biasa
            fig = plt.figure(figsize=(15, 8))
            gs = gridspec.GridSpec(2, 3, figure=fig)
            
            # Baris 1
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_rgb)
            ax1.set_title('Original', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(grayscale, cmap='gray')
            ax2.set_title('Grayscale', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            if method in ['Prewitt', 'Laplacian', 'Wavelet'] and len(original_image.shape) == 3:
                preprocessed, _ = ImageProcessor.hsv_preprocessing(original_image)
                ax3.imshow(preprocessed, cmap='gray')
                ax3.set_title('HSV Preprocessing', fontsize=12, fontweight='bold')
            else:
                ax3.imshow(enhanced, cmap='gray')
                title3 = 'Enhanced' if enhance_contrast else 'Processed'
                ax3.set_title(title3, fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # Baris 2
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(edge_result, cmap='gray')
            ax4.set_title(f'{method} Edge Detection', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1:])  # Span 2 kolom
            ax5.imshow(contour_image)
            ax5.set_title('Contour Detection', fontsize=12, fontweight='bold')
            ax5.axis('off')
        
        plt.tight_layout()
        
        # Simpan gambar gabungan
        output_file = os.path.join(output_dir, f"{method.lower()}_extraction_steps.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Simpan info file
        info_file = os.path.join(output_dir, "info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Metode: {method}\n")
            f.write(f"Threshold: {threshold_value}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Enhance Contrast: {'Ya' if enhance_contrast else 'Tidak'}\n")
            f.write(f"\nFile yang disimpan:\n")
            f.write(f"- extraction_steps: {method.lower()}_extraction_steps.png\n")
        
        return output_dir, output_file