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