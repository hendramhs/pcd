import sys
import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import cv2
import math

class TrashClassificationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrashClassificationApp, self).__init__()
        # Load UI file
        uic.loadUi('trash_detection_app.ui', self)
        
        # Initialize variables
        self.original_image = None
        self.edge_image = None
        self.classified_image = None
        self.current_method = 'Sobel'
        self.threshold_value = 100
        self.trash_objects = []
        
        # Connect signals and slots
        self.loadImageButton.clicked.connect(self.load_image)
        self.detectEdgeButton.clicked.connect(self.detect_edges)
        self.classifyButton.clicked.connect(self.classify_trash)
        self.saveButton.clicked.connect(self.save_result)
        self.methodComboBox.currentTextChanged.connect(self.update_method)
        self.thresholdSlider.valueChanged.connect(self.update_threshold)
        self.enhanceCheckBox.stateChanged.connect(self.detect_edges)
        
        # Set initial state
        self.detectEdgeButton.setEnabled(False)
        self.classifyButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        
        # Status bar initialization
        self.statusbar.showMessage('Siap untuk memuat citra sampah sungai')
        
        # Initialize classification table
        self.setup_classification_table()
        
        # Initialize summary labels
        self.update_summary_display()
    
    def setup_classification_table(self):
        """Setup table for displaying classification results"""
        try:
            # Configure table
            self.classificationTable.setColumnCount(6)
            self.classificationTable.setHorizontalHeaderLabels([
                'ID', 'Jenis Sampah', 'Luas (px²)', 'Tingkat Keyakinan', 'Posisi X', 'Posisi Y'
            ])
            
            # Set column widths
            header = self.classificationTable.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
            header.setSectionResizeMode(1, QHeaderView.Stretch)           # Jenis Sampah
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Luas
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Keyakinan
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Posisi X
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Posisi Y
            
            # Set alternating row colors
            self.classificationTable.setAlternatingRowColors(True)
            
        except AttributeError:
            print("Classification table not found in UI")
    
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Pilih Citra Sampah Sungai", "", 
            "Citra (*.png *.jpg *.jpeg *.bmp);;Semua File (*)", 
            options=options
        )
        
        if file_name:
            try:
                # Load image using OpenCV
                self.original_image = cv2.imread(file_name)
                if self.original_image is None:
                    raise Exception("Gagal memuat citra")
                
                # Convert to RGB for display
                display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Display original image
                self.display_image(display_image, self.originalImageLabel)
                
                # Enable detect button
                self.detectEdgeButton.setEnabled(True)
                
                # Update status
                self.statusbar.showMessage(f'Citra dimuat: {os.path.basename(file_name)}')
                
                # Clear previous results
                self.reset_results()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal memuat citra: {str(e)}")
    
    def reset_results(self):
        """Reset all previous results"""
        self.edgeImageLabel.setText("Hasil deteksi tepi akan ditampilkan di sini")
        self.classifiedImageLabel.setText("Hasil klasifikasi akan ditampilkan di sini")
        self.edge_image = None
        self.classified_image = None
        self.trash_objects = []
        self.classifyButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        
        # Clear table
        try:
            self.classificationTable.setRowCount(0)
        except AttributeError:
            pass
        
        # Update summary
        self.update_summary_display()
    
    def detect_edges(self):
        if self.original_image is None:
            return
        
        try:
            # Convert to grayscale
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Apply contrast enhancement if checked
            if self.enhanceCheckBox.isChecked():
                gray_image = self.enhance_contrast(gray_image)
            
            # Apply edge detection based on selected method
            if self.current_method == 'Sobel':
                self.edge_image = self.sobel_edge_detection(gray_image)
            elif self.current_method == 'Prewitt':
                self.edge_image = self.prewitt_edge_detection(gray_image)
            elif self.current_method == 'Canny':
                self.edge_image = self.canny_edge_detection(gray_image)
            elif self.current_method == 'Laplacian':
                self.edge_image = self.laplacian_edge_detection(gray_image)
            
            # Display edge image
            self.display_image(self.edge_image, self.edgeImageLabel, is_gray=True)
            
            # Enable classify button
            self.classifyButton.setEnabled(True)
            
            # Update status
            self.statusbar.showMessage(f'Deteksi tepi selesai menggunakan metode {self.current_method}')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mendeteksi tepi: {str(e)}")
    
    def classify_trash(self):
        """Classify detected trash objects based on geometric features"""
        if self.edge_image is None:
            return
        
        try:
            # Find contours in the edge image
            contours, _ = cv2.findContours(self.edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to remove noise
            min_contour_area = 150  # Minimum area for valid trash objects
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            # Create visualization image
            self.classified_image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            
            # Clear previous results
            self.trash_objects = []
            
            # Classify each contour
            for i, contour in enumerate(valid_contours):
                classification_result = self.classify_contour(contour, i)
                if classification_result:
                    self.trash_objects.append(classification_result)
                    self.draw_classification_result(contour, classification_result)
            
            # Display classified image
            self.display_image(self.classified_image, self.classifiedImageLabel)
            
            # Update classification table
            self.update_classification_table()
            
            # Update summary
            self.update_summary_display()
            
            # Enable save button
            self.saveButton.setEnabled(True)
            
            # Update status
            self.statusbar.showMessage(f'Klasifikasi selesai: {len(self.trash_objects)} objek sampah terdeteksi')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengklasifikasi sampah: {str(e)}")
    
    def classify_contour(self, contour, obj_id):
        """Classify a single contour based on geometric features"""
        try:
            # Calculate basic geometric features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return None
            
            # Calculate shape features
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Get convex hull and solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Classification logic based on geometric features
            trash_type, confidence = self.determine_trash_type(
                circularity, aspect_ratio, extent, solidity, area, w, h
            )
            
            return {
                'id': obj_id + 1,
                'type': trash_type,
                'confidence': confidence,
                'area': int(area),
                'centroid': (cx, cy),
                'bounding_rect': (x, y, w, h),
                'contour': contour
            }
            
        except Exception as e:
            print(f"Error classifying contour {obj_id}: {str(e)}")
            return None
    
    def determine_trash_type(self, circularity, aspect_ratio, extent, solidity, area, width, height):
        """Determine trash type based on geometric features"""
        confidence_scores = {}
        
        # Botol Plastik - cylindrical shape, moderate aspect ratio
        if 0.3 < circularity < 0.7 and 2.0 < aspect_ratio < 4.0 and extent > 0.6:
            confidence_scores['Botol Plastik'] = min(85, 60 + (circularity * 40))
        
        # Kaleng - more circular, smaller aspect ratio
        if 0.5 < circularity < 0.9 and 1.0 < aspect_ratio < 2.5 and solidity > 0.8:
            confidence_scores['Kaleng'] = min(90, 65 + (circularity * 35))
        
        # Ban Bekas - very circular, large area
        if circularity > 0.7 and 0.8 < aspect_ratio < 1.3 and area > 1000:
            confidence_scores['Ban Bekas'] = min(95, 70 + (circularity * 30))
        
        # Kardus/Kertas - rectangular, high extent
        if 0.1 < circularity < 0.4 and extent > 0.7 and solidity > 0.7:
            if aspect_ratio > 1.2:
                confidence_scores['Kardus/Kertas'] = min(80, 50 + (extent * 40))
        
        # Kantong Plastik - irregular shape, low solidity
        if circularity < 0.5 and solidity < 0.7 and extent < 0.6:
            confidence_scores['Kantong Plastik'] = min(75, 40 + ((1-solidity) * 50))
        
        # Sampah Organik - irregular, medium circularity
        if 0.2 < circularity < 0.6 and solidity < 0.8 and extent < 0.7:
            confidence_scores['Sampah Organik'] = min(70, 35 + (circularity * 45))
        
        # Default classification
        if not confidence_scores:
            confidence_scores['Sampah Tidak Dikenal'] = 30
        
        # Return the classification with highest confidence
        best_type = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_type]
        
        return best_type, int(best_confidence)
    
    def draw_classification_result(self, contour, result):
        """Draw classification result on the image"""
        # Define colors for different trash types
        colors = {
            'Botol Plastik': (0, 255, 0),      # Green
            'Kaleng': (255, 0, 0),             # Red
            'Ban Bekas': (255, 255, 0),        # Yellow
            'Kardus/Kertas': (255, 165, 0),    # Orange
            'Kantong Plastik': (128, 0, 128),  # Purple
            'Sampah Organik': (139, 69, 19),   # Brown
            'Sampah Tidak Dikenal': (128, 128, 128)  # Gray
        }
        
        color = colors.get(result['type'], (128, 128, 128))
        
        # Draw contour
        cv2.drawContours(self.classified_image, [contour], -1, color, 2)
        
        # Draw bounding rectangle
        x, y, w, h = result['bounding_rect']
        cv2.rectangle(self.classified_image, (x, y), (x + w, y + h), color, 1)
        
        # Add label
        label = f"{result['id']}: {result['type']} ({result['confidence']}%)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Position label above the bounding box
        label_y = max(y - 10, label_size[1] + 5)
        cv2.rectangle(self.classified_image, 
                     (x, label_y - label_size[1] - 5), 
                     (x + label_size[0] + 5, label_y + 5), 
                     color, -1)
        
        cv2.putText(self.classified_image, label, (x + 2, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw centroid
        cx, cy = result['centroid']
        cv2.circle(self.classified_image, (cx, cy), 3, color, -1)
    
    def update_classification_table(self):
        """Update the classification results table"""
        try:
            self.classificationTable.setRowCount(len(self.trash_objects))
            
            for i, obj in enumerate(self.trash_objects):
                # ID
                self.classificationTable.setItem(i, 0, QTableWidgetItem(str(obj['id'])))
                
                # Jenis Sampah
                type_item = QTableWidgetItem(obj['type'])
                if obj['confidence'] >= 80:
                    type_item.setBackground(Qt.green)
                elif obj['confidence'] >= 60:
                    type_item.setBackground(Qt.yellow)
                else:
                    type_item.setBackground(Qt.red)
                self.classificationTable.setItem(i, 1, type_item)
                
                # Luas
                self.classificationTable.setItem(i, 2, QTableWidgetItem(str(obj['area'])))
                
                # Tingkat Keyakinan
                confidence_item = QTableWidgetItem(f"{obj['confidence']}%")
                self.classificationTable.setItem(i, 3, confidence_item)
                
                # Posisi X dan Y
                cx, cy = obj['centroid']
                self.classificationTable.setItem(i, 4, QTableWidgetItem(str(cx)))
                self.classificationTable.setItem(i, 5, QTableWidgetItem(str(cy)))
                
        except AttributeError:
            print("Classification table not available")
    
    def update_summary_display(self):
        """Update summary statistics display"""
        try:
            # Count by type
            type_counts = {}
            total_area = 0
            high_confidence_count = 0
            
            for obj in self.trash_objects:
                obj_type = obj['type']
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                total_area += obj['area']
                if obj['confidence'] >= 80:
                    high_confidence_count += 1
            
            # Update labels
            if hasattr(self, 'totalObjectsLabel'):
                self.totalObjectsLabel.setText(str(len(self.trash_objects)))
            
            if hasattr(self, 'totalAreaLabel'):
                self.totalAreaLabel.setText(f"{total_area:,} px²")
            
            if hasattr(self, 'highConfidenceLabel'):
                self.highConfidenceLabel.setText(str(high_confidence_count))
            
            # Update type breakdown
            type_summary = []
            for trash_type, count in type_counts.items():
                type_summary.append(f"{trash_type}: {count}")
            
            if hasattr(self, 'typeBreakdownLabel'):
                self.typeBreakdownLabel.setText('\n'.join(type_summary) if type_summary else 'Tidak ada')
                
        except Exception as e:
            print(f"Error updating summary: {str(e)}")
    
    # Edge detection methods (same as original)
    def enhance_contrast(self, image):
        return cv2.equalizeHist(image)
    
    def sobel_edge_detection(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        _, binary_edge = cv2.threshold(magnitude, self.threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    def prewitt_edge_detection(self, image):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        _, binary_edge = cv2.threshold(magnitude, self.threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    def canny_edge_detection(self, image):
        lower = max(0, int(self.threshold_value * 0.5))
        edges = cv2.Canny(image, lower, self.threshold_value)
        return edges
    
    def laplacian_edge_detection(self, image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        abs_laplacian = np.uint8(255 * np.absolute(laplacian) / np.max(np.absolute(laplacian)))
        _, binary_edge = cv2.threshold(abs_laplacian, self.threshold_value, 255, cv2.THRESH_BINARY)
        return binary_edge
    
    def display_image(self, image, label, is_gray=False):
        height, width = image.shape[:2]
        label_width = label.width()
        label_height = label.height()
        aspect_ratio = width / height
        
        if label_width / label_height > aspect_ratio:
            new_height = label_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = label_width
            new_height = int(new_width / aspect_ratio)
        
        if is_gray:
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
    
    def update_method(self, method):
        self.current_method = method
        if self.original_image is not None:
            self.detect_edges()
    
    def update_threshold(self, value):
        self.threshold_value = value
        try:
            self.thresholdValueLabel.setText(str(value))
        except AttributeError:
            pass
        if self.original_image is not None:
            self.detect_edges()
    
    def save_result(self):
        if self.classified_image is None:
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil Klasifikasi", "", 
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)", 
            options=options
        )
        
        if file_name:
            try:
                if not any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                    file_name += '.png'
                
                # Convert RGB back to BGR for saving
                save_image = cv2.cvtColor(self.classified_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name, save_image)
                
                # Also save classification report
                self.save_classification_report(file_name)
                
                self.statusbar.showMessage(f'Hasil disimpan ke: {os.path.basename(file_name)}')
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan hasil: {str(e)}")
    
    def save_classification_report(self, image_file):
        """Save classification report as text file"""
        try:
            report_file = image_file.rsplit('.', 1)[0] + '_report.txt'
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("LAPORAN KLASIFIKASI SAMPAH SUNGAI\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total Objek Terdeteksi: {len(self.trash_objects)}\n")
                f.write(f"Total Area Sampah: {sum(obj['area'] for obj in self.trash_objects):,} px²\n\n")
                
                # Type breakdown
                type_counts = {}
                for obj in self.trash_objects:
                    obj_type = obj['type']
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                f.write("BREAKDOWN JENIS SAMPAH:\n")
                f.write("-" * 25 + "\n")
                for trash_type, count in sorted(type_counts.items()):
                    percentage = (count / len(self.trash_objects)) * 100 if self.trash_objects else 0
                    f.write(f"{trash_type}: {count} ({percentage:.1f}%)\n")
                
                f.write("\nDETAIL SETIAP OBJEK:\n")
                f.write("-" * 20 + "\n")
                for obj in self.trash_objects:
                    f.write(f"ID: {obj['id']}\n")
                    f.write(f"Jenis: {obj['type']}\n")
                    f.write(f"Keyakinan: {obj['confidence']}%\n")
                    f.write(f"Luas: {obj['area']} px²\n")
                    f.write(f"Posisi: {obj['centroid']}\n")
                    f.write("-" * 20 + "\n")
                    
        except Exception as e:
            print(f"Error saving report: {str(e)}")

# Main application entry point
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TrashClassificationApp()
    window.show()
    sys.exit(app.exec_())