from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
import cv2
import os

from image_processor import ImageProcessor
from trash_classifier import TrashClassifier
from utils import Utils

class TrashClassificationUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(TrashClassificationUI, self).__init__()
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
        # Tambahkan koneksi untuk tombol simpan ekstraksi
        self.pushButton_2.clicked.connect(self.save_extraction_steps)
        self.methodComboBox.currentTextChanged.connect(self.update_method)
        self.thresholdSlider.valueChanged.connect(self.update_threshold)
        # Hapus koneksi enhanceCheckBox untuk mencegah pemrosesan otomatis
        # self.enhanceCheckBox.stateChanged.connect(self.detect_edges)
        
        # Set initial state
        self.detectEdgeButton.setEnabled(False)
        self.classifyButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        # Tambahkan state untuk tombol ekstraksi
        self.pushButton_2.setEnabled(False)
        
        # Status bar initialization
        self.statusbar.showMessage('Siap untuk memuat citra sampah sungai')
        
        # Initialize classification table
        self.setup_classification_table()
        
        # Initialize summary labels
        self.update_summary_display()
        
        # Update UI text based on initial method
        self.update_ui_text()
    
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
                Utils.display_image(display_image, self.originalImageLabel)
                
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
        # Update label text based on current method
        if self.current_method == 'Wavelet':
            self.edgeImageLabel.setText("Hasil deteksi tekstur akan ditampilkan di sini")
        else:
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
        """Apply edge detection to the loaded image"""
        if self.original_image is None:
            return
        
        try:
            # Convert to grayscale untuk metode yang memerlukan
            gray_image = ImageProcessor.convert_to_grayscale(self.original_image)
            
            # Apply contrast enhancement if checked
            if self.enhanceCheckBox.isChecked():
                gray_image = ImageProcessor.enhance_contrast(gray_image)
            
            # Apply detection based on selected method dengan preprocessing
            if self.current_method == 'Sobel':
                # Gunakan enhanced Sobel dengan HSV dan adaptive thresholding
                self.edge_image = ImageProcessor.enhanced_sobel_edge_detection(
                    self.original_image, self.threshold_value, use_hsv=True, use_adaptive=True
                )
            elif self.current_method == 'Prewitt':
                # Untuk Prewitt, gunakan preprocessing HSV terlebih dahulu
                if len(self.original_image.shape) == 3:
                    preprocessed_image, _ = ImageProcessor.hsv_preprocessing(self.original_image)
                else:
                    preprocessed_image = gray_image
                self.edge_image = ImageProcessor.prewitt_edge_detection(preprocessed_image, self.threshold_value)
            elif self.current_method == 'Canny':
                # Gunakan enhanced Canny dengan preprocessing
                self.edge_image = ImageProcessor.enhanced_canny_edge_detection(
                    self.original_image, self.threshold_value, use_hsv=True, use_adaptive=True
                )
            elif self.current_method == 'Laplacian':
                # Untuk Laplacian, gunakan preprocessing HSV
                if len(self.original_image.shape) == 3:
                    preprocessed_image, _ = ImageProcessor.hsv_preprocessing(self.original_image)
                else:
                    preprocessed_image = gray_image
                self.edge_image = ImageProcessor.laplacian_edge_detection(preprocessed_image, self.threshold_value)
            elif self.current_method == 'Wavelet':
                # Untuk wavelet, gunakan preprocessing HSV
                if len(self.original_image.shape) == 3:
                    preprocessed_image, _ = ImageProcessor.hsv_preprocessing(self.original_image)
                else:
                    preprocessed_image = gray_image
                self.edge_image = ImageProcessor.wavelet_texture_analysis(preprocessed_image, self.threshold_value)
            
            # Display processed image
            Utils.display_image(self.edge_image, self.edgeImageLabel, is_gray=True)
            
            # Enable classify button dan extraction save button
            self.classifyButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)  # Enable simpan ekstraksi
            
            # Update status
            method_name = 'analisis tekstur wavelet' if self.current_method == 'Wavelet' else f'deteksi tepi {self.current_method}'
            self.statusbar.showMessage(f'Pemrosesan selesai menggunakan metode {method_name}')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memproses citra: {str(e)}")
    
    def classify_trash(self):
        """Classify detected trash objects based on geometric features and texture"""
        if self.edge_image is None:
            return
        
        try:
            # Find contours in the processed image
            contours, _ = cv2.findContours(self.edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to remove noise
            min_contour_area = 150  # Minimum area for valid trash objects
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            # Create visualization image
            self.classified_image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            
            # Clear previous results
            self.trash_objects = []
            
            # Determine if using texture analysis
            use_texture = (self.current_method == 'Wavelet')
            
            # Classify each contour
            for i, contour in enumerate(valid_contours):
                classification_result = TrashClassifier.classify_contour(
                    contour, i, self.original_image if use_texture else None, use_texture
                )
                if classification_result:
                    self.trash_objects.append(classification_result)
                    self.classified_image = TrashClassifier.draw_classification_result(
                        self.classified_image, contour, classification_result
                    )
            
            # Display classified image
            Utils.display_image(self.classified_image, self.classifiedImageLabel)
            
            # Update classification table
            self.update_classification_table()
            
            # Update summary
            self.update_summary_display()
            
            # Enable save button
            self.saveButton.setEnabled(True)
            
            # Update status
            method_info = 'dengan analisis tekstur wavelet' if use_texture else 'berdasarkan fitur geometris'
            self.statusbar.showMessage(f'Klasifikasi selesai {method_info}: {len(self.trash_objects)} objek sampah terdeteksi')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengklasifikasi sampah: {str(e)}")
    
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
    
    def update_method(self, method):
        self.current_method = method
        # Update UI text based on selected method
        self.update_ui_text()
        # Don't automatically process image when changing method
        # User needs to click the button manually
        
    def update_ui_text(self):
        """Update UI text based on current method"""
        if self.current_method == 'Wavelet':
            # Change button text to texture detection
            self.detectEdgeButton.setText("Deteksi Tekstur")
            # Change group box title
            self.edgeGroupBox.setTitle("Citra Deteksi Tekstur")
            # Change settings group title
            self.settingsGroupBox.setTitle("Pengaturan Deteksi Tekstur")
            # Update result label if no image processed yet
            if self.edge_image is None:
                self.edgeImageLabel.setText("Hasil deteksi tekstur akan ditampilkan di sini")
        else:
            # Change button text back to edge detection
            self.detectEdgeButton.setText("Deteksi Tepi")
            # Change group box title back
            self.edgeGroupBox.setTitle("Citra Deteksi Tepi")
            # Change settings group title back
            self.settingsGroupBox.setTitle("Pengaturan Deteksi Tepi")
            # Update result label if no image processed yet
            if self.edge_image is None:
                self.edgeImageLabel.setText("Hasil deteksi tepi akan ditampilkan di sini")
        # Hapus panggilan detect_edges() dari sini
    
    def update_threshold(self, value):
        self.threshold_value = value
        try:
            self.thresholdValueLabel.setText(str(value))
        except AttributeError:
            pass
        # Tetap real-time untuk threshold slider
        if self.original_image is not None:
            self.detect_edges()
    
    def save_extraction_steps(self):
        """Simpan tahapan ekstraksi dalam format gabungan"""
        if self.original_image is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat!")
            return
        
        try:
            # Simpan tahapan ekstraksi
            output_dir, output_file = ImageProcessor.save_extraction_steps(
                self.original_image, 
                self.current_method, 
                self.threshold_value,
                self.enhanceCheckBox.isChecked()
            )
            
            # Update status
            self.statusbar.showMessage(f'Ekstraksi disimpan ke: {os.path.basename(output_dir)}')
            
            # Tampilkan dialog sukses
            QMessageBox.information(
                self, "Sukses", 
                f"Tahapan ekstraksi {self.current_method} berhasil disimpan ke:\n{output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menyimpan ekstraksi: {str(e)}")
    
    def save_result(self):
        """Simpan hasil klasifikasi (existing function)"""
        if self.classified_image is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil klasifikasi!")
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
                report_file = file_name.rsplit('.', 1)[0] + '_report.txt'
                Utils.save_classification_report(report_file, self.trash_objects)
                
                self.statusbar.showMessage(f'Hasil disimpan ke: {os.path.basename(file_name)}')
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan hasil: {str(e)}")