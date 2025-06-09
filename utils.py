import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class Utils:
    @staticmethod
    def display_image(image, label, is_gray=False):
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
    
    @staticmethod
    def save_classification_report(report_file, trash_objects):
        """Save classification report as text file"""
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("LAPORAN KLASIFIKASI SAMPAH SUNGAI\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total Objek Terdeteksi: {len(trash_objects)}\n")
                f.write(f"Total Area Sampah: {sum(obj['area'] for obj in trash_objects):,} px²\n\n")
                
                # Type breakdown
                type_counts = {}
                for obj in trash_objects:
                    obj_type = obj['type']
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                
                f.write("BREAKDOWN JENIS SAMPAH:\n")
                f.write("-" * 25 + "\n")
                for trash_type, count in sorted(type_counts.items()):
                    percentage = (count / len(trash_objects)) * 100 if trash_objects else 0
                    f.write(f"{trash_type}: {count} ({percentage:.1f}%)\n")
                
                f.write("\nDETAIL SETIAP OBJEK:\n")
                f.write("-" * 20 + "\n")
                for obj in trash_objects:
                    f.write(f"ID: {obj['id']}\n")
                    f.write(f"Jenis: {obj['type']}\n")
                    f.write(f"Keyakinan: {obj['confidence']}%\n")
                    f.write(f"Luas: {obj['area']} px²\n")
                    f.write(f"Posisi: {obj['centroid']}\n")
                    f.write("-" * 20 + "\n")
                    
        except Exception as e:
            print(f"Error saving report: {str(e)}")