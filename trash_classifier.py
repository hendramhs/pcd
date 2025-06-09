import cv2
import math
import numpy as np

class TrashClassifier:
    @staticmethod
    def classify_contour(contour, obj_id):
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
            trash_type, confidence = TrashClassifier.determine_trash_type(
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
    
    @staticmethod
    def determine_trash_type(circularity, aspect_ratio, extent, solidity, area, width, height):
        """Determine trash type based on geometric features"""
        confidence_scores = {}
        
        # Botol Plastik - cylindrical shape, moderate aspect ratio
        if 0.3 < circularity < 0.7 and 2.0 < aspect_ratio < 4.0 and extent > 0.6:
            confidence_scores['Botol Plastik'] = min(85, 60 + (circularity * 40))
        
        # Kaleng - more circular, smaller aspect ratio
        if 0.5 < circularity < 0.9 and 1.0 < aspect_ratio < 2.5 and solidity > 0.8:
            confidence_scores['Kaleng'] = min(90, 65 + (circularity * 35))
        
        # Kardus/Kertas - rectangular, high extent
        if 0.1 < circularity < 0.4 and extent > 0.7 and solidity > 0.7:
            if aspect_ratio > 1.2:
                confidence_scores['Kardus/Kertas'] = min(80, 50 + (extent * 40))
        
        # Kantong Plastik - irregular shape, low solidity
        if circularity < 0.5 and solidity < 0.7 and extent < 0.6:
            confidence_scores['Kantong Plastik'] = min(75, 40 + ((1-solidity) * 50))
        
        # Default classification
        if not confidence_scores:
            confidence_scores['Sampah Tidak Dikenal'] = 30
        
        # Return the classification with highest confidence
        best_type = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_type]
        
        return best_type, int(best_confidence)
    
    @staticmethod
    def draw_classification_result(image, contour, result):
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
        cv2.drawContours(image, [contour], -1, color, 2)
        
        # Draw bounding rectangle
        x, y, w, h = result['bounding_rect']
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        
        # Add label
        label = f"{result['id']}: {result['type']} ({result['confidence']}%)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Position label above the bounding box
        label_y = max(y - 10, label_size[1] + 5)
        cv2.rectangle(image, 
                     (x, label_y - label_size[1] - 5), 
                     (x + label_size[0] + 5, label_y + 5), 
                     color, -1)
        
        cv2.putText(image, label, (x + 2, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw centroid
        cx, cy = result['centroid']
        cv2.circle(image, (cx, cy), 3, color, -1)
        
        return image