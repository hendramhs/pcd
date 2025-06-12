import cv2
import math
import numpy as np

class TrashClassifier:
    @staticmethod
    def classify_contour(contour, obj_id, original_image=None, use_texture=False):
        """Classify a single contour based on geometric features and optionally texture features"""
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
            
            # Ekstraksi fitur tekstur wavelet jika diminta
            texture_features = None
            if use_texture and original_image is not None:
                # Crop region of interest
                roi = original_image[y:y+h, x:x+w]
                if roi.size > 0:
                    from image_processor import ImageProcessor
                    texture_features = ImageProcessor.extract_wavelet_features(roi)
            
            # Classification logic based on geometric features
            trash_type, confidence = TrashClassifier.determine_trash_type(
                circularity, aspect_ratio, extent, solidity, area, w, h
            )
            
            result = {
                'id': obj_id + 1,
                'type': trash_type,
                'confidence': confidence,
                'area': int(area),
                'centroid': (cx, cy),
                'bounding_rect': (x, y, w, h),
                'circularity': round(circularity, 3),
                'aspect_ratio': round(aspect_ratio, 3),
                'extent': round(extent, 3),
                'solidity': round(solidity, 3)
            }
            
            # Tambahkan fitur tekstur jika ada
            if texture_features is not None:
                result['texture_features'] = texture_features.tolist()
            
            return result
            
        except Exception as e:
            print(f"Error classifying contour {obj_id}: {str(e)}")
            return None
    
    @staticmethod
    def determine_trash_type(circularity, aspect_ratio, extent, solidity, area, width, height):
        """Determine trash type based on geometric features"""
        confidence_scores = {}
        
        # === BOTOL PLASTIK ===
        botol_score = TrashClassifier._calculate_botol_score(
            circularity, aspect_ratio, extent, solidity, area
        )
        if botol_score > 45:
            confidence_scores['Botol Plastik'] = min(95, botol_score)
        
        # === KALENG ===
        kaleng_score = TrashClassifier._calculate_kaleng_score(
            circularity, aspect_ratio, extent, solidity
        )
        if kaleng_score > 60:
            confidence_scores['Kaleng'] = min(90, kaleng_score)
        
        # === KARDUS/KERTAS ===
        kardus_score = TrashClassifier._calculate_kardus_score(
            circularity, aspect_ratio, extent, solidity
        )
        if kardus_score > 50:
            confidence_scores['Kardus/Kertas'] = min(85, kardus_score)
        
        # === KANTONG PLASTIK ===
        kantong_score = TrashClassifier._calculate_kantong_score(
            circularity, aspect_ratio, extent, solidity
        )
        if kantong_score > 35:
            confidence_scores['Kantong Plastik'] = min(75, kantong_score)
        
        # Default classification
        if not confidence_scores:
            confidence_scores['Sampah Tidak Dikenal'] = 30
        
        # Resolve conflicts between similar types
        TrashClassifier._resolve_classification_conflicts(confidence_scores)
        
        # Return the classification with highest confidence
        best_type = max(confidence_scores, key=confidence_scores.get)
        best_confidence = confidence_scores[best_type]
        
        return best_type, int(best_confidence)
    
    @staticmethod
    def _calculate_botol_score(circularity, aspect_ratio, extent, solidity, area):
        """Calculate confidence score for Botol Plastik classification"""
        score = 0
        
        # Circularity scoring
        if 0.15 < circularity < 0.85:
            score += 25 if 0.4 < circularity < 0.8 else 15
                
        # Aspect ratio scoring
        if 1.2 < aspect_ratio < 6.0:
            score += 30 if 1.5 < aspect_ratio < 4.0 else 20
                
        # Extent scoring
        if extent > 0.45:
            score += 25 if extent > 0.6 else 15
                
        # Solidity scoring
        if solidity > 0.55:
            score += 25 if solidity > 0.7 else 15
                
        # Area scoring
        if area > 300:
            score += 15 if area > 800 else 10
                
        # Combination bonuses
        if (0.3 < circularity < 0.8 and 1.5 < aspect_ratio < 4.5 and 
            extent > 0.5 and solidity > 0.6 and area > 500):
            score += 20
            
        if (0.4 < circularity < 0.7 and 2.0 < aspect_ratio < 3.5 and 
            extent > 0.65 and solidity > 0.75):
            score += 15
            
        return score
    
    @staticmethod
    def _calculate_kaleng_score(circularity, aspect_ratio, extent, solidity):
        """Calculate confidence score for Kaleng classification"""
        score = 0
        
        if 0.6 < circularity < 1.0:
            score += 30
        if 0.8 < aspect_ratio < 2.0:
            score += 25
        if solidity > 0.8:
            score += 25
        if extent > 0.7:
            score += 15
            
        return score
    
    @staticmethod
    def _calculate_kardus_score(circularity, aspect_ratio, extent, solidity):
        """Calculate confidence score for Kardus/Kertas classification"""
        score = 0
        
        if 0.1 < circularity < 0.5:
            score += 25
        if extent > 0.7:
            score += 30
        if solidity > 0.7:
            score += 20
        if aspect_ratio > 1.2:
            score += 15
            
        return score
    
    @staticmethod
    def _calculate_kantong_score(circularity, aspect_ratio, extent, solidity):
        """Calculate confidence score for Kantong Plastik classification"""
        score = 0
        
        if circularity < 0.5:
            score += 15
        if solidity < 0.7:
            score += 20
        if extent < 0.6:
            score += 25
        if aspect_ratio < 2.0:
            score += 15
        
        # Penalty for bottle-like characteristics
        if (aspect_ratio > 1.8 and solidity > 0.6 and extent > 0.5 and circularity > 0.25):
            score -= 40
            
        return score
    
    @staticmethod
    def _resolve_classification_conflicts(confidence_scores):
        """Resolve conflicts between similar classification types"""
        # Prioritize Botol Plastik over Kantong Plastik when scores are close
        if ('Botol Plastik' in confidence_scores and 
            'Kantong Plastik' in confidence_scores):
            if confidence_scores['Botol Plastik'] >= confidence_scores['Kantong Plastik'] - 15:
                del confidence_scores['Kantong Plastik']
        
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