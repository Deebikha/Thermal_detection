import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import time
from typing import Tuple, List, Dict

class ThermalPreprocessor:
    """Thermal image preprocessing pipeline"""
    
    def __init__(self):
        self.temp_range = (-20, 50)  # Celsius
        
    def adaptive_noise_reduction(self, thermal_img):
    # Apply Gaussian blur
        gaussian = cv2.GaussianBlur(thermal_img, (5, 5), 0)

    # Apply median blur
        median = cv2.medianBlur(thermal_img, 5)

    # Convert everything to float32 before blending
        gaussian_f = gaussian.astype(np.float32)
        median_f = median.astype(np.float32)

    # Weighted combination
        filtered = cv2.addWeighted(gaussian_f, 0.7, median_f, 0.3, 0)

    # Convert back to uint8 for later processing
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        return filtered

    def temperature_normalization(self, thermal_img: np.ndarray) -> np.ndarray:
        """Normalize temperature values and enhance contrast"""
        # Convert to float
        img_float = thermal_img.astype(np.float32)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        if len(img_float.shape) == 2:
            normalized = clahe.apply(img_float.astype(np.uint8))
        else:
            normalized = img_float
            
        # Dynamic range optimization
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def edge_enhancement(self, thermal_img: np.ndarray) -> np.ndarray:
        """Enhance edges and thermal gradients"""

    # Ensure grayscale
        if len(thermal_img.shape) == 3:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
        thermal_img_f = thermal_img.astype(np.float32)
    # Sobel gradients
        sobelx = cv2.Sobel(thermal_img_f, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(thermal_img_f, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

    # Laplacian (works only on single-channel images)
        laplacian = cv2.Laplacian(thermal_img_f, cv2.CV_32F)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    # Combine edges
        enhanced = cv2.addWeighted(thermal_img_f, 0.7, gradient, 0.3, 0)

        return enhanced.astype(np.uint8)

    
    def morphological_processing(self, thermal_img: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance target signatures"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Opening to remove noise
        opening = cv2.morphologyEx(thermal_img, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill gaps
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing
    
    def preprocess(self, thermal_img: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Step 1: Noise reduction
        filtered = self.adaptive_noise_reduction(thermal_img)
        
        # Step 2: Temperature normalization
        normalized = self.temperature_normalization(filtered)
        
        # Step 3: Edge enhancement
        enhanced = self.edge_enhancement(normalized)
        
        # Step 4: Morphological processing
        processed = self.morphological_processing(enhanced)
        
        return processed


class MultiSpectralFusion:
    """Multi-spectral sensor fusion module"""
    
    def __init__(self):
        self.weights = {
            'thermal': 0.5,
            'nir': 0.25,
            'visible': 0.25
        }
    
    def adaptive_fusion(self, thermal: np.ndarray, nir: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Adaptive multi-spectral fusion with learned weights"""
        h, w = thermal.shape[:2]

        # Make sure all images are single-channel
        if len(thermal.shape) == 2:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
        if nir is not None and len(nir.shape) == 2:
            nir = cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR)
        if visible is not None and len(visible.shape) == 2:
            visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR)

    # Resize to same size
        if nir is not None:
            nir = cv2.resize(nir, (w, h))
        else:
            nir = np.zeros_like(thermal)

        if visible is not None:
            visible = cv2.resize(visible, (w, h))
        else:
            visible = np.zeros_like(thermal)

    # Normalize to 0-1
        thermal_norm = cv2.normalize(thermal, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        nir_norm = cv2.normalize(nir, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        visible_norm = cv2.normalize(visible, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Weighted fusion
        fused = (self.weights['thermal'] * thermal_norm +
             self.weights['nir'] * nir_norm +
             self.weights['visible'] * visible_norm)

    # Convert back to 0-255 uint8
        fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

        return fused



class ThermalYOLOv8:
    """Modified YOLOv8 for thermal object detection"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize the thermal-enhanced YOLOv8 model"""
        self.model = YOLO(model_path)
        self.preprocessor = ThermalPreprocessor()
        self.fusion = MultiSpectralFusion()
        
        # Military target classes
        self.target_classes = {
            0: 'person',
            1: 'vehicle_light',
            2: 'vehicle_armored',
            3: 'weapon',
            4: 'aircraft'
        }
        
    def process_frame(self, thermal_img: np.ndarray, 
                     nir_img: np.ndarray = None,
                     visible_img: np.ndarray = None) -> Tuple[np.ndarray, List]:
        """Process a single frame through the detection pipeline"""
        
        # Preprocess thermal image
        thermal_processed = self.preprocessor.preprocess(thermal_img)
        
        # Multi-spectral fusion
        fused_img = self.fusion.adaptive_fusion(thermal_processed, nir_img, visible_img)
        
        # Run YOLOv8 detection
        results = self.model(fused_img, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                }
                detections.append(detection)
        
        # Draw detections
        annotated_frame = self.draw_detections(fused_img.copy(), detections)
        
        return annotated_frame, detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        for det in detections:
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            label = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {conf:.2f}"
            cv2.putText(image, label_text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image


class MilitaryTargetAnalyzer:
    """Military-specific target analysis module"""
    
    def __init__(self):
        self.threat_levels = {
            'person': 'MEDIUM',
            'vehicle_armored': 'HIGH',
            'weapon': 'HIGH',
            'aircraft': 'CRITICAL'
        }
    
    def analyze_targets(self, detections: List[Dict]) -> Dict:
        """Analyze detected targets for military assessment"""
        analysis = {
            'total_targets': len(detections),
            'threat_assessment': {},
            'priority_targets': []
        }
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Threat level assessment
            threat_level = self.threat_levels.get(class_name, 'LOW')
            
            if class_name not in analysis['threat_assessment']:
                analysis['threat_assessment'][class_name] = {
                    'count': 0,
                    'threat_level': threat_level,
                    'avg_confidence': 0
                }
            
            analysis['threat_assessment'][class_name]['count'] += 1
            analysis['threat_assessment'][class_name]['avg_confidence'] += confidence
            
            # Priority targets (high confidence and high threat)
            if confidence > 0.7 and threat_level in ['HIGH', 'CRITICAL']:
                analysis['priority_targets'].append({
                    'class': class_name,
                    'confidence': confidence,
                    'threat': threat_level
                })
        
        # Calculate average confidence
        for class_name in analysis['threat_assessment']:
            count = analysis['threat_assessment'][class_name]['count']
            analysis['threat_assessment'][class_name]['avg_confidence'] /= count
        
        return analysis


def simulate_thermal_from_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """Simulate thermal image from RGB for demo purposes"""
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # Apply thermal-like color mapping
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Add some noise to simulate thermal sensor
    noise = np.random.normal(0, 10, thermal.shape).astype(np.float32)
    thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return thermal


def main():
    """Main execution function"""
    print("=" * 60)
    print("Thermal-Enhanced Multi-Spectral Object Detection System")
    print("Modified YOLOv8 for Military Applications")
    print("=" * 60)
    
    # Initialize system
    print("\n[1/4] Initializing detection system...")
    detector = ThermalYOLOv8('yolov8n.pt')
    analyzer = MilitaryTargetAnalyzer()
    
    # Setup video capture (use webcam or video file)
    print("[2/4] Setting up video capture...")
    cap = cv2.VideoCapture(0)  # Change to video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("[3/4] Starting detection loop...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate thermal image from RGB (for demo)
        thermal_img = simulate_thermal_from_rgb(frame)
        
        # Process frame
        start_time = time.time()
        annotated_frame, detections = detector.process_frame(
            thermal_img, 
            nir_img=None, 
            visible_img=frame
        )
        processing_time = time.time() - start_time
        
        # Analyze targets
        analysis = analyzer.analyze_targets(detections)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        else:
            fps = 0
        
        # Display information
        info_text = [
            f"FPS: {fps:.1f}" if fps > 0 else "FPS: Calculating...",
            f"Processing Time: {processing_time*1000:.1f}ms",
            f"Targets: {analysis['total_targets']}",
            f"Priority: {len(analysis['priority_targets'])}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        # Display results
        cv2.imshow('Thermal Detection System', annotated_frame)
        cv2.imshow('Simulated Thermal Input', thermal_img)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("\n[4/4] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("System shutdown complete.")


if __name__ == "__main__":
    main()