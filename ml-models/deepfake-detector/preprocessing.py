import cv2
import numpy as np
import dlib
from typing import Tuple, List, Optional, Dict
import face_recognition
from scipy.spatial import distance
from imutils import face_utils
import mediapipe as mp
from PIL import Image
import torch
from torchvision import transforms

class BiometricPreprocessor:
    """
    Advanced preprocessing for biometric deepfake detection
    """
    
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image_path: str) -> Dict:
        """
        Comprehensive image preprocessing pipeline
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detect_faces(rgb_image)
        if len(faces) == 0:
            raise ValueError("No faces detected in image")
        
        results = []
        for face_bbox in faces:
            face_data = {
                'bbox': face_bbox,
                'aligned_face': None,
                'landmarks': None,
                'features': {},
                'quality_metrics': {}
            }
            
            # Extract face region
            x, y, w, h = face_bbox
            face_roi = rgb_image[y:y+h, x:x+w]
            
            # Align face
            aligned_face = self.align_face(rgb_image, face_bbox)
            face_data['aligned_face'] = aligned_face
            
            # Extract landmarks
            landmarks = self.extract_landmarks(rgb_image, face_bbox)
            face_data['landmarks'] = landmarks
            
            # Extract features
            face_data['features'] = self.extract_facial_features(aligned_face, landmarks)
            
            # Quality assessment
            face_data['quality_metrics'] = self.assess_image_quality(face_roi)
            
            # Texture analysis
            face_data['texture_features'] = self.analyze_texture(face_roi)
            
            # Frequency analysis
            face_data['frequency_features'] = self.analyze_frequency(face_roi)
            
            # Color analysis
            face_data['color_features'] = self.analyze_color(face_roi)
            
            results.append(face_data)
        
        return results
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray, 1)
        
        bboxes = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def align_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Align face using facial landmarks"""
        x, y, w, h = bbox
        
        # Get facial landmarks
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rect = dlib.rectangle(x, y, x+w, y+h)
        shape = self.landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Get eye centers
        left_eye = shape[36:42].mean(axis=0).astype(int)
        right_eye = shape[42:48].mean(axis=0).astype(int)
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get rotation matrix
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, 
                      (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Crop face
        aligned_face = aligned[y:y+h, x:x+w]
        aligned_face = cv2.resize(aligned_face, (224, 224))
        
        return aligned_face
    
    def extract_landmarks(self, image: np.ndarray, 
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract 68 facial landmarks"""
        x, y, w, h = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rect = dlib.rectangle(x, y, x+w, y+h)
        shape = self.landmark_predictor(gray, rect)
        landmarks = face_utils.shape_to_np(shape)
        
        return landmarks
    
    def extract_facial_features(self, face: np.ndarray, 
                                landmarks: np.ndarray) -> Dict:
        """Extract geometric facial features"""
        features = {}
        
        # Eye aspect ratio (EAR)
        features['left_ear'] = self.calculate_ear(landmarks[36:42])
        features['right_ear'] = self.calculate_ear(landmarks[42:48])
        features['avg_ear'] = (features['left_ear'] + features['right_ear']) / 2
        
        # Mouth aspect ratio (MAR)
        features['mar'] = self.calculate_mar(landmarks[48:68])
        
        # Face symmetry
        features['symmetry_score'] = self.calculate_symmetry(landmarks)
        
        # Inter-eye distance
        left_eye_center = landmarks[36:42].mean(axis=0)
        right_eye_center = landmarks[42:48].mean(axis=0)
        features['inter_eye_distance'] = distance.euclidean(left_eye_center, right_eye_center)
        
        # Face width to height ratio
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        features['face_ratio'] = face_width / (face_height + 1e-6)
        
        # Nose to mouth distance
        nose_tip = landmarks[30]
        mouth_center = landmarks[48:68].mean(axis=0)
        features['nose_mouth_distance'] = distance.euclidean(nose_tip, mouth_center)
        
        return features
    
    @staticmethod
    def calculate_ear(eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    @staticmethod
    def calculate_mar(mouth_landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio"""
        A = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        B = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        C = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        mar = (A + B) / (2.0 * C + 1e-6)
        return mar
    
    @staticmethod
    def calculate_symmetry(landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        # Split face into left and right
        center_x = landmarks[:, 0].mean()
        left_points = landmarks[landmarks[:, 0] < center_x]
        right_points = landmarks[landmarks[:, 0] >= center_x]
        
        # Mirror right side
        right_mirrored = right_points.copy()
        right_mirrored[:, 0] = 2 * center_x - right_mirrored[:, 0]
        
        # Calculate distance between left and mirrored right
        if len(left_points) > 0 and len(right_mirrored) > 0:
            distances = []
            for lp in left_points:
                min_dist = np.min([distance.euclidean(lp, rp) for rp in right_mirrored])
                distances.append(min_dist)
            symmetry = 1.0 / (1.0 + np.mean(distances))
        else:
            symmetry = 0.0
        
        return symmetry
    
    def assess_image_quality(self, face: np.ndarray) -> Dict:
        """Assess image quality metrics"""
        metrics = {}
        
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        metrics['blur_score'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['is_blurry'] = metrics['blur_score'] < 100
        
        # Brightness
        metrics['brightness'] = np.mean(gray)
        metrics['is_too_dark'] = metrics['brightness'] < 50
        metrics['is_too_bright'] = metrics['brightness'] > 200
        
        # Contrast
        metrics['contrast'] = np.std(gray)
        metrics['is_low_contrast'] = metrics['contrast'] < 30
        
        # Noise estimation
        metrics['noise_level'] = self.estimate_noise(gray)
        
        # Resolution
        metrics['resolution'] = face.shape[0] * face.shape[1]
        metrics['is_low_resolution'] = metrics['resolution'] < 50000
        
        return metrics
    
    @staticmethod
    def estimate_noise(image: np.ndarray) -> float:
        """Estimate noise level in image"""
        H, W = image.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, np.array(M)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
        
        return sigma
    
    def analyze_texture(self, face: np.ndarray) -> Dict:
        """Analyze texture features"""
        features = {}
        
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Patterns
        lbp = self.compute_lbp(gray)
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_hist'] = np.histogram(lbp, bins=256)[0]
        
        # Gabor filters
        gabor_features = self.apply_gabor_filters(gray)
        features['gabor_mean'] = np.mean(gabor_features)
        features['gabor_std'] = np.std(gabor_features)
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    @staticmethod
    def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern"""
        lbp = np.zeros_like(image)
        
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary_string = ''
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    
                    x, y = int(round(x)), int(round(y))
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        binary_string += '1' if image[x, y] >= center else '0'
                
                lbp[i, j] = int(binary_string, 2) if binary_string else 0
        
        return lbp
    
    def apply_gabor_filters(self, image: np.ndarray) -> np.ndarray:
        """Apply Gabor filters for texture analysis"""
        filters = []
        ksize = 31
        
        for theta in np.arange(0, np.pi, np.pi / 4):
            for sigma in [3, 5]:
                for lambd in [5, 10]:
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0)
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    filters.append(filtered)
        
        return np.array(filters)
    
    def analyze_frequency(self, face: np.ndarray) -> Dict:
        """Analyze frequency domain features"""
        features = {}
        
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        features['fft_mean'] = np.mean(magnitude_spectrum)
        features['fft_std'] = np.std(magnitude_spectrum)
        features['fft_max'] = np.max(magnitude_spectrum)
        
        # High frequency content
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        high_freq = f_shift * mask
        features['high_freq_energy'] = np.sum(np.abs(high_freq))
        
        # DCT
        dct = cv2.dct(np.float32(gray))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        
        return features
    
    def analyze_color(self, face: np.ndarray) -> Dict:
        """Analyze color features"""
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'{channel}_mean'] = np.mean(face[:, :, i])
            features[f'{channel}_std'] = np.std(face[:, :, i])
        
        # HSV statistics
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])
        
        # Color histogram
        hist_r = cv2.calcHist([face], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([face], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([face], [2], None, [256], [0, 256])
        
        features['hist_r_entropy'] = self.calculate_entropy(hist_r)
        features['hist_g_entropy'] = self.calculate_entropy(hist_g)
        features['hist_b_entropy'] = self.calculate_entropy(hist_b)
        
        # Skin tone analysis
        features['skin_tone_score'] = self.analyze_skin_tone(face)
        
        return features
    
    @staticmethod
    def calculate_entropy(histogram: np.ndarray) -> float:
        """Calculate entropy of histogram"""
        histogram = histogram.flatten()
        histogram = histogram / (np.sum(histogram) + 1e-6)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-6))
        return entropy
    
    def analyze_skin_tone(self, face: np.ndarray) -> float:
        """Analyze skin tone consistency"""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)
        
        # Skin tone range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Calculate skin pixel ratio
        skin_ratio = np.sum(mask > 0) / mask.size
        
        return skin_ratio
    
    def preprocess_video(self, video_path: str, 
                        max_frames: int = 100) -> List[Dict]:
        """Preprocess video for deepfake detection"""
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        processed_frames = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Save frame temporarily
                    temp_path = f'temp_frame_{frame_count}.jpg'
                    cv2.imwrite(temp_path, frame)
                    
                    # Process frame
                    frame_data = self.preprocess_image(temp_path)
                    frame_data[0]['frame_number'] = frame_count
                    processed_frames.append(frame_data[0])
                    
                    # Clean up
                    import os
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        return processed_frames


def main():
    """Test preprocessing pipeline"""
    preprocessor = BiometricPreprocessor()
    
    # Test with sample image
    image_path = 'sample_face.jpg'
    
    try:
        results = preprocessor.preprocess_image(image_path)
        
        print(f"Processed {len(results)} faces")
        
        for i, face_data in enumerate(results):
            print(f"\nFace {i+1}:")
            print(f"  Bbox: {face_data['bbox']}")
            print(f"  Landmarks: {face_data['landmarks'].shape}")
            print(f"  Features: {list(face_data['features'].keys())}")
            print(f"  Quality metrics: {face_data['quality_metrics']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
