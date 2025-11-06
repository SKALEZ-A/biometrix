import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import librosa
from scipy import signal
from PIL import Image

class XceptionDeepfakeDetector(nn.Module):
    """
    Xception-based deepfake detector for face manipulation detection.
    Based on FaceForensics++ architecture.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(XceptionDeepfakeDetector, self).__init__()
        
        # Load pretrained Xception
        self.xception = models.xception(pretrained=pretrained)
        
        # Modify first conv layer to accept different input sizes
        self.xception.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace final FC layer
        num_features = self.xception.fc.in_features
        self.xception.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception(x)


class EfficientNetDeepfakeDetector(nn.Module):
    """
    EfficientNet-based deepfake detector with attention mechanism.
    """
    
    def __init__(self, num_classes: int = 2, model_name: str = 'efficientnet_b4'):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        # Load EfficientNet
        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=True)
        elif model_name == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=True)
        else:
            self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid()
        )
        
        # Classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Classification
        output = self.backbone.classifier(features)
        return output


class AudioDeepfakeDetector(nn.Module):
    """
    LSTM-based audio deepfake detector for voice cloning detection.
    """
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, num_layers: int = 3, num_classes: int = 2):
        super(AudioDeepfakeDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.fc(context_vector)
        return output


class MultiModalDeepfakeDetector(nn.Module):
    """
    Multi-modal deepfake detector combining visual and audio features.
    """
    
    def __init__(self, num_classes: int = 2):
        super(MultiModalDeepfakeDetector, self).__init__()
        
        # Visual branch
        self.visual_branch = EfficientNetDeepfakeDetector(num_classes=512)
        
        # Audio branch
        self.audio_branch = AudioDeepfakeDetector(num_classes=512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        # Extract features from both modalities
        visual_features = self.visual_branch(visual_input)
        audio_features = self.audio_branch(audio_input)
        
        # Concatenate features
        combined_features = torch.cat([visual_features, audio_features], dim=1)
        
        # Fusion and classification
        output = self.fusion(combined_features)
        return output


class DeepfakeDetectionPipeline:
    """
    Complete pipeline for deepfake detection including preprocessing and inference.
    """
    
    def __init__(self, model_type: str = 'xception', device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Initialize model
        if model_type == 'xception':
            self.model = XceptionDeepfakeDetector()
        elif model_type == 'efficientnet':
            self.model = EfficientNetDeepfakeDetector()
        elif model_type == 'audio':
            self.model = AudioDeepfakeDetector()
        elif model_type == 'multimodal':
            self.model = MultiModalDeepfakeDetector()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_weights(self, weights_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {weights_path}")
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def preprocess_video(self, video_path: str, num_frames: int = 32) -> torch.Tensor:
        """Extract and preprocess frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = self.image_transform(frame)
                frames.append(frame_tensor)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return torch.stack(frames).to(self.device)
    
    def preprocess_audio(self, audio_path: str, sr: int = 16000, duration: int = 5) -> torch.Tensor:
        """Preprocess audio for inference."""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).to(self.device)
        
        return audio_tensor
    
    def detect_face_manipulation(self, image_path: str) -> Dict[str, float]:
        """Detect face manipulation in image."""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path)
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            
            return {
                'is_fake': probabilities[0][1].item() > 0.5,
                'fake_probability': probabilities[0][1].item(),
                'real_probability': probabilities[0][0].item(),
                'confidence': max(probabilities[0]).item()
            }
    
    def detect_video_deepfake(self, video_path: str) -> Dict[str, any]:
        """Detect deepfake in video."""
        with torch.no_grad():
            frames = self.preprocess_video(video_path)
            
            # Analyze each frame
            frame_predictions = []
            for frame in frames:
                output = self.model(frame.unsqueeze(0))
                probabilities = F.softmax(output, dim=1)
                frame_predictions.append(probabilities[0][1].item())
            
            # Aggregate predictions
            avg_fake_prob = np.mean(frame_predictions)
            max_fake_prob = np.max(frame_predictions)
            consistency = 1.0 - np.std(frame_predictions)
            
            return {
                'is_fake': avg_fake_prob > 0.5,
                'average_fake_probability': avg_fake_prob,
                'max_fake_probability': max_fake_prob,
                'consistency_score': consistency,
                'frame_predictions': frame_predictions,
                'suspicious_frames': [i for i, p in enumerate(frame_predictions) if p > 0.7]
            }
    
    def detect_voice_cloning(self, audio_path: str) -> Dict[str, float]:
        """Detect voice cloning in audio."""
        with torch.no_grad():
            audio_tensor = self.preprocess_audio(audio_path)
            output = self.model(audio_tensor)
            probabilities = F.softmax(output, dim=1)
            
            return {
                'is_cloned': probabilities[0][1].item() > 0.5,
                'cloned_probability': probabilities[0][1].item(),
                'authentic_probability': probabilities[0][0].item(),
                'confidence': max(probabilities[0]).item()
            }
    
    def analyze_manipulation_artifacts(self, image_path: str) -> Dict[str, any]:
        """Analyze specific manipulation artifacts in image."""
        image = cv2.imread(image_path)
        
        artifacts = {
            'compression_artifacts': self._detect_compression_artifacts(image),
            'blending_inconsistencies': self._detect_blending_issues(image),
            'lighting_inconsistencies': self._detect_lighting_issues(image),
            'resolution_mismatches': self._detect_resolution_issues(image),
            'color_inconsistencies': self._detect_color_issues(image)
        }
        
        return artifacts
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> float:
        """Detect JPEG compression artifacts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply DCT
        dct = cv2.dct(np.float32(gray))
        
        # Analyze high-frequency components
        high_freq = dct[gray.shape[0]//2:, gray.shape[1]//2:]
        artifact_score = np.std(high_freq) / (np.mean(np.abs(high_freq)) + 1e-6)
        
        return min(artifact_score / 10.0, 1.0)
    
    def _detect_blending_issues(self, image: np.ndarray) -> float:
        """Detect blending inconsistencies."""
        # Edge detection
        edges = cv2.Canny(image, 100, 200)
        
        # Analyze edge continuity
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        discontinuity_score = np.sum(edges != dilated) / edges.size
        
        return min(discontinuity_score * 10, 1.0)
    
    def _detect_lighting_issues(self, image: np.ndarray) -> float:
        """Detect lighting inconsistencies."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Analyze lighting gradients
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        inconsistency_score = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)
        
        return min(inconsistency_score / 5.0, 1.0)
    
    def _detect_resolution_issues(self, image: np.ndarray) -> float:
        """Detect resolution mismatches."""
        # Analyze frequency content in different regions
        regions = [
            image[:image.shape[0]//2, :image.shape[1]//2],
            image[:image.shape[0]//2, image.shape[1]//2:],
            image[image.shape[0]//2:, :image.shape[1]//2],
            image[image.shape[0]//2:, image.shape[1]//2:]
        ]
        
        freq_contents = []
        for region in regions:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            freq_contents.append(np.mean(magnitude))
        
        resolution_variance = np.std(freq_contents) / (np.mean(freq_contents) + 1e-6)
        
        return min(resolution_variance, 1.0)
    
    def _detect_color_issues(self, image: np.ndarray) -> float:
        """Detect color inconsistencies."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze color distribution
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        
        # Calculate color histogram
        h_hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
        
        # Detect unusual color distributions
        h_entropy = -np.sum(h_hist * np.log(h_hist + 1e-6))
        s_entropy = -np.sum(s_hist * np.log(s_hist + 1e-6))
        
        color_inconsistency = abs(h_entropy - s_entropy) / max(h_entropy, s_entropy)
        
        return min(color_inconsistency, 1.0)


class DeepfakeTrainer:
    """
    Trainer class for deepfake detection models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filepath: str, epoch: int, best_acc: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': best_acc
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
