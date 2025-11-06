import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeDetectorCNN, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output

class DeepfakeDetectorLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=2):
        super(DeepfakeDetectorLSTM, self).__init__()
        
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        features = []
        for t in range(seq_len):
            frame_features = self.feature_extractor(x[:, t, :, :, :])
            features.append(frame_features)
        
        features = torch.stack(features, dim=1)
        
        lstm_out, _ = self.lstm(features)
        
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        output = self.classifier(attended_features)
        return output

class EfficientNetDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        self.backbone = models.efficientnet_b4(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        channel_att = self.channel_attention(features)
        features = features * channel_att.unsqueeze(-1).unsqueeze(-1)
        
        pooled = F.adaptive_avg_pool2d(features, 1)
        flattened = torch.flatten(pooled, 1)
        
        output = self.classifier(flattened)
        return output

class MultiModalDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiModalDeepfakeDetector, self).__init__()
        
        self.visual_encoder = models.resnet50(pretrained=True)
        visual_features = self.visual_encoder.fc.in_features
        self.visual_encoder.fc = nn.Identity()
        
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(visual_features + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, visual_input, audio_input):
        visual_features = self.visual_encoder(visual_input)
        audio_features = self.audio_encoder(audio_input)
        
        combined = torch.cat([visual_features, audio_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output

def get_model(model_type='cnn', **kwargs):
    """Factory function to get deepfake detection model"""
    models_dict = {
        'cnn': DeepfakeDetectorCNN,
        'lstm': DeepfakeDetectorLSTM,
        'efficientnet': EfficientNetDeepfakeDetector,
        'multimodal': MultiModalDeepfakeDetector
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = models_dict[model_type](**kwargs)
    logger.info(f"Created {model_type} deepfake detector model")
    
    return model
