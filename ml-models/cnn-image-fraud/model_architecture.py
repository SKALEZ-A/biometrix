"""
CNN-based Image Fraud Detection Model Architecture
Detects fraudulent documents, fake IDs, and manipulated images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Optional
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for deep CNN architecture"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """Spatial attention mechanism for focusing on fraud indicators"""
    
    def __init__(self, channels: int):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class ImageFraudCNN(nn.Module):
    """
    Advanced CNN for detecting image-based fraud
    Supports multiple fraud types: document forgery, ID manipulation, deepfakes
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(ImageFraudCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Attention modules
        self.attention1 = AttentionModule(128)
        self.attention2 = AttentionModule(256)
        self.attention3 = AttentionModule(512)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks with attention
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention1(x)
        x = self.layer3(x)
        x = self.attention2(x)
        x = self.layer4(x)
        x = self.attention3(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class EfficientNetFraudDetector(nn.Module):
    """EfficientNet-based fraud detector for production efficiency"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super(EfficientNetFraudDetector, self).__init__()
        
        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class MultiScaleFraudDetector(nn.Module):
    """Multi-scale analysis for detecting fraud at different image resolutions"""
    
    def __init__(self, num_classes: int = 10):
        super(MultiScaleFraudDetector, self).__init__()
        
        # Three parallel branches for different scales
        self.scale1 = self._create_scale_branch(3, 64)
        self.scale2 = self._create_scale_branch(3, 64)
        self.scale3 = self._create_scale_branch(3, 64)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _create_scale_branch(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        # Process at different scales
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(F.interpolate(x, scale_factor=0.5))
        scale3_out = self.scale3(F.interpolate(x, scale_factor=0.25))
        
        # Resize to same dimensions
        scale2_out = F.interpolate(scale2_out, size=scale1_out.shape[2:])
        scale3_out = F.interpolate(scale3_out, size=scale1_out.shape[2:])
        
        # Concatenate and fuse
        fused = torch.cat([scale1_out, scale2_out, scale3_out], dim=1)
        fused = self.fusion(fused)
        fused = torch.flatten(fused, 1)
        
        # Classify
        output = self.classifier(fused)
        return output

class DocumentAuthenticityNet(nn.Module):
    """Specialized network for document authenticity verification"""
    
    def __init__(self):
        super(DocumentAuthenticityNet, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Texture analysis branch
        self.texture_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Edge analysis branch
        self.edge_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Authentic vs Fraudulent
        )
    
    def forward(self, x):
        features = self.features(x)
        
        texture = self.texture_branch(features)
        texture = torch.flatten(texture, 1)
        
        edges = self.edge_branch(features)
        edges = torch.flatten(edges, 1)
        
        combined = torch.cat([texture, edges], dim=1)
        output = self.classifier(combined)
        
        return output

def create_model(model_type: str = 'resnet', **kwargs):
    """Factory function to create fraud detection models"""
    
    models_dict = {
        'resnet': ImageFraudCNN,
        'efficientnet': EfficientNetFraudDetector,
        'multiscale': MultiScaleFraudDetector,
        'document': DocumentAuthenticityNet
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models_dict[model_type](**kwargs)

class FraudDetectionEnsemble(nn.Module):
    """Ensemble of multiple models for robust fraud detection"""
    
    def __init__(self, models_list: List[nn.Module], weights: Optional[List[float]] = None):
        super(FraudDetectionEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.weights = weights if weights else [1.0 / len(models_list)] * len(models_list)
    
    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output
