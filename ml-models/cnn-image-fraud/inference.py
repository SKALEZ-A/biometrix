"""
Inference engine for CNN-based image fraud detection
Real-time fraud detection on images and documents
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import time
from model_architecture import create_model, FraudDetectionEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionInference:
    """Inference engine for fraud detection"""
    
    def __init__(self, model_path: str, model_type: str = 'resnet',
                 device: str = 'cuda', class_names: Optional[List[str]] = None):
        self.device = device
        self.model_type = model_type
        self.class_names = class_names or [f"Class_{i}" for i in range(10)]
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Inference engine initialized with {model_type} on {device}")
    
    def _load_model(self, model_path: str, model_type: str):
        """Load trained model"""
        model = create_model(model_type)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference"""
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Union[str, Image.Image, np.ndarray],
                return_probabilities: bool = True) -> Dict:
        """
        Predict fraud on a single image
        
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        inference_time = time.time() - start_time
        
        result = {
            'predicted_class': self.class_names[predicted_class.item()],
            'predicted_class_id': predicted_class.item(),
            'confidence': confidence.item(),
            'inference_time_ms': inference_time * 1000
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]],
                     batch_size: int = 32) -> List[Dict]:
        """Predict fraud on a batch of images"""
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = torch.cat([self.preprocess_image(img) for img in batch])
            
            with torch.no_grad():
                outputs = self.model(batch_tensors)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
            
            for j in range(len(batch)):
                result = {
                    'predicted_class': self.class_names[predicted_classes[j].item()],
                    'predicted_class_id': predicted_classes[j].item(),
                    'confidence': confidences[j].item(),
                    'probabilities': {
                        self.class_names[k]: probabilities[j][k].item()
                        for k in range(len(self.class_names))
                    }
                }
                results.append(result)
        
        return results
    
    def detect_fraud_regions(self, image: Union[str, Image.Image, np.ndarray],
                            threshold: float = 0.7) -> Dict:
        """
        Detect specific regions in image that indicate fraud
        Uses gradient-based attention
        """
        
        image_tensor = self.preprocess_image(image)
        image_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(image_tensor)
        predicted_class = outputs.argmax(dim=1)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        outputs[0, predicted_class].backward()
        
        # Get gradients
        gradients = image_tensor.grad.data
        
        # Create attention map
        attention_map = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Find suspicious regions
        suspicious_regions = np.where(attention_map > threshold)
        
        return {
            'attention_map': attention_map.tolist(),
            'suspicious_regions': {
                'coordinates': list(zip(suspicious_regions[0].tolist(),
                                      suspicious_regions[1].tolist())),
                'count': len(suspicious_regions[0])
            },
            'predicted_class': self.class_names[predicted_class.item()],
            'confidence': F.softmax(outputs, dim=1)[0, predicted_class].item()
        }
    
    def analyze_document_authenticity(self, document_image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Comprehensive document authenticity analysis
        """
        
        # Basic prediction
        prediction = self.predict(document_image)
        
        # Fraud region detection
        fraud_regions = self.detect_fraud_regions(document_image)
        
        # Combine results
        analysis = {
            'is_authentic': prediction['predicted_class'] == 'authentic',
            'fraud_probability': 1.0 - prediction['confidence'] if prediction['predicted_class'] == 'authentic' else prediction['confidence'],
            'fraud_type': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'suspicious_regions_count': fraud_regions['suspicious_regions']['count'],
            'risk_level': self._calculate_risk_level(prediction, fraud_regions),
            'recommendations': self._generate_recommendations(prediction, fraud_regions)
        }
        
        return analysis
    
    def _calculate_risk_level(self, prediction: Dict, fraud_regions: Dict) -> str:
        """Calculate overall risk level"""
        
        confidence = prediction['confidence']
        suspicious_count = fraud_regions['suspicious_regions']['count']
        
        if confidence > 0.9 and suspicious_count < 10:
            return 'LOW'
        elif confidence > 0.7 and suspicious_count < 50:
            return 'MEDIUM'
        elif confidence > 0.5:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _generate_recommendations(self, prediction: Dict, fraud_regions: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if prediction['confidence'] < 0.7:
            recommendations.append("Manual review recommended due to low confidence")
        
        if fraud_regions['suspicious_regions']['count'] > 50:
            recommendations.append("Multiple suspicious regions detected - detailed forensic analysis needed")
        
        if prediction['predicted_class'] in ['forged_signature', 'altered_document']:
            recommendations.append("Document shows signs of manipulation - verify with original source")
        
        if not recommendations:
            recommendations.append("Document appears authentic - proceed with standard verification")
        
        return recommendations

class EnsembleInference:
    """Inference using ensemble of models"""
    
    def __init__(self, model_paths: List[str], model_types: List[str],
                 weights: Optional[List[float]] = None, device: str = 'cuda'):
        self.device = device
        self.models = []
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
        
        for path, model_type in zip(model_paths, model_types):
            inference_engine = FraudDetectionInference(path, model_type, device)
            self.models.append(inference_engine)
        
        logger.info(f"Ensemble inference initialized with {len(self.models)} models")
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """Ensemble prediction"""
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(image, return_probabilities=True)
            predictions.append((pred, weight))
        
        # Weighted voting
        weighted_probs = {}
        for pred, weight in predictions:
            for class_name, prob in pred['probabilities'].items():
                if class_name not in weighted_probs:
                    weighted_probs[class_name] = 0.0
                weighted_probs[class_name] += prob * weight
        
        # Get final prediction
        final_class = max(weighted_probs, key=weighted_probs.get)
        final_confidence = weighted_probs[final_class]
        
        return {
            'predicted_class': final_class,
            'confidence': final_confidence,
            'probabilities': weighted_probs,
            'individual_predictions': [pred for pred, _ in predictions]
        }

class BatchProcessor:
    """Process large batches of images efficiently"""
    
    def __init__(self, inference_engine: FraudDetectionInference,
                 batch_size: int = 32, num_workers: int = 4):
        self.inference_engine = inference_engine
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def process_directory(self, directory: str, output_file: str = 'results.json'):
        """Process all images in a directory"""
        
        image_paths = list(Path(directory).glob('**/*.jpg')) + \
                     list(Path(directory).glob('**/*.png')) + \
                     list(Path(directory).glob('**/*.jpeg'))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_results = self.inference_engine.predict_batch(
                [str(p) for p in batch_paths],
                batch_size=self.batch_size
            )
            
            for path, result in zip(batch_paths, batch_results):
                result['image_path'] = str(path)
                results.append(result)
            
            logger.info(f"Processed {min(i + self.batch_size, len(image_paths))}/{len(image_paths)} images")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        return results

if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Single model inference
    # inference = FraudDetectionInference(
    #     model_path='checkpoints/best_model.pth',
    #     model_type='resnet',
    #     device=device
    # )
    
    # result = inference.predict('path/to/image.jpg')
    # print(json.dumps(result, indent=2))
    
    logger.info("Inference engine ready")
