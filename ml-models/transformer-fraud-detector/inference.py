import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime

from model import create_fraud_detector


class FraudDetectorInference:
    """
    Inference engine for fraud detection models
    Handles model loading, preprocessing, and prediction
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create and load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Fraud threshold: {self.threshold}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        model = create_fraud_detector(
            input_dim=self.config['input_dim'],
            model_type=self.config.get('model_type', 'standard'),
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_encoder_layers=self.config.get('num_encoder_layers', 4)
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess(
        self,
        transactions: Union[List[Dict], Dict]
    ) -> torch.Tensor:
        """
        Preprocess transaction data for inference
        
        Args:
            transactions: Single transaction or list of transactions
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(transactions, dict):
            transactions = [transactions]
        
        # Extract features
        features_list = []
        for trans in transactions:
            features = self._extract_features(trans)
            features_list.append(features)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        # Pad or truncate to sequence length
        seq_length = self.config.get('seq_length', 50)
        if features_tensor.shape[1] < seq_length:
            padding = torch.zeros(
                features_tensor.shape[0],
                seq_length - features_tensor.shape[1],
                features_tensor.shape[2]
            )
            features_tensor = torch.cat([features_tensor, padding], dim=1)
        else:
            features_tensor = features_tensor[:, :seq_length, :]
        
        return features_tensor.to(self.device)
    
    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract numerical features from transaction"""
        # This should match your training feature extraction
        features = []
        
        # Amount features
        features.append(transaction.get('amount', 0.0))
        features.append(np.log1p(transaction.get('amount', 0.0)))
        
        # Time features
        timestamp = transaction.get('timestamp', 0)
        features.append(timestamp % 86400)  # Time of day
        features.append(timestamp % 604800)  # Day of week
        
        # Location features
        features.append(transaction.get('latitude', 0.0))
        features.append(transaction.get('longitude', 0.0))
        
        # Merchant features
        features.append(transaction.get('merchant_category', 0))
        features.append(transaction.get('merchant_risk_score', 0.0))
        
        # User features
        features.append(transaction.get('user_age', 0))
        features.append(transaction.get('account_age_days', 0))
        features.append(transaction.get('previous_transactions', 0))
        
        # Device features
        features.append(transaction.get('device_fingerprint_hash', 0))
        features.append(transaction.get('ip_risk_score', 0.0))
        
        # Behavioral features
        features.append(transaction.get('velocity_1h', 0))
        features.append(transaction.get('velocity_24h', 0))
        features.append(transaction.get('avg_transaction_amount', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    @torch.no_grad()
    def predict(
        self,
        transactions: Union[List[Dict], Dict],
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict:
        """
        Predict fraud probability for transactions
        
        Args:
            transactions: Transaction data
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return encoded features
            
        Returns:
            Dictionary containing predictions and optional outputs
        """
        # Preprocess
        features = self.preprocess(transactions)
        
        # Forward pass
        logits, encoded = self.model(features)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        fraud_probs = probs[:, 1].cpu().numpy()
        
        # Get predictions
        predictions = (fraud_probs >= self.threshold).astype(int)
        
        # Prepare results
        results = {
            'predictions': predictions.tolist(),
            'is_fraud': predictions.tolist(),
        }
        
        if return_probabilities:
            results['fraud_probability'] = fraud_probs.tolist()
            results['legitimate_probability'] = probs[:, 0].cpu().numpy().tolist()
        
        if return_features:
            results['encoded_features'] = encoded.cpu().numpy().tolist()
        
        # Add risk levels
        results['risk_level'] = self._get_risk_levels(fraud_probs)
        
        return results
    
    def _get_risk_levels(self, probabilities: np.ndarray) -> List[str]:
        """Convert probabilities to risk levels"""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append('low')
            elif prob < 0.6:
                risk_levels.append('medium')
            elif prob < 0.8:
                risk_levels.append('high')
            else:
                risk_levels.append('critical')
        return risk_levels
    
    def predict_batch(
        self,
        transactions_batch: List[List[Dict]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict fraud for multiple transaction sequences
        
        Args:
            transactions_batch: List of transaction sequences
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(transactions_batch), batch_size):
            batch = transactions_batch[i:i + batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)
        
        return results
    
    def explain_prediction(
        self,
        transaction: Dict,
        top_k: int = 5
    ) -> Dict:
        """
        Explain fraud prediction using attention weights
        
        Args:
            transaction: Transaction to explain
            top_k: Number of top features to return
            
        Returns:
            Explanation dictionary
        """
        # Get prediction
        result = self.predict(transaction, return_features=True)
        
        # Get attention weights if available
        attention_weights = self.model.get_attention_weights()
        
        explanation = {
            'prediction': result['predictions'][0],
            'fraud_probability': result['fraud_probability'][0],
            'risk_level': result['risk_level'][0],
        }
        
        if attention_weights is not None:
            # Analyze attention to identify important features
            attention_scores = attention_weights.mean(dim=1).squeeze().cpu().numpy()
            top_indices = np.argsort(attention_scores)[-top_k:][::-1]
            
            explanation['important_features'] = {
                'indices': top_indices.tolist(),
                'scores': attention_scores[top_indices].tolist(),
            }
        
        return explanation
    
    def update_threshold(self, new_threshold: float):
        """Update fraud detection threshold"""
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = new_threshold
        print(f"Threshold updated to {new_threshold}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': self.config.get('model_type', 'standard'),
            'input_dim': self.config['input_dim'],
            'd_model': self.config.get('d_model', 256),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'threshold': self.threshold,
        }


class EnsembleFraudDetector:
    """
    Ensemble of multiple fraud detection models for improved accuracy
    """
    
    def __init__(
        self,
        model_paths: List[str],
        config_paths: List[str],
        weights: Optional[List[float]] = None,
        device: str = 'cuda'
    ):
        self.models = []
        
        # Load all models
        for model_path, config_path in zip(model_paths, config_paths):
            detector = FraudDetectorInference(
                model_path=model_path,
                config_path=config_path,
                device=device
            )
            self.models.append(detector)
        
        # Set ensemble weights
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        print(f"Ensemble created with {len(self.models)} models")
        print(f"Weights: {self.weights}")
    
    def predict(
        self,
        transactions: Union[List[Dict], Dict],
        aggregation: str = 'weighted_average'
    ) -> Dict:
        """
        Predict using ensemble of models
        
        Args:
            transactions: Transaction data
            aggregation: Method to aggregate predictions
                        ('weighted_average', 'majority_vote', 'max', 'min')
            
        Returns:
            Aggregated prediction results
        """
        # Get predictions from all models
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            result = model.predict(transactions, return_probabilities=True)
            all_predictions.append(result['predictions'])
            all_probabilities.append(result['fraud_probability'])
        
        # Aggregate predictions
        if aggregation == 'weighted_average':
            fraud_probs = np.average(all_probabilities, axis=0, weights=self.weights)
            predictions = (fraud_probs >= 0.5).astype(int)
        
        elif aggregation == 'majority_vote':
            predictions = np.round(np.average(all_predictions, axis=0)).astype(int)
            fraud_probs = np.average(all_probabilities, axis=0)
        
        elif aggregation == 'max':
            fraud_probs = np.max(all_probabilities, axis=0)
            predictions = (fraud_probs >= 0.5).astype(int)
        
        elif aggregation == 'min':
            fraud_probs = np.min(all_probabilities, axis=0)
            predictions = (fraud_probs >= 0.5).astype(int)
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return {
            'predictions': predictions.tolist(),
            'fraud_probability': fraud_probs.tolist(),
            'individual_predictions': all_predictions,
            'individual_probabilities': all_probabilities,
            'aggregation_method': aggregation,
        }


def main():
    """Example usage"""
    # Initialize detector
    detector = FraudDetectorInference(
        model_path='checkpoints/best_model.pt',
        config_path='config.json',
        device='cuda',
        threshold=0.5
    )
    
    # Example transaction
    transaction = {
        'amount': 1500.00,
        'timestamp': 1635724800,
        'latitude': 37.7749,
        'longitude': -122.4194,
        'merchant_category': 5411,
        'merchant_risk_score': 0.3,
        'user_age': 35,
        'account_age_days': 730,
        'previous_transactions': 150,
        'device_fingerprint_hash': 12345,
        'ip_risk_score': 0.2,
        'velocity_1h': 2,
        'velocity_24h': 5,
        'avg_transaction_amount': 250.00,
    }
    
    # Make prediction
    result = detector.predict(transaction)
    
    print("\nPrediction Results:")
    print(f"Is Fraud: {result['is_fraud'][0]}")
    print(f"Fraud Probability: {result['fraud_probability'][0]:.4f}")
    print(f"Risk Level: {result['risk_level'][0]}")
    
    # Get explanation
    explanation = detector.explain_prediction(transaction)
    print("\nExplanation:")
    print(json.dumps(explanation, indent=2))
    
    # Model info
    info = detector.get_model_info()
    print("\nModel Info:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
