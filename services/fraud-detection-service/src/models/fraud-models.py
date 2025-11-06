"""
Fraud Detection ML Models
Implements multiple machine learning models for fraud detection including
XGBoost, Isolation Forest, LSTM, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import joblib
import json

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: float
    model_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    anomaly_score: float
    risk_factors: List[str]
    timestamp: datetime

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: List[List[int]]

class FraudFeatureEngineering:
    """Feature engineering for fraud detection"""
    
    @staticmethod
    def extract_transaction_features(transaction: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from transaction data"""
        features = {}
        
        # Amount-based features
        features['amount'] = float(transaction.get('amount', 0))
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_squared'] = features['amount'] ** 2
        
        # Time-based features
        timestamp = transaction.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_night'] = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
        
        # Merchant features
        features['merchant_category_code'] = hash(transaction.get('merchant_category', '')) % 1000
        features['merchant_risk_score'] = float(transaction.get('merchant_risk_score', 0.5))
        
        # User behavior features
        features['user_transaction_count'] = int(transaction.get('user_transaction_count', 0))
        features['user_avg_amount'] = float(transaction.get('user_avg_amount', 0))
        features['amount_deviation'] = abs(features['amount'] - features['user_avg_amount'])
        features['amount_deviation_ratio'] = (
            features['amount_deviation'] / features['user_avg_amount']
            if features['user_avg_amount'] > 0 else 0
        )
        
        # Device features
        features['device_trust_score'] = float(transaction.get('device_trust_score', 0.5))
        features['is_new_device'] = int(transaction.get('is_new_device', False))
        features['device_location_match'] = int(transaction.get('device_location_match', True))
        
        # Location features
        features['distance_from_home'] = float(transaction.get('distance_from_home', 0))
        features['is_foreign_transaction'] = int(transaction.get('is_foreign_transaction', False))
        features['location_velocity'] = float(transaction.get('location_velocity', 0))
        
        # Behavioral biometric features
        features['biometric_match_score'] = float(transaction.get('biometric_match_score', 1.0))
        features['keystroke_anomaly'] = float(transaction.get('keystroke_anomaly', 0))
        features['mouse_anomaly'] = float(transaction.get('mouse_anomaly', 0))
        
        # Network features
        features['ip_reputation_score'] = float(transaction.get('ip_reputation_score', 0.5))
        features['is_vpn'] = int(transaction.get('is_vpn', False))
        features['is_tor'] = int(transaction.get('is_tor', False))
        
        # Historical features
        features['declined_transactions_24h'] = int(transaction.get('declined_transactions_24h', 0))
        features['chargebacks_90d'] = int(transaction.get('chargebacks_90d', 0))
        features['velocity_1h'] = int(transaction.get('velocity_1h', 0))
        features['velocity_24h'] = int(transaction.get('velocity_24h', 0))
        
        return features
    
    @staticmethod
    def create_interaction_features(features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features"""
        interactions = {}
        
        # Amount × Time interactions
        interactions['amount_x_hour'] = features['amount'] * features['hour_of_day']
        interactions['amount_x_weekend'] = features['amount'] * features['is_weekend']
        interactions['amount_x_night'] = features['amount'] * features['is_night']
        
        # Biometric × Amount interactions
        interactions['biometric_x_amount'] = (
            features['biometric_match_score'] * features['amount_log']
        )
        
        # Device × Location interactions
        interactions['device_x_location'] = (
            features['device_trust_score'] * features['device_location_match']
        )
        
        # Velocity × Amount interactions
        interactions['velocity_x_amount'] = features['velocity_24h'] * features['amount_log']
        
        return interactions
    
    @staticmethod
    def normalize_features(features: Dict[str, float], 
                          scaler_params: Optional[Dict] = None) -> Dict[str, float]:
        """Normalize features using z-score normalization"""
        if scaler_params is None:
            return features
        
        normalized = {}
        for key, value in features.items():
            if key in scaler_params:
                mean = scaler_params[key]['mean']
                std = scaler_params[key]['std']
                normalized[key] = (value - mean) / std if std > 0 else 0
            else:
                normalized[key] = value
        
        return normalized

class XGBoostFraudDetector:
    """XGBoost-based fraud detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = []
        self.threshold = 0.5
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: List[str],
              params: Optional[Dict] = None) -> ModelMetrics:
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for XGBoostFraudDetector")
        
        self.feature_names = feature_names
        
        # Default parameters optimized for fraud detection
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]),  # Handle imbalance
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        
        if params:
            default_params.update(params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)
        
        # Calculate metrics
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        return self._calculate_metrics(y, y_pred, y_pred_proba)
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, bool]:
        """Predict fraud probability"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert features to array in correct order
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        fraud_prob = self.model.predict_proba(X)[0, 1]
        is_fraud = fraud_prob >= self.threshold
        
        return fraud_prob, is_fraud
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))
    
    def save_model(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> ModelMetrics:
        """Calculate model performance metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1_score=f1_score(y_true, y_pred),
            auc_roc=roc_auc_score(y_true, y_pred_proba),
            false_positive_rate=fp / (fp + tn) if (fp + tn) > 0 else 0,
            false_negative_rate=fn / (fn + tp) if (fn + tp) > 0 else 0,
            confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        )

class IsolationForestAnomalyDetector:
    """Isolation Forest for anomaly detection"""
    
    def __init__(self, contamination: float = 0.01):
        self.contamination = contamination
        self.model = None
        self.feature_names = []
    
    def train(self, X: np.ndarray, feature_names: List[str]):
        """Train Isolation Forest model"""
        from sklearn.ensemble import IsolationForest
        
        self.feature_names = feature_names
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
    
    def predict_anomaly_score(self, features: Dict[str, float]) -> float:
        """Predict anomaly score (0 = normal, 1 = anomaly)"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # Convert to 0-1 scale where 1 is anomaly
        prediction = self.model.predict(X)[0]
        score = self.model.score_samples(X)[0]
        
        # Normalize score to 0-1 range
        anomaly_score = 1 / (1 + np.exp(score))  # Sigmoid transformation
        
        return float(anomaly_score)
    
    def save_model(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']

class LSTMSequenceDetector:
    """LSTM-based sequence anomaly detection"""
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 64):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = None
    
    def build_model(self, input_dim: int):
        """Build LSTM model architecture"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("tensorflow is required for LSTMSequenceDetector")
        
        model = keras.Sequential([
            layers.LSTM(self.hidden_size, return_sequences=True, 
                       input_shape=(self.sequence_length, input_dim)),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_size // 2, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC'],
        )
        
        self.model = model
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 50, batch_size: int = 32):
        """Train LSTM model"""
        if self.model is None:
            self.build_model(X.shape[2])
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
        )
        
        return history
    
    def predict_sequence(self, sequence: np.ndarray) -> float:
        """Predict fraud probability for a sequence"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        fraud_prob = self.model.predict(sequence, verbose=0)[0, 0]
        return float(fraud_prob)
    
    def save_model(self, path: str):
        """Save model to disk"""
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)

class EnsembleFraudDetector:
    """Ensemble of multiple fraud detection models"""
    
    def __init__(self):
        self.xgboost_model = XGBoostFraudDetector()
        self.isolation_forest = IsolationForestAnomalyDetector()
        self.lstm_model = None  # Optional
        self.weights = {
            'xgboost': 0.5,
            'isolation_forest': 0.3,
            'lstm': 0.2,
        }
    
    def predict(self, features: Dict[str, float], 
                transaction_sequence: Optional[np.ndarray] = None) -> FraudPrediction:
        """Predict using ensemble of models"""
        model_scores = {}
        
        # XGBoost prediction
        xgb_prob, xgb_is_fraud = self.xgboost_model.predict(features)
        model_scores['xgboost'] = float(xgb_prob)
        
        # Isolation Forest anomaly score
        anomaly_score = self.isolation_forest.predict_anomaly_score(features)
        model_scores['isolation_forest'] = float(anomaly_score)
        
        # LSTM prediction (if available and sequence provided)
        if self.lstm_model and transaction_sequence is not None:
            lstm_prob = self.lstm_model.predict_sequence(transaction_sequence)
            model_scores['lstm'] = float(lstm_prob)
        else:
            model_scores['lstm'] = 0.5  # Neutral score
        
        # Calculate weighted ensemble score
        ensemble_score = (
            model_scores['xgboost'] * self.weights['xgboost'] +
            model_scores['isolation_forest'] * self.weights['isolation_forest'] +
            model_scores['lstm'] * self.weights['lstm']
        )
        
        # Determine if fraud
        is_fraud = ensemble_score >= 0.5
        
        # Calculate confidence based on model agreement
        scores = list(model_scores.values())
        confidence = 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0.5
        
        # Get feature importance
        feature_importance = self.xgboost_model.get_feature_importance()
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(features, model_scores)
        
        return FraudPrediction(
            transaction_id=features.get('transaction_id', 'unknown'),
            fraud_probability=float(ensemble_score),
            is_fraud=is_fraud,
            confidence=float(confidence),
            model_scores=model_scores,
            feature_importance=feature_importance,
            anomaly_score=float(anomaly_score),
            risk_factors=risk_factors,
            timestamp=datetime.now(),
        )
    
    def _identify_risk_factors(self, features: Dict[str, float], 
                               model_scores: Dict[str, float]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # High-risk amount
        if features.get('amount', 0) > features.get('user_avg_amount', 0) * 3:
            risk_factors.append('Unusually high transaction amount')
        
        # Biometric mismatch
        if features.get('biometric_match_score', 1.0) < 0.7:
            risk_factors.append('Biometric authentication mismatch')
        
        # New device
        if features.get('is_new_device', 0) == 1:
            risk_factors.append('Transaction from new device')
        
        # Suspicious location
        if features.get('distance_from_home', 0) > 1000:
            risk_factors.append('Transaction far from usual location')
        
        # High velocity
        if features.get('velocity_1h', 0) > 5:
            risk_factors.append('High transaction velocity')
        
        # VPN/Tor usage
        if features.get('is_vpn', 0) == 1 or features.get('is_tor', 0) == 1:
            risk_factors.append('VPN or Tor network detected')
        
        # Night transaction
        if features.get('is_night', 0) == 1:
            risk_factors.append('Transaction during unusual hours')
        
        # High anomaly score
        if model_scores.get('isolation_forest', 0) > 0.7:
            risk_factors.append('Anomalous transaction pattern detected')
        
        return risk_factors
    
    def save_models(self, base_path: str):
        """Save all models"""
        self.xgboost_model.save_model(f"{base_path}/xgboost_model.pkl")
        self.isolation_forest.save_model(f"{base_path}/isolation_forest_model.pkl")
        if self.lstm_model:
            self.lstm_model.save_model(f"{base_path}/lstm_model.h5")
    
    def load_models(self, base_path: str):
        """Load all models"""
        self.xgboost_model.load_model(f"{base_path}/xgboost_model.pkl")
        self.isolation_forest.load_model(f"{base_path}/isolation_forest_model.pkl")
        # LSTM loading is optional
