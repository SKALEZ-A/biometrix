import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TransactionAnomalyDetector:
    """
    Isolation Forest-based anomaly detector for transaction fraud detection.
    """
    
    def __init__(self, contamination: float = 0.01, n_estimators: int = 200, max_samples: int = 256):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw to train each base estimator
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1,
            warm_start=False
        )
        
        self.scaler = RobustScaler()
        self.pca = None
        self.feature_names = []
        self.is_fitted = False
        
    def extract_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from transaction data.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Basic transaction features
        features['amount'] = transactions['amount']
        features['hour_of_day'] = pd.to_datetime(transactions['timestamp'], unit='ms').dt.hour
        features['day_of_week'] = pd.to_datetime(transactions['timestamp'], unit='ms').dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_squared'] = features['amount'] ** 2
        features['amount_sqrt'] = np.sqrt(features['amount'])
        
        # User-specific features (if available)
        if 'userId' in transactions.columns:
            user_stats = transactions.groupby('userId')['amount'].agg([
                ('user_avg_amount', 'mean'),
                ('user_std_amount', 'std'),
                ('user_max_amount', 'max'),
                ('user_min_amount', 'min'),
                ('user_transaction_count', 'count')
            ]).reset_index()
            
            features = features.merge(
                transactions[['userId']].merge(user_stats, on='userId', how='left'),
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Deviation from user's normal behavior
            features['amount_deviation_from_user_avg'] = (
                features['amount'] - features['user_avg_amount']
            ) / (features['user_std_amount'] + 1e-6)
            
        # Merchant-specific features (if available)
        if 'merchantId' in transactions.columns:
            merchant_stats = transactions.groupby('merchantId')['amount'].agg([
                ('merchant_avg_amount', 'mean'),
                ('merchant_std_amount', 'std'),
                ('merchant_transaction_count', 'count')
            ]).reset_index()
            
            features = features.merge(
                transactions[['merchantId']].merge(merchant_stats, on='merchantId', how='left'),
                left_index=True,
                right_index=True,
                how='left'
            )
            
        # Geolocation features (if available)
        if 'geolocation' in transactions.columns:
            features['latitude'] = transactions['geolocation'].apply(
                lambda x: x.get('latitude', 0) if isinstance(x, dict) else 0
            )
            features['longitude'] = transactions['geolocation'].apply(
                lambda x: x.get('longitude', 0) if isinstance(x, dict) else 0
            )
            
        # Device fingerprint features (if available)
        if 'deviceFingerprint' in transactions.columns:
            features['device_trust_score'] = transactions['deviceFingerprint'].apply(
                lambda x: x.get('trustScore', 0.5) if isinstance(x, dict) else 0.5
            )
            
        # Time-based features
        if 'timestamp' in transactions.columns:
            timestamps = pd.to_datetime(transactions['timestamp'], unit='ms')
            
            # Time since last transaction (per user)
            if 'userId' in transactions.columns:
                transactions_sorted = transactions.sort_values(['userId', 'timestamp'])
                time_diffs = transactions_sorted.groupby('userId')['timestamp'].diff()
                features['time_since_last_transaction'] = time_diffs.fillna(0) / 1000  # Convert to seconds
                
            # Transaction velocity (transactions per hour)
            features['hour_of_day'] = timestamps.dt.hour
            features['is_night_transaction'] = features['hour_of_day'].between(0, 6).astype(int)
            
        # Risk score features (if available)
        if 'riskScore' in transactions.columns:
            features['risk_score'] = transactions['riskScore']
            features['risk_score_squared'] = features['risk_score'] ** 2
            
        # Fill missing values
        features = features.fillna(0)
        
        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def fit(self, transactions: pd.DataFrame, use_pca: bool = False, n_components: int = 10):
        """
        Fit the anomaly detector on transaction data.
        
        Args:
            transactions: DataFrame with transaction data
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
        """
        # Extract features
        features = self.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if use_pca:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"Model fitted on {len(features)} transactions with {len(self.feature_names)} features")
        
    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in transactions.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        features = self.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores for transactions.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            Array of anomaly scores (lower scores indicate anomalies)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        features = self.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        # Convert to probabilities (0 to 1, where 1 is most anomalous)
        # Normalize scores to [0, 1] range
        min_score = scores.min()
        max_score = scores.max()
        probabilities = 1 - (scores - min_score) / (max_score - min_score + 1e-6)
        
        return probabilities
    
    def evaluate(self, transactions: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on labeled data.
        
        Args:
            transactions: DataFrame with transaction data
            labels: True labels (1 for fraud, 0 for normal)
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(transactions)
        probabilities = self.predict_proba(transactions)
        
        # Convert predictions to binary (1 for fraud, 0 for normal)
        binary_predictions = (predictions == -1).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(labels, binary_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc_score = roc_auc_score(labels, probabilities)
        except:
            auc_score = 0.0
        
        metrics = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_score': float(auc_score),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
        }
        
        return metrics
    
    def get_feature_importance(self, transactions: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """
        Calculate feature importance using permutation importance.
        
        Args:
            transactions: DataFrame with transaction data
            n_samples: Number of samples to use for importance calculation
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating feature importance")
        
        # Sample transactions if dataset is large
        if len(transactions) > n_samples:
            transactions = transactions.sample(n=n_samples, random_state=42)
        
        # Extract features
        features = self.extract_features(transactions)
        X_scaled = self.scaler.transform(features)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Get baseline scores
        baseline_scores = self.model.score_samples(X_scaled)
        baseline_mean = baseline_scores.mean()
        
        # Calculate importance for each feature
        importances = []
        
        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get scores with permuted feature
            permuted_scores = self.model.score_samples(X_permuted)
            permuted_mean = permuted_scores.mean()
            
            # Importance is the change in mean score
            importance = abs(baseline_mean - permuted_mean)
            importances.append(importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TransactionAnomalyDetector':
        """Load a model from disk."""
        model_data = joblib.load(filepath)
        
        detector = cls(
            contamination=model_data['contamination'],
            n_estimators=model_data['n_estimators'],
            max_samples=model_data['max_samples']
        )
        
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.feature_names = model_data['feature_names']
        detector.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
        return detector


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector for transaction fraud detection.
    """
    
    def __init__(self, encoding_dim: int = 32, hidden_layers: List[int] = [64, 32]):
        """
        Initialize the autoencoder anomaly detector.
        
        Args:
            encoding_dim: Dimension of the encoded representation
            hidden_layers: List of hidden layer sizes
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def build_model(self, input_dim: int):
        """Build the autoencoder model."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        x = encoder_input
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        decoded = layers.Dense(input_dim, activation='linear')(x)
        
        # Autoencoder model
        self.model = keras.Model(encoder_input, decoded)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
    def fit(self, transactions: pd.DataFrame, epochs: int = 50, batch_size: int = 256):
        """
        Fit the autoencoder on transaction data.
        
        Args:
            transactions: DataFrame with transaction data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Extract features (reuse from IsolationForest)
        detector = TransactionAnomalyDetector()
        features = detector.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Build model
        self.build_model(X_scaled.shape[1])
        
        # Train model
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate reconstruction errors on training data
        reconstructions = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Set threshold at 95th percentile
        self.threshold = np.percentile(reconstruction_errors, 95)
        self.is_fitted = True
        
        print(f"Autoencoder fitted. Threshold: {self.threshold:.6f}")
        
    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in transactions.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        detector = TransactionAnomalyDetector()
        features = detector.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Get reconstruction errors
        reconstructions = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Predict anomalies
        predictions = np.where(reconstruction_errors > self.threshold, -1, 1)
        
        return predictions
    
    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores for transactions.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            Array of anomaly scores (higher scores indicate anomalies)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        detector = TransactionAnomalyDetector()
        features = detector.extract_features(transactions)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Get reconstruction errors
        reconstructions = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Normalize to [0, 1]
        probabilities = reconstruction_errors / (self.threshold * 2)
        probabilities = np.clip(probabilities, 0, 1)
        
        return probabilities


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detectors for robust fraud detection.
    """
    
    def __init__(self):
        self.isolation_forest = TransactionAnomalyDetector(contamination=0.01)
        self.autoencoder = AutoencoderAnomalyDetector()
        self.is_fitted = False
        
    def fit(self, transactions: pd.DataFrame):
        """Fit all detectors."""
        print("Fitting Isolation Forest...")
        self.isolation_forest.fit(transactions)
        
        print("Fitting Autoencoder...")
        self.autoencoder.fit(transactions)
        
        self.is_fitted = True
        print("Ensemble fitted successfully")
        
    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores using ensemble.
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            Array of ensemble anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from both models
        if_scores = self.isolation_forest.predict_proba(transactions)
        ae_scores = self.autoencoder.predict_proba(transactions)
        
        # Ensemble: weighted average
        ensemble_scores = 0.6 * if_scores + 0.4 * ae_scores
        
        return ensemble_scores
    
    def predict(self, transactions: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        """
        Predict anomalies using ensemble.
        
        Args:
            transactions: DataFrame with transaction data
            threshold: Threshold for anomaly classification
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        scores = self.predict_proba(transactions)
        predictions = np.where(scores > threshold, -1, 1)
        return predictions
