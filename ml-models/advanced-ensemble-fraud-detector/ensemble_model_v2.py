import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib
import logging
from datetime import datetime

class AdvancedEnsembleFraudDetectorV2:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('AdvancedEnsembleV2')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features from raw biometric and transaction data."""
        self.logger.info("Preparing features from input data")
        
        # Basic biometric features
        if 'fingerprint_hash' in data.columns:
            data['fingerprint_entropy'] = data['fingerprint_hash'].apply(lambda x: len(set(x)) / len(x) if x else 0)
        
        if 'facial_embedding' in data.columns:
            data['facial_variance'] = data['facial_embedding'].apply(lambda x: np.var(x) if isinstance(x, (list, np.ndarray)) else 0)
        
        # Transaction features
        if 'amount' in data.columns and 'time_delta' in data.columns:
            data['amount_per_time'] = data['amount'] / (data['time_delta'] + 1e-8)
            data['log_amount'] = np.log1p(data['amount'])
        
        # Behavioral features
        if 'velocity' in data.columns and 'location' in data.columns:
            data['velocity_score'] = np.abs(data['velocity']) * data['location'].apply(lambda x: 1 if 'high_risk' in str(x) else 0)
        
        # Drop non-numeric or irrelevant columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if self.feature_names is None:
            self.feature_names = numeric_cols
        else:
            numeric_cols = [col for col in numeric_cols if col in self.feature_names]
        
        X = data[numeric_cols].fillna(0)
        return self.scaler.fit_transform(X) if len(X) > 0 else np.array([])
    
    def build_ensemble(self) -> VotingClassifier:
        """Build a soft-voting ensemble with diverse base models."""
        self.logger.info("Building advanced ensemble model")
        
        # Base classifiers with different strengths
        rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_depth=10, 
            min_samples_split=5,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        gb_clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=self.random_state
        )
        
        lr_clf = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'
        )
        
        # Advanced voting with weights (RF higher weight for non-linear patterns)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('gb', gb_clf),
                ('lr', lr_clf)
            ],
            voting='soft',
            weights=[0.4, 0.35, 0.25]
        )
        
        return self.ensemble_model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the ensemble model with cross-validation."""
        if self.ensemble_model is None:
            self.build_ensemble()
        
        self.logger.info("Training ensemble model")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        # Train
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_val)
        y_pred_proba = self.ensemble_model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, X, y, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        self.logger.info(f"Training metrics: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fraud probability with confidence scores."""
        if self.ensemble_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.ensemble_model.predict(X)
        probabilities = self.ensemble_model.predict_proba(X)[:, 1]
        
        # Add uncertainty estimation (simple variance across base models)
        base_preds = np.array([est.predict_proba(X)[:, 1] for est in self.ensemble_model.estimators_])
        uncertainty = np.std(base_preds, axis=0)
        
        self.logger.info(f"Predictions made for {len(X)} samples")
        return predictions, probabilities, uncertainty
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler."""
        model_data = {
            'ensemble': self.ensemble_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model."""
        model_data = joblib.load(filepath)
        self.ensemble_model = model_data['ensemble']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.logger.info(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 10000
    X_synth = np.random.randn(n_samples, 10)
    y_synth = (np.sum(X_synth[:, :3], axis=1) > 1.5).astype(int)  # Simple fraud simulation
    
    detector = AdvancedEnsembleFraudDetectorV2(n_estimators=200)
    X_processed = detector.prepare_features(pd.DataFrame(X_synth, columns=[f'feature_{i}' for i in range(10)]))
    
    metrics = detector.train(X_processed, y_synth)
    print("Training Metrics:", metrics)
    
    # Test prediction
    test_X = X_processed[:5]
    preds, probs, unc = detector.predict(test_X)
    print("Sample Predictions:", preds)
    print("Sample Probabilities:", probs)
    print("Sample Uncertainty:", unc)
    
    detector.save_model('ensemble_model_v2.pkl')
