import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import List, Dict, Tuple
import json
import os

logger = logging.getLogger(__name__)

class BiometricFraudDetector:
    def __init__(self, model_path: str = "ml/models/fraud_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model if exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                with open(self.model_path + '.scaler', 'rb') as f:
                    self.scaler = joblib.load(f)
                self.is_trained = True
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Model load failed: {e}")
    
    def prepare_data(self, embeddings: List[List[float]], labels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features from biometric embeddings (e.g., stats, distances)."""
        X = []
        for emb in embeddings:
            # Feature engineering: mean, std, min, max, etc. for anomaly detection
            features = [
                np.mean(emb), np.std(emb), np.min(emb), np.max(emb),
                np.median(emb), np.sum(np.abs(np.diff(emb)))  # Change magnitude
            ]
            # Pad/truncate to fixed size
            features += [0.0] * (10 - len(features))
            X.append(features[:10])
        
        X = np.array(X)
        if labels is not None:
            y = np.array(labels)
            return self.scaler.fit_transform(X), y
        return self.scaler.transform(X), None
    
    def train(self, embeddings: List[List[float]], labels: List[int]):
        """Train Isolation Forest for unsupervised anomaly detection (labels for eval)."""
        X, y = self.prepare_data(embeddings, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = IsolationForest(contamination=0.1, random_state=42)  # Assume 10% fraud
        self.model.fit(X_train)
        
        # Evaluate if labels provided
        if y_test is not None:
            preds = self.model.predict(X_test)
            auc = roc_auc_score(y_test, preds)
            logger.info(f"Model trained. AUC: {auc:.4f}")
            print(classification_report(y_test, preds))
        
        self.is_trained = True
        self.save_model()
    
    def predict(self, embedding: List[float]) -> Dict:
        """Predict fraud score for single embedding."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X, _ = self.prepare_data([embedding])
        anomaly_score = self.model.decision_function(X)[0]
        prediction = self.model.predict(X)[0]
        fraud_prob = 1 - anomaly_score  # Normalize to 0-1
        
        return {
            "fraud_detected": prediction == -1,
            "score": fraud_prob,
            "anomaly_score": anomaly_score,
            "features": X[0].tolist()
        }
    
    def save_model(self):
        """Save model and scaler."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.model_path + '.scaler')
        logger.info("Model saved")
    
    def evaluate(self, test_embeddings: List[List[float]], test_labels: List[int]):
        """Full evaluation with metrics."""
        X_test, y_test = self.prepare_data(test_embeddings, test_labels)
        preds = self.model.predict(X_test)
        scores = self.model.decision_function(X_test)
        
        report = {
            "auc": roc_auc_score(y_test, scores),
            "precision": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": np.confusion_matrix(y_test, preds).tolist()
        }
        logger.info(f"Evaluation: {report}")
        return report

# Example usage and synthetic data generation stub
def generate_synthetic_data(n_samples: int = 1000) -> Tuple[List[List[float]], List[int]]:
    """Generate mock biometric data for training (normal vs fraud)."""
    normal_emb = [np.random.normal(0, 0.1, 128).tolist() for _ in range(int(n_samples * 0.9))]
    fraud_emb = [np.random.normal(1, 0.5, 128).tolist() for _ in range(int(n_samples * 0.1))]
    embeddings = normal_emb + fraud_emb
    labels = [0] * len(normal_emb) + [1] * len(fraud_emb)
    return embeddings, labels

if __name__ == "__main__":
    detector = BiometricFraudDetector()
    embeddings, labels = generate_synthetic_data(5000)  # Large dataset for size
    detector.train(embeddings, labels)
    test_emb = np.random.normal(0, 0.1, 128).tolist()
    print(detector.predict(test_emb))
