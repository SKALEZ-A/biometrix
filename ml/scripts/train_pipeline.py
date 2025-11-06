import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fraud_detector import BiometricFraudDetector, generate_synthetic_data
from data.processors import DataProcessor  # Assume data utils
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path: str = "config/ml_config.json"):
        self.config = self.load_config(config_path)
        self.detector = BiometricFraudDetector(self.config['model_path'])
        self.data_processor = DataProcessor()
    
    def load_config(self, path: str) -> dict:
        """Load ML config (hyperparams, paths)."""
        default_config = {
            "n_samples": 10000,
            "test_size": 0.2,
            "contamination": 0.1,
            "model_path": "ml/models/fraud_model.pkl",
            "data_dir": "ml/data/processed",
            "cv_folds": 5
        }
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        logger.info(f"Loaded config: {default_config}")
        return default_config
    
    def generate_and_preprocess_data(self) -> tuple:
        """Generate synthetic data and preprocess."""
        logger.info("Generating synthetic biometric data...")
        embeddings, labels = generate_synthetic_data(self.config['n_samples'])
        
        # Simulate loading from files for realism
        df = pd.DataFrame({
            'embedding': [json.dumps(e) for e in embeddings],
            'label': labels,
            'timestamp': pd.date_range(start='2025-01-01', periods=len(embeddings), freq='H')
        })
        df.to_csv(f"{self.config['data_dir']}/training_data.csv", index=False)
        
        # Preprocess
        processed_embeddings = [json.loads(row['embedding']) for _, row in df.iterrows()]
        X, y = self.data_processor.preprocess(processed_embeddings, df['label'].tolist())
        
        logger.info(f"Data prepared: {len(X)} samples, {sum(y)} fraud cases")
        return X, y
    
    def train_with_validation(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train model with cross-validation."""
        logger.info("Starting training with CV...")
        
        # Cross-validation
        cv_scores = cross_val_score(self.detector.model, X, y, cv=self.config['cv_folds'], scoring='roc_auc')
        logger.info(f"CV AUC scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Full train
        split = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.detector.train(X_train.tolist(), y_train.tolist())  # Convert for internal use
        
        # Evaluate
        eval_results = self.detector.evaluate(X_test.tolist(), y_test.tolist())
        
        results = {
            "cv_auc": cv_scores.mean(),
            "test_auc": eval_results.get('auc', 0),
            "trained_at": datetime.now().isoformat(),
            "config": self.config
        }
        
        # Save results
        with open(f"ml/reports/training_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed")
        return results
    
    def run(self):
        """Full pipeline execution."""
        try:
            X, y = self.generate_and_preprocess_data()
            results = self.train_with_validation(X, y)
            logger.info(f"Pipeline success: {results}")
            return results
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

# Data processor stub (expandable)
class DataProcessor:
    def preprocess(self, embeddings: list, labels: list) -> tuple:
        from models.fraud_detector import BiometricFraudDetector
        detector = BiometricFraudDetector()
        return detector.prepare_data(embeddings, labels)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
    print("Training pipeline executed successfully.")
