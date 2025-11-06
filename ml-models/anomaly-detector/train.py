from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

class AnomalyDetectorTrainer:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        
        feature_cols = [col for col in df.columns if col not in ['transaction_id', 'is_fraud']]
        X = df[feature_cols]
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df['is_fraud'] if 'is_fraud' in df.columns else None
    
    def train(self, X_train):
        self.model.fit(X_train)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        anomaly_scores = -scores
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        return predictions, anomaly_scores
    
    def evaluate(self, X_test, y_test):
        predictions, scores = self.predict(X_test)
        
        predictions_binary = (predictions == -1).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions_binary),
            'precision': precision_score(y_test, predictions_binary),
            'recall': recall_score(y_test, predictions_binary),
            'f1': f1_score(y_test, predictions_binary)
        }
        
        return metrics
    
    def save_model(self, path):
        joblib.dump(self.model, f'{path}.model')
        joblib.dump(self.scaler, f'{path}.scaler')
        
    def load_model(self, path):
        self.model = joblib.load(f'{path}.model')
        self.scaler = joblib.load(f'{path}.scaler')

if __name__ == '__main__':
    trainer = AnomalyDetectorTrainer(contamination=0.05)
    X_train, y_train = trainer.prepare_data('transaction_data.csv')
    trainer.train(X_train)
    trainer.save_model('anomaly_detector')
