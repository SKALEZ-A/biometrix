import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

class LSTMFraudForecaster:
    def __init__(self, sequence_length: int = 30, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.history = None
        
    def build_model(self, lstm_units: List[int] = [128, 64, 32]):
        model = keras.Sequential()
        
        model.add(layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.features)
        ))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        for units in lstm_units[1:]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(0.3))
            model.add(layers.BatchNormalization())
        
        model.add(layers.LSTM(32, return_sequences=False))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple:
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            if labels is not None:
                y.append(labels[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y) if labels is not None else None
        
        return X, y
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              epochs: int = 50, batch_size: int = 32):
        
        print(f"Training LSTM forecaster with {len(X_train)} samples...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train.values)
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            )
        ]
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)
        
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("Training complete!")
        
        return self.history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        full_predictions = np.zeros(len(X))
        full_predictions[:self.sequence_length] = np.nan
        full_predictions[self.sequence_length:] = predictions.flatten()
        
        return full_predictions
    
    def predict_next_n_steps(self, X: pd.DataFrame, n_steps: int = 10) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        last_sequence = X_scaled[-self.sequence_length:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            current_input = current_sequence.reshape(1, self.sequence_length, self.features)
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            new_features = np.zeros(self.features)
            new_features[0] = next_pred
            
            current_sequence = np.vstack([current_sequence[1:], new_features])
        
        return np.array(predictions)
    
    def detect_anomalies(self, X: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        predictions = self.predict(X)
        
        valid_predictions = predictions[~np.isnan(predictions)]
        
        anomalies = valid_predictions > threshold
        anomaly_indices = np.where(anomalies)[0] + self.sequence_length
        
        anomaly_scores = valid_predictions[anomalies]
        
        return {
            'anomaly_count': int(np.sum(anomalies)),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_rate': float(np.mean(anomalies))
        }
    
    def get_attention_weights(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before getting attention weights")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        
        attention_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[0].output
        )
        
        attention_outputs = attention_model.predict(X_seq, verbose=0)
        
        attention_weights = np.mean(np.abs(attention_outputs), axis=2)
        
        return attention_weights
    
    def explain_prediction(self, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        if index < self.sequence_length:
            raise ValueError(f"Index must be >= {self.sequence_length}")
        
        sequence_start = index - self.sequence_length
        sequence_data = X.iloc[sequence_start:index]
        
        prediction = self.predict(X)[index]
        
        attention_weights = self.get_attention_weights(X)
        sequence_attention = attention_weights[sequence_start]
        
        feature_importance = np.mean(np.abs(sequence_data.values), axis=0)
        feature_importance = feature_importance / np.sum(feature_importance)
        
        top_features = []
        for i, (feature, importance) in enumerate(zip(X.columns, feature_importance)):
            top_features.append({
                'feature': feature,
                'importance': float(importance),
                'recent_value': float(X.iloc[index - 1, i])
            })
        
        top_features.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'prediction': float(prediction),
            'is_anomaly': bool(prediction > 0.5),
            'confidence': float(abs(prediction - 0.5) * 2),
            'top_features': top_features[:10],
            'temporal_attention': sequence_attention.tolist()
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        predictions = self.predict(X_test)
        
        valid_mask = ~np.isnan(predictions)
        predictions_valid = predictions[valid_mask]
        y_test_valid = y_test.values[valid_mask]
        
        y_pred = (predictions_valid > 0.5).astype(int)
        
        cm = confusion_matrix(y_test_valid, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_test_valid, y_pred)),
            'precision': float(precision_score(y_test_valid, y_pred)),
            'recall': float(recall_score(y_test_valid, y_pred)),
            'f1_score': float(f1_score(y_test_valid, y_pred)),
            'roc_auc': float(roc_auc_score(y_test_valid, predictions_valid)),
            'pr_auc': float(average_precision_score(y_test_valid, predictions_valid)),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        self.model.save(f"{filepath}_model.h5")
        
        metadata = {
            'sequence_length': self.sequence_length,
            'features': self.features,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        self.sequence_length = metadata['sequence_length']
        self.features = metadata['features']
        self.scaler = metadata['scaler']
        self.is_trained = metadata['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        if self.history is None:
            raise ValueError("No training history available")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['loss', 'accuracy', 'auc', 'precision']
        titles = ['Loss', 'Accuracy', 'AUC', 'Precision']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            ax.plot(self.history.history[metric], label=f'Training {title}')
            if f'val_{metric}' in self.history.history:
                ax.plot(self.history.history[f'val_{metric}'], label=f'Validation {title}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(f'{title} over Epochs')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved to training_history.png")
    
    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet"
        
        from io import StringIO
        import sys
        
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

class TemporalFeatureEngineer:
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        features = df.copy()
        
        for col in columns:
            for lag in lags:
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return features
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        features = df.copy()
        
        for col in columns:
            for window in windows:
                features[f'{col}_rolling_mean_{window}'] = features[col].rolling(window=window).mean()
                features[f'{col}_rolling_std_{window}'] = features[col].rolling(window=window).std()
                features[f'{col}_rolling_min_{window}'] = features[col].rolling(window=window).min()
                features[f'{col}_rolling_max_{window}'] = features[col].rolling(window=window).max()
        
        return features
    
    @staticmethod
    def create_ewm_features(df: pd.DataFrame, columns: List[str], spans: List[int]) -> pd.DataFrame:
        features = df.copy()
        
        for col in columns:
            for span in spans:
                features[f'{col}_ewm_{span}'] = features[col].ewm(span=span).mean()
        
        return features
    
    @staticmethod
    def create_diff_features(df: pd.DataFrame, columns: List[str], periods: List[int]) -> pd.DataFrame:
        features = df.copy()
        
        for col in columns:
            for period in periods:
                features[f'{col}_diff_{period}'] = features[col].diff(periods=period)
        
        return features
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        features = df.copy()
        
        features['hour'] = pd.to_datetime(features[timestamp_col]).dt.hour
        features['day_of_week'] = pd.to_datetime(features[timestamp_col]).dt.dayofweek
        features['day_of_month'] = pd.to_datetime(features[timestamp_col]).dt.day
        features['month'] = pd.to_datetime(features[timestamp_col]).dt.month
        features['quarter'] = pd.to_datetime(features[timestamp_col]).dt.quarter
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_month_start'] = pd.to_datetime(features[timestamp_col]).dt.is_month_start.astype(int)
        features['is_month_end'] = pd.to_datetime(features[timestamp_col]).dt.is_month_end.astype(int)
        
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features

def train_lstm_forecaster(train_data: pd.DataFrame, target_col: str,
                          val_data: pd.DataFrame = None,
                          sequence_length: int = 30,
                          epochs: int = 50) -> LSTMFraudForecaster:
    
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    engineer = TemporalFeatureEngineer()
    X_train = engineer.create_lag_features(X_train, ['amount'], [1, 2, 3, 5, 7])
    X_train = engineer.create_rolling_features(X_train, ['amount'], [7, 14, 30])
    X_train = X_train.fillna(0)
    
    model = LSTMFraudForecaster(sequence_length=sequence_length, features=X_train.shape[1])
    
    X_val, y_val = None, None
    if val_data is not None:
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        X_val = engineer.create_lag_features(X_val, ['amount'], [1, 2, 3, 5, 7])
        X_val = engineer.create_rolling_features(X_val, ['amount'], [7, 14, 30])
        X_val = X_val.fillna(0)
    
    model.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    return model

if __name__ == "__main__":
    print("LSTM Fraud Forecaster initialized")
    print("Supports time-series fraud prediction with temporal attention")
