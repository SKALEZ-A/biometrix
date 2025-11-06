"""
Behavioral LSTM Model for Fraud Detection
Uses LSTM networks to model sequential behavioral patterns
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime

class BehavioralLSTMModel:
    """LSTM model for behavioral biometric fraud detection"""
    
    def __init__(self, 
                 sequence_length: int = 50,
                 feature_dim: int = 20,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize Behavioral LSTM Model
        
        Args:
            sequence_length: Length of behavioral sequence
            feature_dim: Number of features per timestep
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """Build LSTM model architecture"""
        
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim), 
                           name='behavioral_sequence')
        
        # First LSTM layer with return sequences
        x = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            name='lstm_1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        # Second LSTM layer
        if len(self.lstm_units) > 1:
            x = layers.LSTM(
                self.lstm_units[1],
                return_sequences=False,
                name='lstm_2'
            )(x)
            x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
            x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        # Attention mechanism
        attention = layers.Dense(self.lstm_units[-1], activation='tanh', 
                               name='attention_dense')(x)
        attention = layers.Dense(1, activation='softmax', 
                               name='attention_weights')(attention)
        x = layers.multiply([x, attention], name='attention_output')
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_4')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='behavioral_lstm')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return model
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32,
             class_weight: Optional[Dict[int, float]] = None) -> keras.callbacks.History:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Class weights for imbalanced data
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'behavioral_lstm_best.h5',
                monitor='val_auc' if X_val is not None else 'auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Calculate class weights if not provided
        if class_weight is None and len(np.unique(y_train)) == 2:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            class_weight = {
                0: 1.0,
                1: neg_count / pos_count if pos_count > 0 else 1.0
            }
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict fraud probability for sequences
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            batch_size: Batch size for prediction
            
        Returns:
            Fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, results))
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        
        # Load configuration
        try:
            with open(filepath.replace('.h5', '_config.json'), 'r') as f:
                config = json.load(f)
                self.sequence_length = config['sequence_length']
                self.feature_dim = config['feature_dim']
                self.lstm_units = config['lstm_units']
                self.dropout_rate = config['dropout_rate']
                self.learning_rate = config['learning_rate']
        except FileNotFoundError:
            print("Warning: Configuration file not found")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built"
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


class BidirectionalBehavioralLSTM(BehavioralLSTMModel):
    """Bidirectional LSTM for behavioral analysis"""
    
    def build_model(self) -> Model:
        """Build Bidirectional LSTM model"""
        
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim),
                           name='behavioral_sequence')
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units[0], return_sequences=True),
            name='bi_lstm_1'
        )(inputs)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        if len(self.lstm_units) > 1:
            x = layers.Bidirectional(
                layers.LSTM(self.lstm_units[1], return_sequences=False),
                name='bi_lstm_2'
            )(x)
            x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
            x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_3')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='bidirectional_behavioral_lstm')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        return model


class AttentionBehavioralLSTM(BehavioralLSTMModel):
    """LSTM with attention mechanism for behavioral analysis"""
    
    def build_model(self) -> Model:
        """Build LSTM model with attention"""
        
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim),
                           name='behavioral_sequence')
        
        # LSTM layer
        lstm_out = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            name='lstm_1'
        )(inputs)
        lstm_out = layers.Dropout(self.dropout_rate, name='dropout_1')(lstm_out)
        
        # Attention mechanism
        attention_weights = layers.Dense(1, activation='tanh', name='attention_score')(lstm_out)
        attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_weights)
        
        # Apply attention
        context_vector = layers.multiply([lstm_out, attention_weights], name='context_vector')
        context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), 
                                      name='sum_context')(context_vector)
        
        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(context_vector)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_2')(x)
        
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='attention_behavioral_lstm')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        return model


class SequencePreprocessor:
    """Preprocess behavioral sequences for LSTM"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.scaler = None
    
    def create_sequences(self, 
                        data: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from time series data
        
        Args:
            data: Time series data (timesteps, features)
            labels: Labels for each timestep
            stride: Stride for sequence creation
            
        Returns:
            Sequences and corresponding labels
        """
        sequences = []
        sequence_labels = []
        
        for i in range(0, len(data) - self.sequence_length + 1, stride):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Use label of last timestep in sequence
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        X = np.array(sequences)
        y = np.array(sequence_labels) if labels is not None else None
        
        return X, y
    
    def normalize_sequences(self, X: np.ndarray, 
                          fit: bool = True) -> np.ndarray:
        """
        Normalize sequences using z-score normalization
        
        Args:
            X: Sequences (samples, sequence_length, features)
            fit: Whether to fit scaler on data
            
        Returns:
            Normalized sequences
        """
        from sklearn.preprocessing import StandardScaler
        
        original_shape = X.shape
        
        # Reshape to 2D for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to 3D
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def augment_sequences(self, X: np.ndarray, y: np.ndarray,
                         noise_level: float = 0.01,
                         augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment sequences with noise for data augmentation
        
        Args:
            X: Sequences
            y: Labels
            noise_level: Standard deviation of Gaussian noise
            augmentation_factor: Number of augmented copies per sequence
            
        Returns:
            Augmented sequences and labels
        """
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor - 1):
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise
            X_augmented.append(X_noisy)
            y_augmented.append(y)
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_final))
        X_final = X_final[indices]
        y_final = y_final[indices]
        
        return X_final, y_final


class BehavioralFeatureExtractor:
    """Extract features from raw behavioral events"""
    
    @staticmethod
    def extract_keystroke_features(events: List[Dict]) -> np.ndarray:
        """Extract features from keystroke events"""
        features = []
        
        for event in events:
            feature_vector = [
                event.get('dwellTime', 0) / 1000.0,  # Normalize to seconds
                event.get('flightTime', 0) / 1000.0,
                event.get('pressure', 0.5),
                event.get('keyCode', 0) / 255.0,  # Normalize key code
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    @staticmethod
    def extract_mouse_features(events: List[Dict]) -> np.ndarray:
        """Extract features from mouse movement events"""
        features = []
        
        for event in events:
            feature_vector = [
                event.get('x', 0) / 1920.0,  # Normalize to screen width
                event.get('y', 0) / 1080.0,  # Normalize to screen height
                event.get('velocity', 0) / 1000.0,
                event.get('acceleration', 0) / 1000.0,
                event.get('curvature', 0),
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    @staticmethod
    def extract_combined_features(keystroke_events: List[Dict],
                                 mouse_events: List[Dict],
                                 max_length: int = 50) -> np.ndarray:
        """Extract and combine features from multiple event types"""
        keystroke_features = BehavioralFeatureExtractor.extract_keystroke_features(
            keystroke_events
        )
        mouse_features = BehavioralFeatureExtractor.extract_mouse_features(
            mouse_events
        )
        
        # Pad or truncate to max_length
        def pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
            if len(arr) > length:
                return arr[:length]
            elif len(arr) < length:
                padding = np.zeros((length - len(arr), arr.shape[1]))
                return np.vstack([arr, padding])
            return arr
        
        keystroke_features = pad_or_truncate(keystroke_features, max_length)
        mouse_features = pad_or_truncate(mouse_features, max_length)
        
        # Combine features
        combined = np.hstack([keystroke_features, mouse_features])
        
        return combined
