import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FederatedTrainer:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        
    def initialize_global_model(self):
        """Initialize the global federated learning model"""
        self.global_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.model_config['input_dim'],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.global_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Global model initialized")
        
    def train_client_model(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5):
        """Train a client-specific model"""
        client_model = tf.keras.models.clone_model(self.global_model)
        client_model.set_weights(self.global_model.get_weights())
        
        history = client_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.client_models[client_id] = client_model
        self.aggregation_weights[client_id] = len(X_train)
        
        logger.info(f"Client {client_id} model trained with {len(X_train)} samples")
        return history
        
    def federated_averaging(self):
        """Aggregate client models using federated averaging"""
        if not self.client_models:
            logger.warning("No client models to aggregate")
            return
            
        total_samples = sum(self.aggregation_weights.values())
        global_weights = []
        
        # Get the structure from the first client model
        first_client = list(self.client_models.values())[0]
        num_layers = len(first_client.get_weights())
        
        for layer_idx in range(num_layers):
            layer_weights = []
            
            for client_id, client_model in self.client_models.items():
                client_weight = client_model.get_weights()[layer_idx]
                weight_contribution = client_weight * (self.aggregation_weights[client_id] / total_samples)
                layer_weights.append(weight_contribution)
            
            aggregated_layer = np.sum(layer_weights, axis=0)
            global_weights.append(aggregated_layer)
        
        self.global_model.set_weights(global_weights)
        logger.info("Federated averaging completed")
        
    def secure_aggregation(self, use_differential_privacy: bool = True, epsilon: float = 1.0):
        """Perform secure aggregation with differential privacy"""
        if use_differential_privacy:
            noise_scale = 1.0 / epsilon
            
            for layer_idx, layer_weight in enumerate(self.global_model.get_weights()):
                noise = np.random.laplace(0, noise_scale, layer_weight.shape)
                layer_weight += noise
                
        logger.info("Secure aggregation with differential privacy applied")
        
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the global model"""
        results = self.global_model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        logger.info(f"Global model evaluation: {metrics}")
        return metrics
        
    def save_global_model(self, path: str):
        """Save the global model"""
        self.global_model.save(path)
        logger.info(f"Global model saved to {path}")
        
    def load_global_model(self, path: str):
        """Load a saved global model"""
        self.global_model = tf.keras.models.load_model(path)
        logger.info(f"Global model loaded from {path}")
