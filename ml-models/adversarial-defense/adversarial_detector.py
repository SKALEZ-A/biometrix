import numpy as np
import tensorflow as tf
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class AdversarialDetector:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.detection_threshold = 0.5
        
    def detect_fgsm_attack(self, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1) -> Tuple[np.ndarray, float]:
        """Detect Fast Gradient Sign Method attacks"""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
        
        gradients = tape.gradient(loss, X_tensor)
        signed_gradients = tf.sign(gradients)
        
        adversarial_X = X_tensor + epsilon * signed_gradients
        adversarial_X = tf.clip_by_value(adversarial_X, 0, 1)
        
        original_predictions = self.model.predict(X, verbose=0)
        adversarial_predictions = self.model.predict(adversarial_X.numpy(), verbose=0)
        
        prediction_diff = np.abs(original_predictions - adversarial_predictions)
        attack_detected = np.mean(prediction_diff) > self.detection_threshold
        
        logger.info(f"FGSM attack detection: {attack_detected}, avg diff: {np.mean(prediction_diff)}")
        return adversarial_X.numpy(), np.mean(prediction_diff)
        
    def detect_pgd_attack(self, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1, 
                         alpha: float = 0.01, iterations: int = 40) -> Tuple[np.ndarray, float]:
        """Detect Projected Gradient Descent attacks"""
        X_adv = X.copy()
        
        for i in range(iterations):
            X_tensor = tf.convert_to_tensor(X_adv, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
                loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
            
            gradients = tape.gradient(loss, X_tensor)
            X_adv = X_adv + alpha * tf.sign(gradients).numpy()
            
            perturbation = np.clip(X_adv - X, -epsilon, epsilon)
            X_adv = np.clip(X + perturbation, 0, 1)
        
        original_predictions = self.model.predict(X, verbose=0)
        adversarial_predictions = self.model.predict(X_adv, verbose=0)
        
        prediction_diff = np.abs(original_predictions - adversarial_predictions)
        attack_detected = np.mean(prediction_diff) > self.detection_threshold
        
        logger.info(f"PGD attack detection: {attack_detected}, avg diff: {np.mean(prediction_diff)}")
        return X_adv, np.mean(prediction_diff)
        
    def adversarial_training(self, X_train: np.ndarray, y_train: np.ndarray, 
                           epochs: int = 10, epsilon: float = 0.1):
        """Train model with adversarial examples"""
        for epoch in range(epochs):
            # Generate adversarial examples
            adversarial_X, _ = self.detect_fgsm_attack(X_train, y_train, epsilon)
            
            # Combine original and adversarial examples
            X_combined = np.vstack([X_train, adversarial_X])
            y_combined = np.vstack([y_train, y_train])
            
            # Train on combined dataset
            history = self.model.fit(
                X_combined, y_combined,
                epochs=1,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            logger.info(f"Adversarial training epoch {epoch + 1}/{epochs} completed")
        
    def input_transformation_defense(self, X: np.ndarray) -> np.ndarray:
        """Apply input transformations to defend against attacks"""
        # Bit depth reduction
        X_transformed = np.round(X * 255) / 255
        
        # JPEG compression simulation
        X_transformed = self._simulate_jpeg_compression(X_transformed)
        
        # Random resizing and padding
        X_transformed = self._random_resize_pad(X_transformed)
        
        return X_transformed
        
    def _simulate_jpeg_compression(self, X: np.ndarray, quality: int = 75) -> np.ndarray:
        """Simulate JPEG compression"""
        # Simplified JPEG compression simulation
        noise = np.random.normal(0, 0.01, X.shape)
        return np.clip(X + noise, 0, 1)
        
    def _random_resize_pad(self, X: np.ndarray) -> np.ndarray:
        """Apply random resizing and padding"""
        # Simplified implementation
        return X
        
    def ensemble_defense(self, X: np.ndarray, models: List[tf.keras.Model]) -> np.ndarray:
        """Use ensemble of models for robust predictions"""
        predictions = []
        
        for model in models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
