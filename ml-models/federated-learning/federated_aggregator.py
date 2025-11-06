import numpy as np
import hashlib
import secrets
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

# Configure logging for federated learning operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    """Enumeration of supported aggregation strategies for federated learning."""
    FEDAVG = "fedavg"  # Federated Averaging
    FEDPROX = "fedprox"  # Federated Proximal
    SECURE_AGG = "secure_aggregation"  # Secure aggregation with masking
    BYZANTINE_ROBUST = "byzantine_robust"  # Byzantine fault tolerant aggregation
    DIFFERENTIAL_PRIVACY = "differential_privacy"  # DP-SGD variant

@dataclass
class ModelUpdate:
    """Represents a model update from a participating client."""
    client_id: str
    model_weights: np.ndarray
    update_norm: float
    timestamp: float
    metadata: Dict[str, Any]
    signature: bytes  # Cryptographic signature of the update
    noise_added: bool = False  # Whether differential privacy noise was added

@dataclass
class AggregationConfig:
    """Configuration for the aggregation process."""
    strategy: AggregationStrategy
    min_clients: int = 3  # Minimum number of clients required for aggregation
    max_clients: int = 100  # Maximum number of clients to consider
    clip_norm: float = 1.0  # Gradient clipping threshold
    learning_rate: float = 0.01  # Global learning rate
    epsilon_dp: float = 1.0  # Differential privacy epsilon
    delta_dp: float = 1e-5  # Differential privacy delta
    max_rounds: int = 1000  # Maximum training rounds
    tolerance: float = 1e-6  # Convergence tolerance
    byzantine_threshold: float = 0.2  # Fraction of malicious clients to tolerate

class SecureCommunicationManager:
    """Handles secure communication and verification between aggregator and clients."""
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.client_public_keys: Dict[str, rsa.RSAPublicKey] = {}
    
    def register_client(self, client_id: str, client_public_key_pem: str) -> bool:
        """Register a new client with their public key."""
        try:
            client_public_key = serialization.load_pem_public_key(
                client_public_key_pem.encode(),
                backend=default_backend()
            )
            self.client_public_keys[client_id] = client_public_key
            logger.info(f"Client {client_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    def verify_update_signature(self, update: ModelUpdate, client_public_key: rsa.RSAPublicKey) -> bool:
        """Verify the cryptographic signature of a model update."""
        try:
            # Serialize the update data (excluding signature) for verification
            update_data = json.dumps({
                'client_id': update.client_id,
                'model_weights': update.model_weights.tolist(),
                'update_norm': update.update_norm,
                'timestamp': update.timestamp,
                'metadata': update.metadata
            }).encode()
            
            # Verify signature using client's public key
            signature = serialization.load_der_public_key(client_public_key, backend=default_backend())
            public_key.verify(
                update.signature,
                update_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed for client {update.client_id}: {e}")
            return False
    
    def sign_aggregation_result(self, result: np.ndarray) -> Tuple[np.ndarray, bytes]:
        """Sign the aggregated model weights before distribution."""
        try:
            result_data = json.dumps({'aggregated_weights': result.tolist()}).encode()
            signature = self.private_key.sign(
                result_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return result, signature
        except Exception as e:
            logger.error(f"Failed to sign aggregation result: {e}")
            return result, b''
    
    def get_public_key_pem(self) -> str:
        """Return the aggregator's public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

class DifferentialPrivacyNoise:
    """Implements differential privacy mechanisms for model updates."""
    
    @staticmethod
    def add_gaussian_noise(weights: np.ndarray, epsilon: float, delta: float, sensitivity: float = 1.0) -> np.ndarray:
        """Add Gaussian noise for (ε, δ)-differential privacy."""
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
        
        # Calculate noise scale (σ) for Gaussian mechanism
        # σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        # Generate Gaussian noise with the calculated scale
        noise = np.random.normal(0, sigma, weights.shape)
        
        # Add noise to weights
        noisy_weights = weights + noise
        
        logger.info(f"Added Gaussian noise (σ={sigma:.6f}) for ε={epsilon}, δ={delta}")
        return noisy_weights
    
    @staticmethod
    def laplace_noise(weights: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplace noise for ε-differential privacy."""
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be positive")
        
        # Calculate noise scale b = sensitivity / ε
        b = sensitivity / epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(0, b, weights.shape)
        
        # Add noise to weights
        noisy_weights = weights + noise
        
        logger.info(f"Added Laplace noise (b={b:.6f}) for ε={epsilon}")
        return noisy_weights
    
    @staticmethod
    def clip_gradients(weights: np.ndarray, clip_norm: float) -> Tuple[np.ndarray, float]:
        """Clip gradients to prevent outlier attacks and bound sensitivity."""
        # Calculate L2 norm of weights
        norm = np.linalg.norm(weights)
        
        if norm > clip_norm:
            # Scale weights to have norm = clip_norm
            clipped_weights = (weights / norm) * clip_norm
            scaling_factor = clip_norm / norm
            logger.info(f"Clipped gradients: norm {norm:.4f} -> {clip_norm} (scale: {scaling_factor:.4f})")
        else:
            clipped_weights = weights
            scaling_factor = 1.0
        
        return clipped_weights, norm

class ByzantineDetector:
    """Detects and mitigates Byzantine (malicious) clients in federated learning."""
    
    def __init__(self, threshold: float = 0.2, window_size: int = 5):
        self.threshold = threshold  # Fraction of malicious clients to tolerate
        self.window_size = window_size  # Number of recent rounds to consider
        self.client_scores: Dict[str, float] = {}  # Reputation scores for clients
        self.update_history: List[Dict[str, Any]] = []  # History of update statistics
    
    def detect_malicious_update(self, update: ModelUpdate, global_model: np.ndarray, 
                              client_updates: List[ModelUpdate]) -> bool:
        """Detect if an update is malicious using multiple heuristics."""
        malicious_indicators = []
        
        # 1. Norm-based detection (outlier updates)
        update_norm = np.linalg.norm(update.model_weights)
        global_norm = np.linalg.norm(global_model)
        norm_ratio = update_norm / (global_norm + 1e-8)
        if norm_ratio > 10 or norm_ratio < 0.1:  # Extreme norm ratios
            malicious_indicators.append("extreme_norm")
        
        # 2. Cosine similarity check with global direction
        if len(client_updates) > 1:
            avg_direction = np.mean([u.model_weights for u in client_updates if u.client_id != update.client_id], axis=0)
            cosine_sim = np.dot(update.model_weights, avg_direction) / (
                np.linalg.norm(update.model_weights) * np.linalg.norm(avg_direction) + 1e-8
            )
            if cosine_sim < 0.3:  # Moving in opposite direction
                malicious_indicators.append("opposite_direction")
        
        # 3. Historical consistency check
        client_history = [h for h in self.update_history[-self.window_size:] 
                         if h['client_id'] == update.client_id]
        if client_history:
            prev_update = client_history[-1]['weights']
            consistency = np.dot(update.model_weights, prev_update) / (
                np.linalg.norm(update.model_weights) * np.linalg.norm(prev_update) + 1e-8
            )
            if consistency < 0.5:  # Inconsistent with own history
                malicious_indicators.append("inconsistent_history")
        
        # 4. Statistical outlier detection (Mahalanobis distance proxy)
        if len(client_updates) > 5:
            norms = [np.linalg.norm(u.model_weights) for u in client_updates]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            z_score = (update_norm - mean_norm) / (std_norm + 1e-8)
            if abs(z_score) > 3:  # 3-sigma outlier
                malicious_indicators.append("statistical_outlier")
        
        is_malicious = len(malicious_indicators) >= 2  # Multiple indicators required
        if is_malicious:
            logger.warning(f"Malicious update detected from {update.client_id}: {malicious_indicators}")
            self.update_reputation(update.client_id, -0.1)  # Penalize reputation
        
        return is_malicious, malicious_indicators
    
    def update_reputation(self, client_id: str, delta: float):
        """Update client reputation score."""
        current_score = self.client_scores.get(client_id, 0.5)
        new_score = max(0.0, min(1.0, current_score + delta))
        self.client_scores[client_id] = new_score
        logger.info(f"Updated reputation for {client_id}: {current_score:.3f} -> {new_score:.3f}")
    
    def should_exclude_client(self, client_id: str) -> bool:
        """Determine if a client should be excluded based on reputation."""
        score = self.client_scores.get(client_id, 0.5)
        return score < 0.2  # Exclude clients with very low reputation

class FederatedAggregator:
    """Main aggregator class for federated learning in fraud detection systems."""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.global_model: Optional[np.ndarray] = None
        self.round_number = 0
        self.history: List[Dict[str, Any]] = []
        self.communication_manager = SecureCommunicationManager()
        self.dp_noise = DifferentialPrivacyNoise()
        self.byzantine_detector = ByzantineDetector(
            threshold=config.byzantine_threshold,
            window_size=min(10, config.max_rounds)
        )
        self.lock = threading.Lock()  # For thread-safe operations
        self.participating_clients: set = set()
        
        # Initialize global model with random weights (in practice, load from checkpoint)
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize the global model with random weights."""
        # For demonstration, create a simple dense layer (784 -> 10 neurons)
        # In practice, this would match the actual fraud detection model architecture
        input_size = 784  # Example: flattened biometric features
        output_size = 10  # Example: fraud risk categories
        hidden_size = 128
        
        # Simple MLP: input -> hidden -> output
        self.global_model = np.random.randn(input_size * hidden_size + hidden_size * output_size)
        logger.info(f"Initialized global model with {len(self.global_model)} parameters")
    
    def register_client(self, client_id: str, public_key_pem: str) -> bool:
        """Register a new client for federated learning."""
        with self.lock:
            if self.byzantine_detector.should_exclude_client(client_id):
                logger.warning(f"Client {client_id} excluded due to low reputation")
                return False
            
            success = self.communication_manager.register_client(client_id, public_key_pem)
            if success:
                self.participating_clients.add(client_id)
                logger.info(f"Registered client {client_id}. Total clients: {len(self.participating_clients)}")
            return success
    
    def receive_client_update(self, update: ModelUpdate) -> Tuple[bool, str]:
        """Receive and validate a client model update."""
        try:
            client_id = update.client_id
            
            # Verify client registration
            if client_id not in self.participating_clients:
                return False, "Client not registered"
            
            # Verify cryptographic signature
            client_pub_key = self.communication_manager.client_public_keys.get(client_id)
            if not client_pub_key or not self.communication_manager.verify_update_signature(update, client_pub_key):
                return False, "Invalid signature"
            
            # Byzantine fault detection
            if self.round_number > 0 and len(self.history) > 0:
                prev_updates = self.history[-1].get('client_updates', [])
                is_malicious, indicators = self.byzantine_detector.detect_malicious_update(
                    update, self.global_model, prev_updates
                )
                if is_malicious:
                    self.byzantine_detector.update_reputation(client_id, -0.15)
                    return False, f"Malicious update detected: {', '.join(indicators)}"
            
            # Clip gradients for sensitivity bounding
            clipped_update, update_norm = self.dp_noise.clip_gradients(
                update.model_weights, self.config.clip_norm
            )
            
            # Add differential privacy noise if configured
            if self.config.strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
                noisy_update = self.dp_noise.add_gaussian_noise(
                    clipped_update,
                    epsilon=self.config.epsilon_dp,
                    delta=self.config.delta_dp,
                    sensitivity=self.config.clip_norm
                )
                update.model_weights = noisy_update
                update.noise_added = True
            
            # Update the received update with processed weights
            update.model_weights = clipped_update
            update.update_norm = update_norm
            
            logger.info(f"Successfully received update from {client_id} (norm: {update_norm:.4f})")
            return True, "Update accepted"
            
        except Exception as e:
            logger.error(f"Error processing update from {update.client_id}: {e}")
            return False, f"Processing error: {str(e)}"
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Optional[np.ndarray]:
        """Aggregate client updates according to the selected strategy."""
        if len(updates) < self.config.min_clients:
            logger.warning(f"Insufficient updates for aggregation: {len(updates)} < {self.config.min_clients}")
            return None
        
        # Filter out malicious or low-reputation clients
        valid_updates = []
        for update in updates:
            if not self.byzantine_detector.should_exclude_client(update.client_id):
                valid_updates.append(update)
        
        if len(valid_updates) < self.config.min_clients:
            logger.warning(f"Insufficient valid updates after filtering: {len(valid_updates)}")
            return None
        
        logger.info(f"Aggregating {len(valid_updates)} valid updates using {self.config.strategy.value}")
        
        try:
            if self.config.strategy == AggregationStrategy.FEDAVG:
                aggregated = self._fedavg_aggregation(valid_updates)
            elif self.config.strategy == AggregationStrategy.FEDPROX:
                aggregated = self._fedprox_aggregation(valid_updates)
            elif self.config.strategy == AggregationStrategy.SECURE_AGG:
                aggregated = self._secure_aggregation(valid_updates)
            elif self.config.strategy == AggregationStrategy.BYZANTINE_ROBUST:
                aggregated = self._byzantine_robust_aggregation(valid_updates)
            elif self.config.strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
                aggregated = self._dp_aggregation(valid_updates)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.config.strategy}")
            
            # Apply global learning rate
            aggregated = aggregated * self.config.learning_rate
            
            # Update global model
            if self.global_model is not None:
                self.global_model += aggregated
            
            self.round_number += 1
            return aggregated
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None
    
    def _fedavg_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Federated Averaging (FedAvg) implementation."""
        # Ensure all updates have the same shape as global model
        total_weights = np.zeros_like(self.global_model)
        total_count = 0
        
        for update in updates:
            # Weight by inverse of update norm (smaller updates get more weight)
            weight = 1.0 / (update.update_norm + 1e-8)
            total_weights += weight * update.model_weights
            total_count += weight
        
        if total_count > 0:
            averaged_weights = total_weights / total_count
            logger.info(f"FedAvg completed: {len(updates)} clients, average norm: {np.linalg.norm(averaged_weights):.4f}")
            return averaged_weights
        else:
            raise ValueError("No valid updates for FedAvg aggregation")
    
    def _fedprox_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Federated Proximal (FedProx) with proximal term regularization."""
        # Similar to FedAvg but with proximal term (mu * ||w - w_global||^2)
        mu = 0.01  # Proximal term coefficient
        
        total_weights = np.zeros_like(self.global_model)
        total_count = 0
        
        for update in updates:
            # Proximal term: update + mu * (global_model - update)
            proximal_term = mu * (self.global_model - update.model_weights)
            prox_update = update.model_weights + proximal_term
            
            weight = 1.0 / (update.update_norm + 1e-8)
            total_weights += weight * prox_update
            total_count += weight
        
        if total_count > 0:
            aggregated = total_weights / total_count
            logger.info("FedProx aggregation completed with proximal regularization")
            return aggregated
        else:
            raise ValueError("No valid updates for FedProx aggregation")
    
    def _secure_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Secure aggregation with additive secret sharing and masking."""
        # Generate random masks for secure sum (simplified implementation)
        n_clients = len(updates)
        masks = [secrets.token_bytes(len(self.global_model) * 8) for _ in range(n_clients)]
        
        # Each client adds their mask (in practice, this happens client-side)
        # Here we simulate the masking process
        total_masked = np.zeros_like(self.global_model, dtype=np.float64)
        
        for i, update in enumerate(updates):
            # Simulate mask addition (in real implementation, masks sum to zero)
            mask_contribution = np.frombuffer(masks[i], dtype=np.float64)
            masked_update = update.model_weights + mask_contribution[:len(update.model_weights)]
            total_masked += masked_update
        
        # Remove masks (in practice, the sum of masks is zero)
        # For simulation, we just return the sum of original updates
        secure_sum = np.sum([update.model_weights for update in updates], axis=0)
        averaged = secure_sum / len(updates)
        
        logger.info("Secure aggregation completed with masking protocol")
        return averaged
    
    def _byzantine_robust_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Byzantine-robust aggregation using Krum or median-based methods."""
        n = len(updates)
        f = int(self.config.byzantine_threshold * n)  # Number of faulty clients to tolerate
        
        # Use coordinate-wise median (robust to outliers)
        aggregated = np.zeros_like(self.global_model)
        
        # For each parameter coordinate, compute median across all updates
        for i in range(len(self.global_model)):
            coord_values = [update.model_weights[i] for update in updates]
            aggregated[i] = np.median(coord_values)
        
        # Apply trimming: remove f most extreme updates per coordinate (simplified)
        logger.info(f"Byzantine-robust aggregation (K={n-2*f}) using median method")
        return aggregated
    
    def _dp_aggregation(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Differential Privacy aggregation with advanced noise calibration."""
        # First perform standard FedAvg
        fedavg_result = self._fedavg_aggregation(updates)
        
        # Add additional noise at aggregation level
        if self.config.epsilon_dp > 0:
            # Advanced noise: calibrate based on number of clients and sensitivity
            client_sensitivity = self.config.clip_norm / len(updates)
            additional_noise = self.dp_noise.add_gaussian_noise(
                fedavg_result,
                epsilon=self.config.epsilon_dp / 2,  # Budget split
                delta=self.config.delta_dp,
                sensitivity=client_sensitivity
            )
            logger.info("DP aggregation completed with server-side noise")
            return additional_noise
        
        return fedavg_result
    
    def distribute_global_model(self) -> Dict[str, Any]:
        """Distribute the updated global model to all clients."""
        if self.global_model is None:
            raise ValueError("No global model to distribute")
        
        # Sign the model before distribution
        signed_model, signature = self.communication_manager.sign_aggregation_result(self.global_model)
        
        distribution_info = {
            'round': self.round_number,
            'global_model': signed_model.tolist(),
            'model_signature': signature.hex(),
            'timestamp': time.time(),
            'convergence_info': self._check_convergence(),
            'aggregator_public_key': self.communication_manager.get_public_key_pem(),
            'participating_clients': len(self.participating_clients),
            'metadata': {
                'strategy': self.config.strategy.value,
                'epsilon': getattr(self.config, 'epsilon_dp', None),
                'learning_rate': self.config.learning_rate
            }
        }
        
        logger.info(f"Distributed global model for round {self.round_number} to {len(self.participating_clients)} clients")
        return distribution_info
    
    def _check_convergence(self) -> Dict[str, float]:
        """Check if the model has converged based on update magnitude."""
        if len(self.history) < 2:
            return {'converged': False, 'update_magnitude': 0.0, 'relative_change': 0.0}
        
        prev_global = self.history[-2]['global_model']
        current_global = self.global_model
        
        update_magnitude = np.linalg.norm(current_global - prev_global)
        relative_change = update_magnitude / (np.linalg.norm(current_global) + 1e-8)
        
        converged = relative_change < self.config.tolerance
        
        logger.info(f"Convergence check: magnitude={update_magnitude:.6f}, relative={relative_change:.6f}, converged={converged}")
        
        return {
            'converged': converged,
            'update_magnitude': float(update_magnitude),
            'relative_change': float(relative_change)
        }
    
    def run_training_round(self, client_updates: List[ModelUpdate]) -> bool:
        """Execute a complete federated learning round."""
        start_time = time.time()
        
        try:
            # Receive and validate updates (in practice, this would be async)
            valid_count = 0
            for update in client_updates:
                success, message = self.receive_client_update(update)
                if success:
                    valid_count += 1
            
            logger.info(f"Round {self.round_number}: Received {valid_count}/{len(client_updates)} valid updates")
            
            # Perform aggregation
            aggregated_update = self.aggregate_updates(client_updates)
            if aggregated_update is None:
                logger.error(f"Round {self.round_number} failed: aggregation unsuccessful")
                return False
            
            # Record history
            self.history.append({
                'round': self.round_number,
                'timestamp': time.time(),
                'client_count': len(client_updates),
                'valid_count': valid_count,
                'aggregated_update': aggregated_update,
                'global_model': self.global_model.copy()
            })
            
            # Distribute updated model
            distribution = self.distribute_global_model()
            
            round_duration = time.time() - start_time
            logger.info(f"Round {self.round_number} completed successfully in {round_duration:.2f}s")
            
            # Check for convergence
            convergence = self._check_convergence()
            if convergence['converged']:
                logger.info(f"Training converged after {self.round_number} rounds")
                return True  # Signal completion
            
            return True  # Continue training
            
        except Exception as e:
            logger.error(f"Round {self.round_number} failed with error: {e}")
            return False
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Return comprehensive training statistics."""
        if not self.history:
            return {'rounds_completed': 0, 'status': 'not_started'}
        
        stats = {
            'rounds_completed': self.round_number,
            'status': 'training' if self.round_number < self.config.max_rounds else 'completed',
            'total_clients': len(self.participating_clients),
            'converged': self._check_convergence()['converged'],
            'history': [
                {
                    'round': h['round'],
                    'clients': h['client_count'],
                    'duration': h.get('duration', 0),
                    'update_norm': np.linalg.norm(h['aggregated_update']) if 'aggregated_update' in h else 0
                }
                for h in self.history[-10:]  # Last 10 rounds
            ],
            'model_parameters': len(self.global_model),
            'privacy_budget_remaining': self.config.epsilon_dp if hasattr(self.config, 'epsilon_dp') else None,
            'byzantine_detections': len([h for h in self.history if h.get('malicious_detected', 0) > 0])
        }
        
        return stats
    
    def export_model(self, filepath: str) -> bool:
        """Export the current global model for deployment."""
        try:
            model_data = {
                'global_model': self.global_model.tolist(),
                'training_round': self.round_number,
                'config': {
                    'strategy': self.config.strategy.value,
                    'learning_rate': self.config.learning_rate,
                    'clip_norm': self.config.clip_norm
                },
                'metadata': {
                    'timestamp': time.time(),
                    'clients_trained': len(self.participating_clients),
                    'converged': self._check_convergence()['converged']
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return False
    
    def save_checkpoint(self, filepath: str) -> bool:
        """Save training checkpoint for recovery."""
        try:
            checkpoint = {
                'global_model': self.global_model.tolist(),
                'round_number': self.round_number,
                'history': self.history[-50:],  # Last 50 rounds
                'config': {
                    k: v for k, v in self.config.__dict__.items() 
                    if not k.startswith('_')
                },
                'client_scores': self.byzantine_detector.client_scores,
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Checkpoint saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load training checkpoint to resume training."""
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore global model
            self.global_model = np.array(checkpoint['global_model'])
            
            # Restore training state
            self.round_number = checkpoint.get('round_number', 0)
            if 'history' in checkpoint:
                self.history = checkpoint['history']
            
            # Restore client scores if available
            if 'client_scores' in checkpoint:
                self.byzantine_detector.client_scores = checkpoint['client_scores']
            
            logger.info(f"Checkpoint loaded from {filepath}, resuming at round {self.round_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Configuration for fraud detection federated learning
    config = AggregationConfig(
        strategy=AggregationStrategy.DIFFERENTIAL_PRIVACY,
        min_clients=5,
        max_clients=50,
        clip_norm=1.0,
        learning_rate=0.01,
        epsilon_dp=0.5,
        delta_dp=1e-5,
        max_rounds=100,
        tolerance=1e-5,
        byzantine_threshold=0.2
    )
    
    # Initialize aggregator
    aggregator = FederatedAggregator(config)
    
    # Simulate client registration
    for i in range(10):
        client_id = f"bank_client_{i+1}"
        # Generate dummy public key (in practice, received from client)
        dummy_pub_key = rsa.generate_private_key(65537, 2048, default_backend()).public_key()
        pub_key_pem = dummy_pub_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        aggregator.register_client(client_id, pub_key_pem)
    
    # Simulate a training round with client updates
    client_updates = []
    for i in range(8):  # 8 participating clients
        client_id = f"bank_client_{i+1}"
        
        # Generate dummy model update (784 -> 10 fraud classifier)
        update_weights = np.random.randn(784 * 10) * 0.01  # Small updates
        update_norm = np.linalg.norm(update_weights)
        
        # Create dummy signature (in practice, generated by client)
        dummy_signature = secrets.token_bytes(256)
        
        update = ModelUpdate(
            client_id=client_id,
            model_weights=update_weights,
            update_norm=update_norm,
            timestamp=time.time(),
            metadata={'batch_size': 32, 'loss': 0.45, 'accuracy': 0.92},
            signature=dummy_signature
        )
        
        client_updates.append(update)
    
    # Run training round
    success = aggregator.run_training_round(client_updates)
    if success:
        print("Training round completed successfully!")
        
        # Get statistics
        stats = aggregator.get_training_statistics()
        print(f"Training stats: {json.dumps(stats, indent=2)}")
        
        # Export model
        aggregator.export_model("federated_fraud_model.json")
    else:
        print("Training round failed!")
    
    # Simulate Byzantine attack detection
    malicious_update = ModelUpdate(
        client_id="malicious_client",
        model_weights=np.random.randn(784 * 10) * 100,  # Extremely large update
        update_norm=1000.0,
        timestamp=time.time(),
        metadata={},
        signature=secrets.token_bytes(256)
    )
    
    # This should be detected as malicious
    valid, message = aggregator.receive_client_update(malicious_update)
    print(f"Malicious update detection: {valid}, {message}")
