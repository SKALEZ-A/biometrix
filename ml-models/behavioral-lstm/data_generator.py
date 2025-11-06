import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import Sequence
import json
import pickle

class BehavioralDataGenerator(Sequence):
    """
    Custom data generator for behavioral LSTM model
    Handles sequential behavioral data for fraud detection
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 sequence_length: int = 50,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Initialize the data generator
        
        Args:
            data: DataFrame containing behavioral sequences
            sequence_length: Length of each sequence
            batch_size: Batch size for training
            shuffle: Whether to shuffle data after each epoch
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        self.user_ids = data['user_id'].unique()
        self.n_users = len(self.user_ids)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.feature_columns = [
            'keystroke_duration', 'keystroke_interval', 'mouse_velocity',
            'mouse_acceleration', 'click_duration', 'scroll_speed',
            'typing_speed', 'error_rate', 'pause_duration', 'session_duration'
        ]
        
        self.sequences = []
        self.labels = []
        
        self._prepare_sequences()
        self.on_epoch_end()
        
    def _prepare_sequences(self):
        """Prepare sequences from raw data"""
        print("Preparing sequences...")
        
        for user_id in self.user_ids:
            user_data = self.data[self.data['user_id'] == user_id].sort_values('timestamp')
            
            if len(user_data) < self.sequence_length:
                continue
            
            features = user_data[self.feature_columns].values
            labels = user_data['is_fraud'].values
            
            # Create overlapping sequences
            for i in range(len(features) - self.sequence_length + 1):
                sequence = features[i:i + self.sequence_length]
                label = labels[i + self.sequence_length - 1]  # Label of last item
                
                self.sequences.append(sequence)
                self.labels.append(label)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        
        # Normalize features
        n_samples, n_timesteps, n_features = self.sequences.shape
        self.sequences = self.sequences.reshape(-1, n_features)
        self.sequences = self.scaler.fit_transform(self.sequences)
        self.sequences = self.sequences.reshape(n_samples, n_timesteps, n_features)
        
        print(f"Prepared {len(self.sequences)} sequences")
        
    def __len__(self) -> int:
        """Return number of batches per epoch"""
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data"""
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.sequences))
        
        batch_sequences = self.sequences[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        if self.augment:
            batch_sequences = self._augment_sequences(batch_sequences)
        
        return batch_sequences, batch_labels
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            indices = np.arange(len(self.sequences))
            np.random.shuffle(indices)
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
    
    def _augment_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Apply data augmentation to sequences"""
        augmented = sequences.copy()
        
        # Add random noise
        noise = np.random.normal(0, 0.01, sequences.shape)
        augmented += noise
        
        # Random time warping
        if np.random.random() > 0.5:
            augmented = self._time_warp(augmented)
        
        # Random magnitude scaling
        if np.random.random() > 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented *= scale_factor
        
        return augmented
    
    def _time_warp(self, sequences: np.ndarray) -> np.ndarray:
        """Apply time warping augmentation"""
        warped = sequences.copy()
        
        for i in range(len(warped)):
            # Random time indices for warping
            warp_points = np.sort(np.random.choice(
                self.sequence_length, 
                size=3, 
                replace=False
            ))
            
            # Create warping function
            original_indices = np.arange(self.sequence_length)
            warped_indices = np.interp(
                original_indices,
                warp_points,
                warp_points + np.random.uniform(-2, 2, size=3)
            )
            
            # Apply warping
            for j in range(sequences.shape[2]):
                warped[i, :, j] = np.interp(
                    warped_indices,
                    original_indices,
                    sequences[i, :, j]
                )
        
        return warped
    
    def save_preprocessor(self, filepath: str):
        """Save scaler and encoder"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> Dict:
        """Load saved preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


class RealTimeBehavioralBuffer:
    """
    Buffer for real-time behavioral data collection
    """
    
    def __init__(self, 
                 sequence_length: int = 50,
                 feature_columns: List[str] = None):
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or [
            'keystroke_duration', 'keystroke_interval', 'mouse_velocity',
            'mouse_acceleration', 'click_duration', 'scroll_speed',
            'typing_speed', 'error_rate', 'pause_duration', 'session_duration'
        ]
        
        self.buffers = {}  # user_id -> deque of events
        self.scaler = None
        
    def add_event(self, user_id: str, event_data: Dict):
        """Add a behavioral event to the buffer"""
        if user_id not in self.buffers:
            self.buffers[user_id] = []
        
        # Extract features
        features = [event_data.get(col, 0.0) for col in self.feature_columns]
        self.buffers[user_id].append(features)
        
        # Keep only last sequence_length events
        if len(self.buffers[user_id]) > self.sequence_length:
            self.buffers[user_id].pop(0)
    
    def get_sequence(self, user_id: str) -> Optional[np.ndarray]:
        """Get current sequence for a user"""
        if user_id not in self.buffers:
            return None
        
        if len(self.buffers[user_id]) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = [[0.0] * len(self.feature_columns)] * (
                self.sequence_length - len(self.buffers[user_id])
            )
            sequence = padding + self.buffers[user_id]
        else:
            sequence = self.buffers[user_id][-self.sequence_length:]
        
        sequence = np.array(sequence).reshape(1, self.sequence_length, -1)
        
        if self.scaler is not None:
            n_samples, n_timesteps, n_features = sequence.shape
            sequence = sequence.reshape(-1, n_features)
            sequence = self.scaler.transform(sequence)
            sequence = sequence.reshape(n_samples, n_timesteps, n_features)
        
        return sequence
    
    def set_scaler(self, scaler: StandardScaler):
        """Set the scaler for normalization"""
        self.scaler = scaler
    
    def clear_buffer(self, user_id: str):
        """Clear buffer for a specific user"""
        if user_id in self.buffers:
            del self.buffers[user_id]
    
    def get_buffer_size(self, user_id: str) -> int:
        """Get current buffer size for a user"""
        return len(self.buffers.get(user_id, []))


def create_synthetic_behavioral_data(n_users: int = 1000,
                                     n_events_per_user: int = 100,
                                     fraud_ratio: float = 0.1) -> pd.DataFrame:
    """
    Create synthetic behavioral data for testing
    """
    print(f"Generating synthetic data for {n_users} users...")
    
    data = []
    
    for user_id in range(n_users):
        is_fraudster = np.random.random() < fraud_ratio
        
        # Generate behavioral patterns
        if is_fraudster:
            # Fraudulent behavior patterns
            keystroke_duration_mean = np.random.uniform(150, 250)
            keystroke_interval_mean = np.random.uniform(100, 200)
            mouse_velocity_mean = np.random.uniform(300, 500)
            typing_speed_mean = np.random.uniform(40, 60)
            error_rate_mean = np.random.uniform(0.1, 0.3)
        else:
            # Normal behavior patterns
            keystroke_duration_mean = np.random.uniform(80, 120)
            keystroke_interval_mean = np.random.uniform(150, 250)
            mouse_velocity_mean = np.random.uniform(150, 300)
            typing_speed_mean = np.random.uniform(60, 80)
            error_rate_mean = np.random.uniform(0.01, 0.05)
        
        for event_id in range(n_events_per_user):
            event = {
                'user_id': f'user_{user_id}',
                'event_id': event_id,
                'timestamp': pd.Timestamp.now() + pd.Timedelta(seconds=event_id),
                'keystroke_duration': max(0, np.random.normal(keystroke_duration_mean, 20)),
                'keystroke_interval': max(0, np.random.normal(keystroke_interval_mean, 30)),
                'mouse_velocity': max(0, np.random.normal(mouse_velocity_mean, 50)),
                'mouse_acceleration': max(0, np.random.normal(100, 20)),
                'click_duration': max(0, np.random.normal(150, 30)),
                'scroll_speed': max(0, np.random.normal(200, 40)),
                'typing_speed': max(0, np.random.normal(typing_speed_mean, 10)),
                'error_rate': np.clip(np.random.normal(error_rate_mean, 0.02), 0, 1),
                'pause_duration': max(0, np.random.normal(500, 100)),
                'session_duration': max(0, np.random.normal(1800, 300)),
                'is_fraud': 1 if is_fraudster else 0
            }
            data.append(event)
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} events")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    
    return df


if __name__ == '__main__':
    # Generate synthetic data
    df = create_synthetic_behavioral_data(n_users=1000, n_events_per_user=100)
    
    # Create data generator
    generator = BehavioralDataGenerator(
        data=df,
        sequence_length=50,
        batch_size=32,
        shuffle=True,
        augment=True
    )
    
    # Test generator
    print(f"\nGenerator info:")
    print(f"Number of batches: {len(generator)}")
    print(f"Number of sequences: {len(generator.sequences)}")
    
    # Get a sample batch
    X_batch, y_batch = generator[0]
    print(f"\nSample batch:")
    print(f"X shape: {X_batch.shape}")
    print(f"y shape: {y_batch.shape}")
    print(f"Fraud ratio in batch: {y_batch.mean():.2%}")
    
    # Save preprocessor
    generator.save_preprocessor('behavioral_preprocessor.pkl')
    
    # Test real-time buffer
    buffer = RealTimeBehavioralBuffer(sequence_length=50)
    buffer.set_scaler(generator.scaler)
    
    # Simulate real-time events
    for i in range(60):
        event = {
            'keystroke_duration': np.random.uniform(80, 120),
            'keystroke_interval': np.random.uniform(150, 250),
            'mouse_velocity': np.random.uniform(150, 300),
            'mouse_acceleration': np.random.uniform(80, 120),
            'click_duration': np.random.uniform(120, 180),
            'scroll_speed': np.random.uniform(160, 240),
            'typing_speed': np.random.uniform(60, 80),
            'error_rate': np.random.uniform(0.01, 0.05),
            'pause_duration': np.random.uniform(400, 600),
            'session_duration': np.random.uniform(1500, 2100)
        }
        buffer.add_event('test_user', event)
    
    # Get sequence
    sequence = buffer.get_sequence('test_user')
    print(f"\nReal-time sequence shape: {sequence.shape}")
    
    print("\nData generator testing completed!")
