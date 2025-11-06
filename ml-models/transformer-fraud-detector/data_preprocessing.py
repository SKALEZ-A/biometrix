import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from datetime import datetime, timedelta

class TransactionDataset(Dataset):
    """
    PyTorch Dataset for transaction fraud detection
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 sequence_length: int = 50,
                 feature_columns: List[str] = None,
                 target_column: str = 'is_fraud',
                 transform=None):
        """
        Initialize dataset
        
        Args:
            data: DataFrame containing transaction data
            sequence_length: Length of transaction sequences
            feature_columns: List of feature column names
            target_column: Name of target column
            transform: Optional transform to apply
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.transform = transform
        
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        self.sequences = []
        self.labels = []
        self._prepare_sequences()
        
    def _prepare_sequences(self):
        """Prepare sequences from transaction data"""
        if 'user_id' in self.data.columns:
            for user_id in self.data['user_id'].unique():
                user_data = self.data[self.data['user_id'] == user_id].sort_values('timestamp')
                
                if len(user_data) < self.sequence_length:
                    continue
                
                features = user_data[self.feature_columns].values
                labels = user_data[self.target_column].values
                
                for i in range(len(features) - self.sequence_length + 1):
                    sequence = features[i:i + self.sequence_length]
                    label = labels[i + self.sequence_length - 1]
                    
                    self.sequences.append(sequence)
                    self.labels.append(label)
        else:
            features = self.data[self.feature_columns].values
            labels = self.data[self.target_column].values
            
            for i in range(len(features) - self.sequence_length + 1):
                sequence = features[i:i + self.sequence_length]
                label = labels[i + self.sequence_length - 1]
                
                self.sequences.append(sequence)
                self.labels.append(label)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


class FraudDataPreprocessor:
    """
    Comprehensive data preprocessing for fraud detection
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'FraudDataPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            data: Training DataFrame
            
        Returns:
            self
        """
        print("Fitting preprocessor...")
        
        self.feature_stats['n_samples'] = len(data)
        self.feature_stats['n_features'] = len(data.columns)
        self.feature_stats['fraud_rate'] = data['is_fraud'].mean() if 'is_fraud' in data.columns else 0
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_fraud' in numeric_columns:
            numeric_columns.remove('is_fraud')
        
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in numeric_columns:
            scaler = StandardScaler()
            scaler.fit(data[[col]].fillna(0))
            self.scalers[col] = scaler
            
            self.feature_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'missing_rate': data[col].isnull().mean()
            }
        
        for col in categorical_columns:
            encoder = LabelEncoder()
            encoder.fit(data[col].fillna('missing'))
            self.encoders[col] = encoder
            
            self.feature_stats[col] = {
                'n_unique': data[col].nunique(),
                'top_values': data[col].value_counts().head(10).to_dict(),
                'missing_rate': data[col].isnull().mean()
            }
        
        self.is_fitted = True
        print("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in data.columns:
                data[col] = scaler.transform(data[[col]].fillna(0))
        
        for col, encoder in self.encoders.items():
            if col in data.columns:
                data[col] = data[col].fillna('missing')
                data[col] = data[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
                data[col] = encoder.transform(data[col])
        
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data
        
        Args:
            data: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(data).transform(data)
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_stats': self.feature_stats,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FraudDataPreprocessor':
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scalers = data['scalers']
        preprocessor.encoders = data['encoders']
        preprocessor.feature_stats = data['feature_stats']
        preprocessor.is_fitted = data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


class FeatureEngineer:
    """
    Feature engineering for fraud detection
    """
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    @staticmethod
    def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features"""
        df = df.copy()
        
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_sqrt'] = np.sqrt(df['amount'])
            df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
            df['is_very_round'] = (df['amount'] % 100 == 0).astype(int)
            
            df['amount_decimal'] = df['amount'] % 1
            df['amount_integer'] = df['amount'].astype(int)
        
        return df
    
    @staticmethod
    def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based features"""
        df = df.copy()
        
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
            
            df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
            df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
            
            for window in [3600, 21600, 86400]:
                df[f'txn_count_{window}s'] = df.groupby('user_id').rolling(
                    window=f'{window}s', on='timestamp'
                ).size().reset_index(level=0, drop=True)
                
                if 'amount' in df.columns:
                    df[f'amount_sum_{window}s'] = df.groupby('user_id')['amount'].rolling(
                        window=window
                    ).sum().reset_index(level=0, drop=True)
        
        return df
    
    @staticmethod
    def add_user_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add user-based features"""
        df = df.copy()
        
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id').agg({
                'amount': ['count', 'mean', 'std', 'min', 'max'],
                'timestamp': ['min', 'max']
            }).reset_index()
            
            user_stats.columns = ['user_id', 'user_txn_count', 'user_amount_mean',
                                 'user_amount_std', 'user_amount_min', 'user_amount_max',
                                 'user_first_txn', 'user_last_txn']
            
            df = df.merge(user_stats, on='user_id', how='left')
            
            if 'timestamp' in df.columns:
                df['account_age_days'] = (df['timestamp'] - df['user_first_txn']).dt.total_seconds() / 86400
            
            if 'amount' in df.columns:
                df['amount_deviation'] = np.abs(df['amount'] - df['user_amount_mean']) / (df['user_amount_std'] + 1)
        
        return df
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering"""
        df = FeatureEngineer.add_temporal_features(df)
        df = FeatureEngineer.add_amount_features(df)
        df = FeatureEngineer.add_velocity_features(df)
        df = FeatureEngineer.add_user_features(df)
        
        return df


def create_data_loaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    batch_size: int = 32,
    sequence_length: int = 50,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TransactionDataset(train_data, sequence_length=sequence_length)
    val_dataset = TransactionDataset(val_data, sequence_length=sequence_length)
    test_dataset = TransactionDataset(test_data, sequence_length=sequence_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def generate_synthetic_data(n_samples: int = 10000, fraud_ratio: float = 0.1) -> pd.DataFrame:
    """
    Generate synthetic transaction data for testing
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraudulent transactions
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    
    data = []
    n_users = int(n_samples / 10)
    
    for i in range(n_samples):
        user_id = f'user_{np.random.randint(0, n_users)}'
        is_fraud = np.random.random() < fraud_ratio
        
        if is_fraud:
            amount = np.random.lognormal(6, 1.5)
            hour = np.random.choice([0, 1, 2, 3, 22, 23])
        else:
            amount = np.random.lognormal(4, 1)
            hour = np.random.randint(6, 22)
        
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
        timestamp = timestamp.replace(hour=hour, minute=np.random.randint(0, 60))
        
        data.append({
            'user_id': user_id,
            'transaction_id': f'txn_{i}',
            'timestamp': timestamp,
            'amount': amount,
            'merchant_id': f'merchant_{np.random.randint(0, 1000)}',
            'category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'utilities']),
            'is_fraud': int(is_fraud)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['user_id', 'timestamp'])
    
    return df


def main():
    """Test preprocessing pipeline"""
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.1)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    
    print("\nEngineering features...")
    df = FeatureEngineer.engineer_all_features(df)
    print(f"Features after engineering: {df.shape[1]}")
    
    print("\nSplitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_fraud'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_fraud'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print("\nFitting preprocessor...")
    preprocessor = FraudDataPreprocessor()
    train_df_processed = preprocessor.fit_transform(train_df)
    val_df_processed = preprocessor.transform(val_df)
    test_df_processed = preprocessor.transform(test_df)
    
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df_processed,
        val_df_processed,
        test_df_processed,
        batch_size=32,
        sequence_length=10
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\nSample batch:")
    for sequences, labels in train_loader:
        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Fraud ratio in batch: {labels.mean():.2%}")
        break
    
    print("\nSaving preprocessor...")
    preprocessor.save('fraud_preprocessor.pkl')
    
    print("\nPreprocessing pipeline completed successfully!")


if __name__ == '__main__':
    main()
