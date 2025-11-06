import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.feature_stats = {}
        
    def engineer_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from transaction data"""
        df = df.copy()
        
        # Time-based features
        df = self._add_temporal_features(df)
        
        # Amount-based features
        df = self._add_amount_features(df)
        
        # Velocity features
        df = self._add_velocity_features(df)
        
        # Behavioral features
        df = self._add_behavioral_features(df)
        
        # Geographic features
        df = self._add_geographic_features(df)
        
        # Device features
        df = self._add_device_features(df)
        
        # Network features
        df = self._add_network_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['year'] = df['timestamp'].dt.year
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time of day categories
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                       (df['day_of_week'] < 5)).astype(int)
            
            # Time since epoch
            df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9
            
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add amount-based features"""
        if 'amount' in df.columns:
            # Log transformation
            df['amount_log'] = np.log1p(df['amount'])
            
            # Amount categories
            df['amount_category'] = pd.cut(df['amount'], 
                                          bins=[0, 10, 50, 100, 500, 1000, np.inf],
                                          labels=['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge'])
            
            # Round number detection
            df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
            df['is_very_round'] = (df['amount'] % 100 == 0).astype(int)
            
            # Decimal places
            df['decimal_places'] = df['amount'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based features"""
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
            
            # Transaction count in time windows
            for window in ['1H', '6H', '24H', '7D', '30D']:
                df[f'txn_count_{window}'] = df.groupby('user_id')['timestamp'].transform(
                    lambda x: x.rolling(window, on=x).count()
                )
            
            # Amount sum in time windows
            if 'amount' in df.columns:
                for window in ['1H', '6H', '24H', '7D', '30D']:
                    df[f'amount_sum_{window}'] = df.groupby('user_id')['amount'].transform(
                        lambda x: x.rolling(window).sum()
                    )
                    df[f'amount_mean_{window}'] = df.groupby('user_id')['amount'].transform(
                        lambda x: x.rolling(window).mean()
                    )
                    df[f'amount_std_{window}'] = df.groupby('user_id')['amount'].transform(
                        lambda x: x.rolling(window).std()
                    )
            
            # Time since last transaction
            df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
            df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
            
            # Transaction frequency
            df['txn_frequency'] = 1 / (df['time_since_last_txn'] + 1)
            
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral pattern features"""
        if 'user_id' in df.columns:
            # User transaction statistics
            user_stats = df.groupby('user_id').agg({
                'amount': ['count', 'mean', 'std', 'min', 'max', 'median'],
                'timestamp': ['min', 'max']
            }).reset_index()
            
            user_stats.columns = ['user_id', 'user_txn_count', 'user_amount_mean', 
                                 'user_amount_std', 'user_amount_min', 'user_amount_max',
                                 'user_amount_median', 'user_first_txn', 'user_last_txn']
            
            df = df.merge(user_stats, on='user_id', how='left')
            
            # Account age
            df['account_age_days'] = (df['timestamp'] - df['user_first_txn']).dt.total_seconds() / 86400
            
            # Deviation from user's normal behavior
            if 'amount' in df.columns:
                df['amount_deviation'] = np.abs(df['amount'] - df['user_amount_mean']) / (df['user_amount_std'] + 1)
                df['is_unusual_amount'] = (df['amount_deviation'] > 3).astype(int)
            
        return df
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic features"""
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from previous transaction
            if 'user_id' in df.columns:
                df = df.sort_values(['user_id', 'timestamp'])
                df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
                df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)
                
                df['distance_from_prev'] = self._haversine_distance(
                    df['latitude'], df['longitude'],
                    df['prev_lat'], df['prev_lon']
                )
                
                # Velocity (km/h)
                if 'time_since_last_txn' in df.columns:
                    df['travel_velocity'] = df['distance_from_prev'] / (df['time_since_last_txn'] / 3600 + 0.001)
                    df['is_impossible_travel'] = (df['travel_velocity'] > 800).astype(int)  # Faster than plane
        
        if 'country' in df.columns:
            # Country change detection
            if 'user_id' in df.columns:
                df['prev_country'] = df.groupby('user_id')['country'].shift(1)
                df['country_changed'] = (df['country'] != df['prev_country']).astype(int)
        
        return df
    
    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add device-based features"""
        if 'device_id' in df.columns:
            # Device usage statistics
            device_stats = df.groupby('device_id').agg({
                'user_id': 'nunique',
                'amount': ['count', 'mean']
            }).reset_index()
            
            device_stats.columns = ['device_id', 'device_user_count', 'device_txn_count', 'device_amount_mean']
            df = df.merge(device_stats, on='device_id', how='left')
            
            # Multiple users per device (suspicious)
            df['is_shared_device'] = (df['device_user_count'] > 1).astype(int)
            
        if 'user_agent' in df.columns:
            # Parse user agent
            df['is_mobile'] = df['user_agent'].str.contains('Mobile|Android|iPhone', case=False, na=False).astype(int)
            df['is_bot'] = df['user_agent'].str.contains('bot|crawler|spider', case=False, na=False).astype(int)
            
        return df
    
    def _add_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network-based features"""
        if 'ip_address' in df.columns:
            # IP usage statistics
            ip_stats = df.groupby('ip_address').agg({
                'user_id': 'nunique',
                'amount': ['count', 'sum']
            }).reset_index()
            
            ip_stats.columns = ['ip_address', 'ip_user_count', 'ip_txn_count', 'ip_amount_sum']
            df = df.merge(ip_stats, on='ip_address', how='left')
            
            # Multiple users from same IP (suspicious)
            df['is_shared_ip'] = (df['ip_user_count'] > 5).astype(int)
            
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        if 'amount' in df.columns:
            # Z-score
            df['amount_zscore'] = stats.zscore(df['amount'])
            
            # Percentile rank
            df['amount_percentile'] = df['amount'].rank(pct=True)
            
            # IQR-based outlier detection
            Q1 = df['amount'].quantile(0.25)
            Q3 = df['amount'].quantile(0.75)
            IQR = Q3 - Q1
            df['is_amount_outlier'] = ((df['amount'] < (Q1 - 1.5 * IQR)) | 
                                       (df['amount'] > (Q3 + 1.5 * IQR))).astype(int)
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between feature pairs"""
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Division (with safety)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
                
                # Addition
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                
                # Subtraction
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        return df
    
    def apply_pca(self, df: pd.DataFrame, 
                  feature_columns: List[str],
                  n_components: int = 10,
                  prefix: str = 'pca') -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        df = df.copy()
        
        X = df[feature_columns].fillna(0)
        
        if prefix not in self.pca_models:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            self.pca_models[prefix] = pca
        else:
            X_pca = self.pca_models[prefix].transform(X)
        
        # Add PCA components as features
        for i in range(n_components):
            df[f'{prefix}_component_{i}'] = X_pca[:, i]
        
        return df
    
    def scale_features(self, df: pd.DataFrame,
                      feature_columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if method not in self.scalers:
            df[feature_columns] = scaler.fit_transform(df[feature_columns].fillna(0))
            self.scalers[method] = scaler
        else:
            df[feature_columns] = self.scalers[method].transform(df[feature_columns].fillna(0))
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame,
                                   group_by: str,
                                   agg_columns: List[str],
                                   agg_functions: List[str]) -> pd.DataFrame:
        """Create aggregated features"""
        df = df.copy()
        
        agg_dict = {col: agg_functions for col in agg_columns}
        grouped = df.groupby(group_by).agg(agg_dict).reset_index()
        
        # Flatten column names
        grouped.columns = [f'{group_by}'] + [f'{col}_{func}' for col in agg_columns for func in agg_functions]
        
        df = df.merge(grouped, on=group_by, how='left')
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values"""
        df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                    categorical_columns: List[str],
                                    method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                if method == 'onehot':
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                elif method == 'label':
                    df[f'{col}_encoded'] = df[col].astype('category').cat.codes
                elif method == 'frequency':
                    freq = df[col].value_counts(normalize=True)
                    df[f'{col}_freq'] = df[col].map(freq)
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame,
                                      target_column: str,
                                      top_n: int = 50) -> pd.DataFrame:
        """Get feature importance ranking using correlation"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        correlations = []
        for col in numeric_columns:
            corr = abs(df[col].corr(df[target_column]))
            correlations.append({'feature': col, 'correlation': corr})
        
        importance_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        return importance_df.head(top_n)


def main():
    """Test feature engineering pipeline"""
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    
    df = pd.DataFrame({
        'user_id': [f'user_{i % 1000}' for i in range(n_samples)],
        'transaction_id': [f'txn_{i}' for i in range(n_samples)],
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
        'amount': np.random.lognormal(4, 1.5, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'device_id': [f'device_{i % 500}' for i in range(n_samples)],
        'ip_address': [f'192.168.{i % 255}.{i % 255}' for i in range(n_samples)],
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    print("Original data shape:", df.shape)
    print("\nOriginal columns:", df.columns.tolist())
    
    # Initialize feature engineer
    engineer = FraudFeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_transaction_features(df)
    
    print("\nEngineered data shape:", df_engineered.shape)
    print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
    
    # Get feature importance
    importance = engineer.get_feature_importance_ranking(df_engineered, 'is_fraud', top_n=20)
    print("\nTop 20 most important features:")
    print(importance)
    
    # Save engineered data
    df_engineered.to_csv('engineered_features.csv', index=False)
    print("\nEngineered features saved to 'engineered_features.csv'")

if __name__ == '__main__':
    main()
