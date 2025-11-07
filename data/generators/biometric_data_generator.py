import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import List, Dict, Any, Optional
from faker import Faker
import hashlib
from scipy import stats
import os
from pathlib import Path

class BiometricDataGenerator:
    def __init__(self, num_users: int = 10000, num_transactions_per_user: int = 100, output_dir: str = 'data/synthetic'):
        self.fake = Faker()
        self.num_users = num_users
        self.num_transactions_per_user = num_transactions_per_user
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # User profiles
        self.users = self._generate_users()
        
        # Fraud parameters
        self.fraud_rate = 0.05  # 5% fraud
        self.fraud_types = ['spoofing', 'velocity', 'geolocation', 'amount', 'device_swap']
        
        print(f"Initialized generator for {num_users} users with {num_transactions_per_user} tx/user")
    
    def _generate_users(self) -> List[Dict[str, Any]]:
        """Generate diverse user profiles."""
        users = []
        for i in range(self.num_users):
            user = {
                'user_id': f'user_{i:08d}',
                'name': self.fake.name(),
                'email': self.fake.email(),
                'age': random.randint(18, 80),
                'country': random.choice(['US', 'UK', 'DE', 'FR', 'IN', 'NG', 'CN', 'RU']),
                'tier': random.choice(['basic', 'premium', 'enterprise'], weights=[0.7, 0.2, 0.1]),
                'devices': [self._generate_device() for _ in range(random.randint(1, 3))],
                'biometrics': self._generate_biometrics(),
                'historical_risk': round(random.uniform(0.1, 0.8), 3)
            }
            users.append(user)
        return users
    
    def _generate_device(self) -> Dict[str, Any]:
        """Generate realistic device fingerprint."""
        os_types = ['Windows', 'macOS', 'iOS', 'Android', 'Linux']
        browsers = ['Chrome', 'Safari', 'Firefox', 'Edge']
        return {
            'device_id': hashlib.md5(str(random.random()).encode()).hexdigest()[:16],
            'os': random.choice(os_types),
            'browser': random.choice(browsers),
            'screen_resolution': f"{random.randint(1024, 2560)}x{random.randint(768, 1440)}",
            'user_agent': self.fake.user_agent(),
            'is_mobile': random.random() < 0.4
        }
    
    def _generate_biometrics(self) -> Dict[str, Any]:
        """Generate multi-modal biometric templates."""
        # Fingerprint (simplified hash + features)
        fp_template = hashlib.sha256(f"{random.random()}".encode()).hexdigest()
        fp_features = {
            'minutiae_count': random.randint(50, 150),
            'ridge_density': round(random.uniform(8, 15), 2),
            'quality_score': round(random.uniform(0.6, 1.0), 3),
            'liveness_score': round(random.uniform(0.7, 1.0), 3) if random.random() < 0.95 else round(random.uniform(0.0, 0.3), 3)  # Rare spoof
        }
        
        # Facial embedding (random vector)
        face_embedding = np.random.randn(512).tolist()  # 512-dim embedding
        face_features = {
            'embedding': face_embedding[:10] + ['...'],  # Truncate for storage
            'quality': round(random.uniform(0.5, 1.0), 3),
            'spoof_probability': round(random.uniform(0.0, 0.1), 3),
            'age_estimate': random.randint(18, 80),
            'emotion': random.choice(['neutral', 'happy', 'surprised'])
        }
        
        # Voice features
        voice_features = {
            'pitch_mean': round(random.uniform(100, 300), 2),
            'formant_f1': round(random.uniform(500, 800), 2),
            'formant_f2': round(random.uniform(1200, 2000), 2),
            'duration': round(random.uniform(2.0, 10.0), 2),
            'confidence': round(random.uniform(0.8, 1.0), 3)
        }
        
        return {
            'fingerprint': {**fp_features, 'template': fp_template},
            'face': face_features,
            'voice': voice_features,
            'iris': {'template': hashlib.sha256(f"{random.random()}".encode()).hexdigest(), 'quality': round(random.uniform(0.7, 1.0), 3)} if random.random() < 0.3 else None
        }
    
    def generate_transactions(self) -> pd.DataFrame:
        """Generate realistic transaction dataset with fraud injection."""
        transactions = []
        
        for user in self.users:
            base_time = datetime.now() - timedelta(days=random.randint(30, 365))
            for i in range(self.num_transactions_per_user):
                tx_time = base_time + timedelta(minutes=random.randint(0, 1440 * 30))  # Up to 30 days
                
                # Base transaction
                tx = {
                    'transaction_id': f'tx_{hashlib.md5(str(random.random()).encode()).hexdigest()[:16]}',
                    'user_id': user['user_id'],
                    'timestamp': tx_time.isoformat(),
                    'amount': round(random.uniform(1, 5000), 2) * (1.5 if user['tier'] == 'premium' else 1),
                    'currency': random.choice(['USD', 'EUR', 'GBP']),
                    'merchant': self.fake.company(),
                    'category': random.choice(['payment', 'transfer', 'purchase', 'withdrawal']),
                    'location': f"{self.fake.city()}, {user['country']}",
                    'ip_address': self.fake.ipv4(),
                    'device_id': random.choice(user['devices'])['device_id'],
                    'velocity': random.randint(1, 10),  # tx per minute
                    'session_duration': random.randint(30, 1800),  # seconds
                    'user_agent': random.choice(user['devices'])['user_agent']
                }
                
                # Inject fraud
                is_fraud = random.random() < self.fraud_rate
                if is_fraud:
                    fraud_type = random.choice(self.fraud_types)
                    if fraud_type == 'spoofing':
                        tx['biometric_spoofed'] = True
                        tx['liveness_score'] = round(random.uniform(0.0, 0.3), 3)
                    elif fraud_type == 'velocity':
                        tx['velocity'] = random.randint(20, 100)
                    elif fraud_type == 'geolocation':
                        tx['location'] = random.choice(['Lagos, NG', 'Moscow, RU', 'Beijing, CN'])
                        tx['ip_address'] = 'proxy_ip_' + str(random.randint(1, 100))
                    elif fraud_type == 'amount':
                        tx['amount'] *= random.uniform(5, 50)
                    elif fraud_type == 'device_swap':
                        tx['device_id'] = 'suspicious_' + str(random.randint(1, 1000))
                
                tx['is_fraud'] = is_fraud
                tx['fraud_type'] = fraud_type if is_fraud else None
                
                # Biometric verification for this tx
                bio = user['biometrics']
                tx['biometric_type'] = random.choice(['fingerprint', 'face', 'voice', 'multi'])
                if tx['biometric_type'] == 'fingerprint' and bio['fingerprint']:
                    tx['bio_confidence'] = bio['fingerprint']['quality_score'] * (0.3 if is_fraud and random.random() < 0.5 else 1)
                elif tx['biometric_type'] == 'face' and bio['face']:
                    tx['bio_confidence'] = bio['face']['quality'] * (0.4 if is_fraud and random.random() < 0.5 else 1)
                else:
                    tx['bio_confidence'] = round(random.uniform(0.6, 1.0), 3)
                
                # Risk indicators
                tx['risk_score'] = round(random.uniform(0.1, 0.9 if is_fraud else 0.6), 3)
                tx['challenge_required'] = tx['risk_score'] > 0.7 or (is_fraud and random.random() < 0.8)
                
                transactions.append(tx)
                
                # Update base_time for velocity simulation
                base_time = tx_time
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['amount_log'] = np.log1p(df['amount'])
        df['velocity_risk'] = (df['velocity'] > 10).astype(int)
        df['geo_risk'] = df['location'].str.contains('NG|RU|CN', case=False).astype(int)
        
        return df
    
    def generate_user_profiles_dataset(self) -> pd.DataFrame:
        """Generate user profiles CSV."""
        profiles = []
        for user in self.users:
            profile = {
                'user_id': user['user_id'],
                'name': user['name'],
                'email': user['email'],
                'age': user['age'],
                'country': user['country'],
                'tier': user['tier'],
                'num_devices': len(user['devices']),
                'historical_risk': user['historical_risk'],
                'biometric_types': ','.join([k for k, v in user['biometrics'].items() if v]),
                'avg_bio_quality': round(np.mean([
                    user['biometrics'].get('fingerprint', {}).get('quality_score', 0),
                    user['biometrics'].get('face', {}).get('quality', 0),
                    user['biometrics'].get('voice', {}).get('confidence', 0)
                ]), 3)
            }
            profiles.append(profile)
        
        df = pd.DataFrame(profiles)
        filepath = self.output_dir / 'user_profiles.csv'
        df.to_csv(filepath, index=False)
        print(f"Generated user profiles: {filepath} ({len(df)} rows)")
        return df
    
    def save_transactions(self, df: pd.DataFrame, format: str = 'csv') -> Path:
        """Save transactions to file."""
        filepath = self.output_dir / f'transactions_{datetime.now().strftime("%Y%m%d")}.{format}'
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', date_format='iso')
        elif format == 'parquet':
            df.to_parquet(filepath)
        
        print(f"Saved {len(df)} transactions to {filepath}")
        return filepath
    
    def generate_anomaly_dataset(self, df: pd.DataFrame, num_anomalies: int = 1000) -> pd.DataFrame:
        """Inject additional anomalies for unsupervised learning."""
        anomalies = []
        for _ in range(num_anomalies):
            anomaly = df.sample(1).iloc[0].to_dict()
            anomaly_type = random.choice(['extreme_amount', 'impossible_velocity', 'geo_jump', 'time_anomaly'])
            
            if anomaly_type == 'extreme_amount':
                anomaly['amount'] *= random.uniform(10, 100)
                anomaly['is_fraud'] = True
                anomaly['fraud_type'] = 'extreme_amount'
            elif anomaly_type == 'impossible_velocity':
                anomaly['velocity'] = random.randint(100, 1000)
                anomaly['is_fraud'] = True
                anomaly['fraud_type'] = 'velocity_spike'
            elif anomaly_type == 'geo_jump':
                anomaly['location'] = random.choice(['Antarctica', 'Space Station', 'Underwater'])
                anomaly['ip_address'] = 'anomalous_ip'
                anomaly['is_fraud'] = True
                anomaly['fraud_type'] = 'geo_impossible'
            elif anomaly_type == 'time_anomaly':
                anomaly['timestamp'] = datetime.now() + timedelta(days=random.randint(100, 10000))
                anomaly['is_fraud'] = True
                anomaly['fraud_type'] = 'temporal'
            
            anomalies.append(anomaly)
        
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])
        filepath = self.output_dir / 'anomalies.csv'
        anomaly_df.to_csv(filepath, index=False)
        print(f"Generated {len(anomaly_df)} anomalies: {filepath}")
        return anomaly_df
    
    def generate_fraud_patterns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract fraud patterns for analysis."""
        fraud_df = df[df['is_fraud'] == True]
        patterns = {
            'by_type': fraud_df.groupby('fraud_type').agg({
                'amount': ['mean', 'count'],
                'velocity': 'mean',
                'bio_confidence': 'mean'
            }).round(3),
            'by_country': fraud_df.groupby('country').size().to_frame('count'),
            'hourly': fraud_df.groupby('hour_of_day').size().to_frame('count'),
            'velocity_distribution': fraud_df['velocity'].describe()
        }
        
        # Save patterns
        with open(self.output_dir / 'fraud_patterns.json', 'w') as f:
            json.dump({k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in patterns.items()}, f, indent=2, default=str)
        
        print("Generated fraud patterns analysis")
        return patterns
    
    def run_full_generation(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run complete data generation pipeline."""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            self.fake.seed(seed)
        
        print("Starting full synthetic data generation...")
        
        # Generate transactions
        tx_df = self.generate_transactions()
        tx_path = self.save_transactions(tx_df, 'parquet')  # Efficient format
        
        # Save CSV version too
        self.save_transactions(tx_df, 'csv')
        
        # User profiles
        self.generate_user_profiles_dataset()
        
        # Anomalies
        anomalies_df = self.generate_anomaly_dataset(tx_df)
        
        # Fraud patterns
        patterns = self.generate_fraud_patterns(pd.concat([tx_df, anomalies_df]))
        
        # Split for ML (80/20 train/test)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(tx_df, test_size=0.2, random_state=42, stratify=tx_df['is_fraud'])
        train_df.to_parquet(self.output_dir / 'train.parquet', index=False)
        test_df.to_parquet(self.output_dir / 'test.parquet', index=False)
        
        stats = {
            'total_transactions': len(tx_df),
            'fraud_count': int(tx_df['is_fraud'].sum()),
            'fraud_rate': round(tx_df['is_fraud'].mean(), 4),
            'avg_amount': round(tx_df['amount'].mean(), 2),
            'countries': tx_df['country'].nunique(),
            'file_size_mb': sum(f.stat().st_size for f in self.output_dir.glob('*') if f.is_file()) / (1024**2)
        }
        
        print(f"Generation complete! Stats: {stats}")
        return stats

# Additional specialized generators
class AdvancedFraudScenarioGenerator(BiometricDataGenerator):
    """Extended generator for specific fraud scenarios like account takeover, synthetic identity."""
    
    def generate_account_takeover_scenarios(self, num_scenarios: int = 100) -> pd.DataFrame:
        scenarios = []
        for _ in range(num_scenarios):
            # Simulate ATO: new device + high value tx from unusual location
            scenario = self.generate_transactions().iloc[0].to_dict()
            scenario['fraud_type'] = 'account_takeover'
            scenario['device_id'] = 'stolen_device_' + str(random.randint(1, 1000))
            scenario['location'] = random.choice(['Unknown', 'VPN_' + self.fake.country()[:2]])
            scenario['amount'] *= random.uniform(2, 10)
            scenario['biometric_confidence'] = round(random.uniform(0.2, 0.5), 3)  # Failed biometric
            scenario['is_fraud'] = True
            scenarios.append(scenario)
        
        df = pd.DataFrame(scenarios)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        filepath = self.output_dir / 'account_takeover_scenarios.csv'
        df.to_csv(filepath, index=False)
        print(f"Generated {len(df)} ATO scenarios: {filepath}")
        return df
    
    def generate_synthetic_identity_fraud(self, num_identities: int = 50) -> List[Dict]:
        identities = []
        for i in range(num_identities):
            identity = {
                'synthetic_id': f'synth_{i:04d}',
                'fake_name': self.fake.name(),
                'fake_email': self.fake.email(),
                'fake_ssn': self.fake.ssn(),  # US-specific, adapt for other countries
                'fake_phone': self.fake.phone_number(),
                'created_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'associated_accounts': random.randint(1, 5),
                'total_fraud_amount': round(random.uniform(1000, 50000), 2),
                'detection_features': {
                    'email_domain_age_days': random.randint(1, 365),
                    'phone_area_code_mismatch': random.choice([True, False]),
                    'ip_geolocation_mismatch': random.choice([True, False]),
                    'device_fingerprint_collision': random.choice([True, False])
                }
            }
            identities.append(identity)
        
        with open(self.output_dir / 'synthetic_identities.json', 'w') as f:
            json.dump(identities, f, indent=2, default=str)
        
        print(f"Generated {len(identities)} synthetic identities")
        return identities

# Run example
if __name__ == "__main__":
    generator = BiometricDataGenerator(num_users=5000, num_transactions_per_user=200)
    stats = generator.run_full_generation(seed=42)
    
    # Advanced scenarios
    adv_gen = AdvancedFraudScenarioGenerator(num_users=1000, num_transactions_per_user=50)
    adv_gen.generate_account_takeover_scenarios(200)
    adv_gen.generate_synthetic_identity_fraud(100)
    
    # Verify output size
    total_size = sum(f.stat().st_size for f in Path('data/synthetic').glob('**/*') if f.is_file()) / (1024**2)
    print(f"Total generated data size: {total_size:.2f} MB")
