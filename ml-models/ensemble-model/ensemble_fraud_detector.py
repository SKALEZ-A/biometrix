import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
from datetime import datetime

class EnsembleFraudDetector:
    def __init__(self, model_weights: Dict[str, float] = None):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_weights = model_weights or {
            'random_forest': 0.25,
            'gradient_boosting': 0.25,
            'logistic_regression': 0.15,
            'svm': 0.15,
            'neural_network': 0.10,
            'adaboost': 0.10
        }
        self.performance_metrics = {}
        self.is_trained = False
        
    def initialize_models(self):
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        
        self.models['adaboost'] = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        
    def train(self, X: pd.DataFrame, y: pd.Series, validate: bool = True):
        print(f"Training ensemble model with {len(X)} samples...")
        
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        
        self.initialize_models()
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            start_time = datetime.now()
            
            model.fit(X_scaled, y)
            
            train_time = (datetime.now() - start_time).total_seconds()
            
            if validate:
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                self.performance_metrics[model_name] = {
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'train_time': train_time
                }
                
                print(f"{model_name} - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            else:
                self.performance_metrics[model_name] = {
                    'train_time': train_time
                }
        
        self.is_trained = True
        print("\nEnsemble training complete!")
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        weighted_predictions = np.zeros((len(X), 2))
        
        for model_name, model in self.models.items():
            model_pred = model.predict_proba(X_scaled)
            weight = self.model_weights[model_name]
            weighted_predictions += weight * model_pred
        
        return weighted_predictions
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        probas = self.predict_proba(X)
        predictions = []
        
        for i, proba in enumerate(probas):
            fraud_prob = proba[1]
            
            model_predictions = {}
            X_scaled = self.scaler.transform(X.iloc[[i]])
            
            for model_name, model in self.models.items():
                model_proba = model.predict_proba(X_scaled)[0]
                model_predictions[model_name] = float(model_proba[1])
            
            variance = np.var(list(model_predictions.values()))
            confidence = 1.0 - min(variance * 4, 1.0)
            
            predictions.append({
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(fraud_prob >= 0.5),
                'confidence': float(confidence),
                'model_predictions': model_predictions,
                'prediction_variance': float(variance)
            })
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        importance_dict = {}
        
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            for feature, importance in zip(self.feature_names, rf_importance):
                importance_dict[feature] = float(importance)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before explanation")
        
        prediction = self.predict_with_confidence(X.iloc[[index]])[0]
        feature_importance = self.get_feature_importance()
        
        sample_features = X.iloc[index].to_dict()
        
        top_features = []
        for feature, importance in list(feature_importance.items())[:10]:
            top_features.append({
                'feature': feature,
                'value': float(sample_features[feature]),
                'importance': importance
            })
        
        return {
            'prediction': prediction,
            'top_contributing_features': top_features,
            'sample_data': sample_features
        }
    
    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        print("Optimizing ensemble weights...")
        
        X_scaled = self.scaler.transform(X_val)
        
        model_predictions = {}
        for model_name, model in self.models.items():
            model_predictions[model_name] = model.predict_proba(X_scaled)[:, 1]
        
        from scipy.optimize import minimize
        
        def objective(weights):
            weighted_pred = np.zeros(len(X_val))
            for i, model_name in enumerate(self.models.keys()):
                weighted_pred += weights[i] * model_predictions[model_name]
            
            from sklearn.metrics import log_loss
            return log_loss(y_val, weighted_pred)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(len(self.models))]
        initial_weights = np.array([1.0 / len(self.models)] * len(self.models))
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = result.x
            for i, model_name in enumerate(self.models.keys()):
                self.model_weights[model_name] = float(optimized_weights[i])
            
            print("Optimized weights:")
            for model_name, weight in self.model_weights.items():
                print(f"  {model_name}: {weight:.4f}")
        else:
            print("Weight optimization failed, using default weights")
    
    def save_model(self, filepath: str):
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_weights = model_data['model_weights']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'pr_auc': float(average_precision_score(y_test, y_proba)),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        }
        
        return metrics
    
    def get_model_agreement(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for model in self.models.values():
            pred = (model.predict_proba(X_scaled)[:, 1] >= 0.5).astype(int)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        agreement = np.mean(predictions, axis=0)
        
        return agreement
    
    def detect_model_drift(self, X_new: pd.DataFrame, X_reference: pd.DataFrame) -> Dict[str, Any]:
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        for feature in self.feature_names:
            if feature in X_new.columns and feature in X_reference.columns:
                statistic, p_value = ks_2samp(
                    X_new[feature].values,
                    X_reference[feature].values
                )
                
                drift_results[feature] = {
                    'ks_statistic': float(statistic),
                    'p_value': float(p_value),
                    'has_drift': bool(p_value < 0.05)
                }
        
        drifted_features = [f for f, r in drift_results.items() if r['has_drift']]
        
        return {
            'feature_drift': drift_results,
            'drifted_features': drifted_features,
            'drift_percentage': len(drifted_features) / len(self.feature_names) * 100
        }
    
    def calibrate_probabilities(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        from sklearn.calibration import CalibratedClassifierCV
        
        print("Calibrating probability predictions...")
        
        X_scaled = self.scaler.transform(X_cal)
        
        for model_name, model in self.models.items():
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated_model.fit(X_scaled, y_cal)
            self.models[model_name] = calibrated_model
            
        print("Calibration complete!")
    
    def get_prediction_intervals(self, X: pd.DataFrame, confidence: float = 0.95) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        all_predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X_scaled)[:, 1]
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)
        
        return np.column_stack([lower_bound, upper_bound])

class FeatureEngineer:
    @staticmethod
    def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        if 'amount' in features.columns:
            features['amount_log'] = np.log1p(features['amount'])
            features['amount_squared'] = features['amount'] ** 2
            features['amount_sqrt'] = np.sqrt(features['amount'])
        
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['is_night'] = features['hour'].isin(range(0, 6)).astype(int)
        
        if 'sender_id' in features.columns and 'receiver_id' in features.columns:
            features['is_self_transfer'] = (features['sender_id'] == features['receiver_id']).astype(int)
        
        return features
    
    @staticmethod
    def create_velocity_features(df: pd.DataFrame, user_col: str, time_col: str) -> pd.DataFrame:
        features = df.copy()
        features = features.sort_values([user_col, time_col])
        
        features['txn_count_1h'] = features.groupby(user_col)[time_col].transform(
            lambda x: x.rolling('1H').count()
        )
        
        features['txn_count_24h'] = features.groupby(user_col)[time_col].transform(
            lambda x: x.rolling('24H').count()
        )
        
        if 'amount' in features.columns:
            features['amount_sum_1h'] = features.groupby(user_col)['amount'].transform(
                lambda x: x.rolling(window=10, min_periods=1).sum()
            )
            
            features['amount_mean_1h'] = features.groupby(user_col)['amount'].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
        
        return features
    
    @staticmethod
    def create_aggregation_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        features = df.copy()
        
        if 'amount' in features.columns:
            agg_features = features.groupby(group_col)['amount'].agg([
                'mean', 'median', 'std', 'min', 'max', 'sum'
            ]).add_prefix(f'{group_col}_amount_')
            
            features = features.merge(agg_features, left_on=group_col, right_index=True, how='left')
        
        count_features = features.groupby(group_col).size().to_frame(f'{group_col}_count')
        features = features.merge(count_features, left_on=group_col, right_index=True, how='left')
        
        return features

def train_ensemble_model(train_data: pd.DataFrame, target_col: str, 
                         val_data: pd.DataFrame = None) -> EnsembleFraudDetector:
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    engineer = FeatureEngineer()
    X_train = engineer.create_transaction_features(X_train)
    
    model = EnsembleFraudDetector()
    model.train(X_train, y_train, validate=True)
    
    if val_data is not None:
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        X_val = engineer.create_transaction_features(X_val)
        
        model.optimize_weights(X_val, y_val)
    
    return model

if __name__ == "__main__":
    print("Ensemble Fraud Detector initialized")
    print("Available models: Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Network, AdaBoost")
