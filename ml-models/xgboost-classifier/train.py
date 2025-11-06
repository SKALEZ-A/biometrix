import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime

class XGBoostFraudTrainer:
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.model = None
        self.feature_importance = None
        
    def get_default_config(self):
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42
        }
    
    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        X = df.drop(['is_fraud', 'transaction_id'], axis=1)
        y = df['is_fraud']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = xgb.XGBClassifier(**self.config)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['logloss', 'auc'],
            early_stopping_rounds=10,
            verbose=True
        )
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")
        
        self.config.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
    
    def save_model(self, path='xgboost_fraud_model.pkl'):
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='xgboost_fraud_model.pkl'):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = pd.DataFrame(model_data['feature_importance'])
        print(f"Model loaded from {path}")

if __name__ == '__main__':
    trainer = XGBoostFraudTrainer()
    
    X_train, X_test, y_train, y_test = trainer.load_data('data/fraud_transactions.csv')
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print("Training XGBoost model...")
    trainer.train(X_train, y_train, X_val, y_val)
    
    print("\nEvaluating model...")
    results = trainer.evaluate(X_test, y_test)
    
    print("\nTop 10 Important Features:")
    print(trainer.feature_importance.head(10))
    
    trainer.save_model()
