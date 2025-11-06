import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

class XGBoostFraudClassifier:
    def __init__(self, params=None):
        self.params = params or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 100),
                evals=evals,
                early_stopping_rounds=10,
                verbose_eval=10
            )
        else:
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 100)
            )
        
        return self
    
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)
        return np.column_stack([1 - probabilities, probabilities])
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self):
        importance = self.model.get_score(importance_type='weight')
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': [importance.get(f'f{i}', 0) for i in range(len(self.feature_names))]
            })
        else:
            importance_df = pd.DataFrame(list(importance.items()), 
                                        columns=['feature', 'importance'])
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save_model(self, filepath):
        self.model.save_model(filepath)
        joblib.dump(self.feature_names, filepath + '.features')
    
    def load_model(self, filepath):
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        self.feature_names = joblib.load(filepath + '.features')
