import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.background_data = None
        
    def set_background_data(self, X_background: np.ndarray):
        """Set background data for SHAP calculations"""
        self.background_data = X_background
        logger.info(f"Background data set with {len(X_background)} samples")
        
    def explain_prediction(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanations for predictions"""
        if self.background_data is None:
            raise ValueError("Background data not set. Call set_background_data first.")
        
        shap_values = self._calculate_shap_values(X)
        
        explanation = {
            'shap_values': shap_values.tolist(),
            'feature_importance': self._get_feature_importance(shap_values),
            'top_features': self._get_top_features(shap_values, top_k=5),
            'base_value': self._get_base_value()
        }
        
        return explanation
        
    def _calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values using kernel SHAP approximation"""
        n_samples = len(X)
        n_features = X.shape[1]
        shap_values = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            for j in range(n_features):
                # Simplified SHAP calculation
                X_with = X[i].copy()
                X_without = X[i].copy()
                X_without[j] = np.mean(self.background_data[:, j])
                
                pred_with = self.model.predict(X_with.reshape(1, -1), verbose=0)[0]
                pred_without = self.model.predict(X_without.reshape(1, -1), verbose=0)[0]
                
                shap_values[i, j] = pred_with - pred_without
        
        return shap_values
        
    def _get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values"""
        importance = np.abs(shap_values).mean(axis=0)
        
        return {
            self.feature_names[i]: float(importance[i])
            for i in range(len(self.feature_names))
        }
        
    def _get_top_features(self, shap_values: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top contributing features"""
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        return [
            {
                'feature': self.feature_names[i],
                'importance': float(importance[i]),
                'shap_value': float(shap_values[:, i].mean())
            }
            for i in top_indices
        ]
        
    def _get_base_value(self) -> float:
        """Get base prediction value"""
        if self.background_data is None:
            return 0.0
        
        base_predictions = self.model.predict(self.background_data, verbose=0)
        return float(np.mean(base_predictions))
        
    def generate_force_plot_data(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Generate data for force plot visualization"""
        shap_values = self._calculate_shap_values(X[instance_idx:instance_idx+1])
        
        return {
            'base_value': self._get_base_value(),
            'shap_values': shap_values[0].tolist(),
            'feature_values': X[instance_idx].tolist(),
            'feature_names': self.feature_names,
            'prediction': float(self.model.predict(X[instance_idx:instance_idx+1], verbose=0)[0])
        }
        
    def generate_summary_plot_data(self, X: np.ndarray) -> Dict[str, Any]:
        """Generate data for summary plot"""
        shap_values = self._calculate_shap_values(X)
        
        return {
            'shap_values': shap_values.tolist(),
            'feature_values': X.tolist(),
            'feature_names': self.feature_names,
            'feature_importance': self._get_feature_importance(shap_values)
        }
