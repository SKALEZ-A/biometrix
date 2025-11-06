import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class LIMEExplainer:
    def __init__(self, model, feature_names, class_names=['Legitimate', 'Fraud']):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        
    def fit(self, X_train: np.ndarray, mode='classification'):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=True
        )
        
    def explain_instance(self, instance: np.ndarray, num_features=10):
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=len(self.class_names)
        )
        
        return explanation
    
    def explain_batch(self, X: np.ndarray, num_features=10):
        explanations = []
        for instance in X:
            exp = self.explain_instance(instance, num_features)
            explanations.append(exp)
        return explanations
    
    def get_feature_importance(self, explanation, label=1):
        feature_weights = explanation.as_list(label=label)
        importance_dict = {}
        
        for feature_desc, weight in feature_weights:
            feature_name = feature_desc.split()[0]
            importance_dict[feature_name] = weight
            
        return importance_dict
    
    def visualize_explanation(self, explanation, label=1, save_path=None):
        fig = explanation.as_pyplot_figure(label=label)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def get_global_importance(self, X: np.ndarray, num_samples=100, num_features=10):
        sample_indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        X_sample = X[sample_indices]
        
        feature_importance_sum = {name: 0.0 for name in self.feature_names}
        
        for instance in X_sample:
            exp = self.explain_instance(instance, num_features)
            importance = self.get_feature_importance(exp)
            
            for feature, weight in importance.items():
                if feature in feature_importance_sum:
                    feature_importance_sum[feature] += abs(weight)
        
        for feature in feature_importance_sum:
            feature_importance_sum[feature] /= len(X_sample)
        
        sorted_importance = sorted(
            feature_importance_sum.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance[:num_features])
    
    def plot_global_importance(self, importance_dict, save_path=None):
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.xlabel('Average Absolute Weight')
        plt.ylabel('Feature')
        plt.title('Global Feature Importance (LIME)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def generate_report(self, instance: np.ndarray, prediction: int, 
                       probability: float, num_features=10) -> Dict[str, Any]:
        explanation = self.explain_instance(instance, num_features)
        importance = self.get_feature_importance(explanation, label=prediction)
        
        report = {
            'prediction': self.class_names[prediction],
            'probability': float(probability),
            'feature_contributions': importance,
            'explanation_fit': explanation.score,
            'local_prediction': explanation.local_pred[prediction]
        }
        
        return report

def create_explainer(model, X_train, feature_names):
    explainer = LIMEExplainer(model, feature_names)
    explainer.fit(X_train)
    return explainer
