import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.results = {}
        
    def evaluate_comprehensive(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.results
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba, save_path=None):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def calculate_cost_benefit(self, y_test, y_pred, 
                               fraud_cost=100, investigation_cost=10):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Cost of false negatives (missed fraud)
        fn_cost = fn * fraud_cost
        
        # Cost of false positives (unnecessary investigations)
        fp_cost = fp * investigation_cost
        
        # Benefit from true positives (prevented fraud)
        tp_benefit = tp * fraud_cost
        
        # Net benefit
        net_benefit = tp_benefit - fp_cost - fn_cost
        
        return {
            'false_negative_cost': fn_cost,
            'false_positive_cost': fp_cost,
            'true_positive_benefit': tp_benefit,
            'net_benefit': net_benefit
        }
    
    def find_optimal_threshold(self, y_test, y_pred_proba, metric='f1'):
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_test, y_pred)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return {
            'optimal_threshold': optimal_threshold,
            'optimal_score': optimal_score,
            'all_thresholds': thresholds,
            'all_scores': scores
        }
