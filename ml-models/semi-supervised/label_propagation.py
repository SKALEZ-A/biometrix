import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, classification_report

class SemiSupervisedFraudDetector:
    def __init__(self, method='label_propagation', kernel='rbf', gamma=20):
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
        
    def train(self, X_labeled, y_labeled, X_unlabeled):
        # Combine labeled and unlabeled data
        X_combined = np.vstack([X_labeled, X_unlabeled])
        
        # Create labels array (-1 for unlabeled)
        y_combined = np.concatenate([
            y_labeled,
            np.full(len(X_unlabeled), -1)
        ])
        
        # Initialize model
        if self.method == 'label_propagation':
            self.model = LabelPropagation(
                kernel=self.kernel,
                gamma=self.gamma,
                max_iter=1000
            )
        elif self.method == 'label_spreading':
            self.model = LabelSpreading(
                kernel=self.kernel,
                gamma=self.gamma,
                max_iter=1000,
                alpha=0.2
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit model
        self.model.fit(X_combined, y_combined)
        
        # Get pseudo-labels for unlabeled data
        pseudo_labels = self.model.transduction_[len(X_labeled):]
        
        return pseudo_labels
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def get_label_distributions(self):
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.label_distributions_
    
    def select_confident_samples(self, X_unlabeled, confidence_threshold=0.9):
        probas = self.predict_proba(X_unlabeled)
        max_probas = np.max(probas, axis=1)
        
        confident_indices = np.where(max_probas >= confidence_threshold)[0]
        confident_samples = X_unlabeled[confident_indices]
        confident_labels = self.predict(confident_samples)
        
        return confident_samples, confident_labels, confident_indices
