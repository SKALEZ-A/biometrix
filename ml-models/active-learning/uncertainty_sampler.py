import numpy as np
from typing import List, Tuple
from scipy.stats import entropy

class UncertaintySampler:
    def __init__(self, model, strategy: str = 'entropy'):
        self.model = model
        self.strategy = strategy
        
    def sample(self, unlabeled_data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, List[int]]:
        if self.strategy == 'entropy':
            return self._entropy_sampling(unlabeled_data, n_samples)
        elif self.strategy == 'margin':
            return self._margin_sampling(unlabeled_data, n_samples)
        elif self.strategy == 'least_confident':
            return self._least_confident_sampling(unlabeled_data, n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _entropy_sampling(self, data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, List[int]]:
        probas = self.model.predict_proba(data)
        uncertainties = entropy(probas.T)
        
        # Get indices of most uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        return data[uncertain_indices], uncertain_indices.tolist()
    
    def _margin_sampling(self, data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, List[int]]:
        probas = self.model.predict_proba(data)
        
        # Calculate margin (difference between top 2 probabilities)
        sorted_probas = np.sort(probas, axis=1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        
        # Get indices of smallest margins (most uncertain)
        uncertain_indices = np.argsort(margins)[:n_samples]
        
        return data[uncertain_indices], uncertain_indices.tolist()
    
    def _least_confident_sampling(self, data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, List[int]]:
        probas = self.model.predict_proba(data)
        
        # Get maximum probability for each sample
        max_probas = np.max(probas, axis=1)
        
        # Get indices of least confident samples
        uncertain_indices = np.argsort(max_probas)[:n_samples]
        
        return data[uncertain_indices], uncertain_indices.tolist()
    
    def diversity_sampling(self, data: np.ndarray, n_samples: int, 
                          labeled_data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Calculate distances to labeled data
        distances = euclidean_distances(data, labeled_data)
        min_distances = np.min(distances, axis=1)
        
        # Select samples farthest from labeled data
        diverse_indices = np.argsort(min_distances)[-n_samples:]
        
        return data[diverse_indices], diverse_indices.tolist()
