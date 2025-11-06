import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, model, param_distributions: Dict[str, List[Any]]):
        self.model = model
        self.param_distributions = param_distributions
        self.best_params = None
        self.best_score = None
        
    def random_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                     n_iter: int = 50, cv: int = 5) -> Dict[str, Any]:
        """Perform random search for hyperparameter tuning"""
        logger.info(f"Starting random search with {n_iter} iterations")
        
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=self.param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_
        }
        
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                   param_grid: Dict[str, List[Any]], cv: int = 5) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning"""
        logger.info("Starting grid search")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }
        
    def bayesian_optimization(self, X_train: np.ndarray, y_train: np.ndarray, 
                            n_iterations: int = 50) -> Dict[str, Any]:
        """Perform Bayesian optimization for hyperparameter tuning"""
        from skopt import BayesSearchCV
        
        logger.info(f"Starting Bayesian optimization with {n_iterations} iterations")
        
        bayes_search = BayesSearchCV(
            self.model,
            search_spaces=self.param_distributions,
            n_iter=n_iterations,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        bayes_search.fit(X_train, y_train)
        
        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_
        
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_results': bayes_search.optimizer_results_
        }
