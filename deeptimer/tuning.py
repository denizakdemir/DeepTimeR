"""Hyperparameter tuning utilities for DeepTimeR models.

This module provides tools for optimizing DeepTimeR model hyperparameters using:
- Grid search
- Random search
- Bayesian optimization

The utilities support tuning of various hyperparameters including:
- Model architecture (hidden layers, dropout rates)
- Training parameters (learning rate, batch size)
- Regularization parameters (temporal smoothness)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import KFold
from scipy.stats import uniform, randint
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

class HyperparameterTuner:
    """Class for tuning DeepTimeR model hyperparameters.
    
    This class provides methods for optimizing model hyperparameters using
    different search strategies. It supports both discrete and continuous
    hyperparameters and uses cross-validation to evaluate parameter settings.
    
    Attributes:
        model_class: DeepTimeR model class to tune.
        param_space: Dictionary defining the parameter search space.
        cv: Number of cross-validation folds.
        scoring: Scoring metric to optimize.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(self,
                 model_class,
                 param_space: Dict,
                 cv: int = 5,
                 scoring: str = 'c_index',
                 random_state: Optional[int] = None):
        """Initialize the hyperparameter tuner.
        
        Args:
            model_class: DeepTimeR model class to tune.
            param_space: Dictionary defining parameter search space.
                       Keys are parameter names, values are either:
                       - List of values for grid search
                       - Tuple of (min, max) for random search
                       - skopt.space object for Bayesian optimization
            cv: Number of cross-validation folds. Defaults to 5.
            scoring: Scoring metric to optimize. Must be one of:
                   - 'c_index': Concordance index
                   - 'brier_score': Integrated Brier score
                   Defaults to 'c_index'.
            random_state: Random seed for reproducibility. Defaults to None.
        
        Raises:
            ValueError: If scoring metric is invalid.
        """
        if scoring not in ['c_index', 'brier_score']:
            raise ValueError("Invalid scoring metric")
            
        self.model_class = model_class
        self.param_space = param_space
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def _evaluate_params(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        params: Dict) -> float:
        """Evaluate a set of parameters using cross-validation.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 2) containing times and events.
            params: Dictionary of parameter values to evaluate.
        
        Returns:
            float: Average score across cross-validation folds.
        """
        kf = KFold(n_splits=self.cv, shuffle=True, 
                  random_state=self.random_state)
        
        scores = []
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            
            # Create evaluator
            evaluator = ModelEvaluator(model)
            
            # Calculate score
            if self.scoring == 'c_index':
                score = evaluator.concordance_index(
                    X_val, y_val[:, 0], y_val[:, 1]
                )
            else:  # brier_score
                score = -evaluator.integrated_brier_score(
                    X_val, y_val[:, 0], y_val[:, 1]
                )  # Negative because we want to maximize
            
            scores.append(score)
        
        return np.mean(scores)
    
    def grid_search(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   param_grid: Dict) -> Dict:
        """Perform grid search over parameter combinations.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 2) containing times and events.
            param_grid: Dictionary of parameter lists to search over.
        
        Returns:
            Dict: Best parameter settings found.
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        # Evaluate each combination
        results = []
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            score = self._evaluate_params(X, y, param_dict)
            results.append((param_dict, score))
        
        # Find best parameters
        best_params, best_score = max(results, key=lambda x: x[1])
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        
        return best_params
    
    def random_search(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     n_iter: int = 20) -> Dict:
        """Perform random search over parameter space.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 2) containing times and events.
            n_iter: Number of parameter settings to try. Defaults to 20.
        
        Returns:
            Dict: Best parameter settings found.
        """
        results = []
        
        for _ in range(n_iter):
            # Sample random parameters
            params = {}
            for name, (min_val, max_val) in self.param_space.items():
                if isinstance(min_val, int):
                    params[name] = randint(min_val, max_val).rvs(
                        random_state=self.random_state
                    )
                else:
                    params[name] = uniform(min_val, max_val).rvs(
                        random_state=self.random_state
                    )
            
            # Evaluate parameters
            score = self._evaluate_params(X, y, params)
            results.append((params, score))
        
        # Find best parameters
        best_params, best_score = max(results, key=lambda x: x[1])
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        
        return best_params
    
    def bayesian_optimization(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            n_iter: int = 20,
                            n_initial_points: int = 5) -> Dict:
        """Perform Bayesian optimization over parameter space.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 2) containing times and events.
            n_iter: Number of parameter settings to try. Defaults to 20.
            n_initial_points: Number of random points before optimization.
                           Defaults to 5.
        
        Returns:
            Dict: Best parameter settings found.
        """
        # Convert parameter space to skopt format
        search_space = {}
        for name, space in self.param_space.items():
            if isinstance(space, tuple):
                min_val, max_val = space
                if isinstance(min_val, int):
                    search_space[name] = Integer(min_val, max_val)
                else:
                    search_space[name] = Real(min_val, max_val)
            elif isinstance(space, list):
                search_space[name] = Categorical(space)
        
        # Create optimizer
        optimizer = BayesSearchCV(
            estimator=self.model_class(),
            search_spaces=search_space,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state,
            n_initial_points=n_initial_points
        )
        
        # Fit optimizer
        optimizer.fit(X, y)
        
        self.best_params_ = optimizer.best_params_
        self.best_score_ = optimizer.best_score_
        self.cv_results_ = optimizer.cv_results_
        
        return optimizer.best_params_
    
    def get_cv_results(self) -> Dict:
        """Get cross-validation results from the last tuning run.
        
        Returns:
            Dict: Dictionary containing cross-validation results.
        """
        if self.cv_results_ is None:
            raise ValueError("No tuning results available. Run a tuning method first.")
        return self.cv_results_ 