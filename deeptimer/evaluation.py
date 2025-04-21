"""Evaluation module for DeepTimeR models.

This module provides tools for evaluating the performance of DeepTimeR models
using various metrics:
- Concordance index (C-index)
- Brier score and integrated Brier score
- Calibration curves
- Cross-validation

The module supports evaluation for different types of time-to-event analysis:
- Standard survival analysis
- Competing risks analysis
- Multi-state modeling
"""

import numpy as np
import tensorflow as tf
from typing import Union, List, Tuple, Optional, Dict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, brier_score_loss

class ModelEvaluator:
    """Class for evaluating DeepTimeR model performance.
    
    This class provides methods for calculating various evaluation metrics
    for DeepTimeR models, including discrimination and calibration measures.
    
    Attributes:
        model: Trained DeepTimeR model instance.
        task_type (str): Type of analysis task.
    """
    
    def __init__(self, model, task_type: str = 'survival'):
        """Initialize the evaluator.
        
        Args:
            model: Trained DeepTimeR model instance.
            task_type: Type of analysis task. Must be one of:
                     - 'survival': Standard survival analysis
                     - 'competing_risks': Competing risks analysis
                     - 'multistate': Multi-state modeling
                     Defaults to 'survival'.
        
        Raises:
            ValueError: If task_type is not one of the supported types.
        """
        if task_type not in ['survival', 'competing_risks', 'multistate']:
            raise ValueError("Invalid task_type")
            
        self.model = model
        self.task_type = task_type
    
    def concordance_index(self, 
                         X: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray,
                         event_type: Optional[np.ndarray] = None) -> float:
        """Calculate Harrell's concordance index (C-index).
        
        The C-index measures the model's ability to correctly rank order
        subjects by their risk. A value of 0.5 indicates random predictions,
        while 1.0 indicates perfect discrimination.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            times: Event or censoring times of shape (n_samples,).
            events: Event indicators (1 for event, 0 for censoring) of shape (n_samples,).
            event_type: Type of event for competing risks of shape (n_samples,).
                      Required for competing risks analysis. Defaults to None.
        
        Returns:
            float: Concordance index value between 0 and 1.
        
        Raises:
            ValueError: If event_type is not provided for competing risks analysis.
        """
        # Get predicted survival probabilities
        preds = self.model.predict(X)
        
        if self.task_type == 'survival':
            # For survival, use predicted survival probabilities
            risk_scores = 1 - preds[:, -1]  # Use last time point
        elif self.task_type == 'competing_risks':
            # For competing risks, use predicted cumulative incidence
            if event_type is None:
                raise ValueError("event_type must be provided for competing risks")
            risk_scores = preds[:, -1, event_type - 1]  # Use last time point for specific risk
        else:
            raise ValueError(f"C-index not implemented for task type: {self.task_type}")
        
        # Calculate concordance
        n_pairs = 0
        n_concordant = 0
        
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                # Only consider pairs where at least one event occurred
                if events[i] == 1 or events[j] == 1:
                    n_pairs += 1
                    
                    # Check if predictions are concordant with observed times
                    if (times[i] < times[j] and risk_scores[i] > risk_scores[j]) or \
                       (times[i] > times[j] and risk_scores[i] < risk_scores[j]):
                        n_concordant += 1
                    elif times[i] == times[j] and risk_scores[i] == risk_scores[j]:
                        n_concordant += 0.5
        
        return n_concordant / n_pairs if n_pairs > 0 else 0.0
    
    def brier_score(self,
                   X: np.ndarray,
                   times: np.ndarray,
                   events: np.ndarray,
                   time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate Brier score at specified time points.
        
        The Brier score measures the accuracy of predicted probabilities.
        Lower values indicate better calibration.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            times: Event or censoring times of shape (n_samples,).
            events: Event indicators of shape (n_samples,).
            time_points: Time points at which to calculate Brier score.
                       If None, uses unique observed times. Defaults to None.
        
        Returns:
            np.ndarray: Array of Brier scores at each time point.
        """
        if time_points is None:
            time_points = np.unique(times)
        
        # Get predicted survival probabilities
        preds = self.model.predict(X)
        
        # Initialize Brier scores
        brier_scores = np.zeros(len(time_points))
        
        for i, t in enumerate(time_points):
            # Find interval containing time point
            interval_idx = np.searchsorted(self.model.time_grid, t) - 1
            interval_idx = min(interval_idx, self.model.n_intervals - 1)
            
            # Get predicted survival probability at this time point
            pred_surv = preds[:, interval_idx]
            
            # Calculate observed survival status
            obs_surv = (times > t).astype(float)
            
            # Calculate Brier score
            brier_scores[i] = brier_score_loss(obs_surv, pred_surv)
        
        return brier_scores
    
    def integrated_brier_score(self,
                             X: np.ndarray,
                             times: np.ndarray,
                             events: np.ndarray) -> float:
        """Calculate integrated Brier score.
        
        The integrated Brier score provides an overall measure of prediction
        accuracy by integrating the Brier score over time.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            times: Event or censoring times of shape (n_samples,).
            events: Event indicators of shape (n_samples,).
        
        Returns:
            float: Integrated Brier score.
        """
        # Get time points for integration
        time_points = np.linspace(0, np.max(times), 100)
        
        # Calculate Brier scores
        brier_scores = self.brier_score(X, times, events, time_points)
        
        # Integrate using trapezoidal rule
        return np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])
    
    def calibration_curve(self,
                        X: np.ndarray,
                        times: np.ndarray,
                        events: np.ndarray,
                        time_point: float,
                        n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate calibration curve at a specific time point.
        
        The calibration curve compares predicted probabilities with observed
        event rates. Perfect calibration would result in points lying on the
        diagonal line.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            times: Event or censoring times of shape (n_samples,).
            events: Event indicators of shape (n_samples,).
            time_point: Time point for calibration.
            n_bins: Number of bins for calibration. Defaults to 10.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - Predicted probabilities for each bin
            - Observed event rates for each bin
        """
        # Get predicted survival probabilities
        preds = self.model.predict(X)
        
        # Find interval containing time point
        interval_idx = np.searchsorted(self.model.time_grid, time_point) - 1
        interval_idx = min(interval_idx, self.model.n_intervals - 1)
        
        # Get predicted survival probability at this time point
        pred_surv = preds[:, interval_idx]
        
        # Bin predicted probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate observed survival in each bin
        obs_surv = np.zeros(n_bins)
        pred_surv_binned = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (pred_surv >= bins[i]) & (pred_surv < bins[i + 1])
            if np.any(mask):
                obs_surv[i] = np.mean((times[mask] > time_point).astype(float))
                pred_surv_binned[i] = np.mean(pred_surv[mask])
        
        return pred_surv_binned, obs_surv

def cross_validate(model_class,
                  X: np.ndarray,
                  y: np.ndarray,
                  n_splits: int = 5,
                  random_state: Optional[int] = None,
                  custom_metrics: Optional[Dict[str, callable]] = None,
                  **kwargs) -> Dict[str, List[float]]:
    """Perform k-fold cross-validation.
    
    This function performs k-fold cross-validation and calculates evaluation
    metrics for each fold.
    
    Args:
        model_class: DeepTimeR model class to use.
        X: Input features of shape (n_samples, n_features).
        y: Target values of shape (n_samples, 2) containing times and events.
        n_splits: Number of cross-validation folds. Defaults to 5.
        random_state: Random seed for reproducibility. Defaults to None.
        custom_metrics: Dictionary of custom metric functions. Each function should
                      take (model, X, y) as arguments. Defaults to None.
        **kwargs: Additional arguments to pass to model constructor.
    
    Returns:
        Dict[str, List[float]]: Dictionary containing lists of evaluation metrics
                              for each fold. Keys are:
                              - 'c_index': Concordance index
                              - 'integrated_brier_score': Integrated Brier score
                              - Plus any custom metrics provided
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize metrics dictionary
    metrics = {
        'c_index': [],
        'integrated_brier_score': []
    }
    
    # Add custom metrics to tracking dictionary
    if custom_metrics is not None:
        for metric_name in custom_metrics:
            metrics[metric_name] = []
            
    # Separate model_kwargs from custom_metrics
    model_kwargs = {k: v for k, v in kwargs.items()}
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create and train model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Create evaluator
        evaluator = ModelEvaluator(model)
        
        # Calculate standard metrics
        c_index = evaluator.concordance_index(X_test, y_test[:, 0], y_test[:, 1])
        ibs = evaluator.integrated_brier_score(X_test, y_test[:, 0], y_test[:, 1])
        
        metrics['c_index'].append(c_index)
        metrics['integrated_brier_score'].append(ibs)
        
        # Calculate custom metrics if provided
        if custom_metrics is not None:
            for metric_name, metric_func in custom_metrics.items():
                try:
                    metric_value = metric_func(model, X_test, y_test)
                    metrics[metric_name].append(metric_value)
                except Exception as e:
                    # If metric fails, use a default value (0.5)
                    metrics[metric_name].append(0.5)
    
    return metrics 