"""Utility functions for DeepTimeR models.

This module provides utility functions for:
- Feature importance analysis
- Visualization of model outputs
- Time-varying effect analysis

The visualization functions support different types of analysis:
- Survival curves
- Cumulative incidence functions
- State occupation probabilities
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union

def get_feature_importance(model, X: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """Calculate feature importance using gradients for each time interval.
    
    This function computes feature importance scores by calculating the
    gradient of model predictions with respect to input features. The
    importance score for each feature at each time point is the average
    absolute gradient value.
    
    Args:
        model: Trained DeepTimeR model instance.
        X: Input features of shape (n_samples, n_features).
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
    
    Returns:
        np.ndarray: Feature importance scores of shape (n_intervals, n_features).
                   Higher values indicate more important features.
    """
    importance = np.zeros((len(time_grid)-1, X.shape[1]))
    
    for t in range(len(time_grid)-1):
        with tf.GradientTape() as tape:
            tape.watch(X)
            pred_t = model.model([X, np.ones((len(X), 1)) * t])
        grads = tape.gradient(pred_t, X)
        importance[t] = np.mean(np.abs(grads), axis=0)
    
    return importance

def plot_feature_importance(importance: np.ndarray,
                          time_grid: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          top_n: int = 5) -> None:
    """Plot feature importance scores for each time interval.
    
    This function creates a bar plot showing the top N most important
    features for each time interval.
    
    Args:
        importance: Feature importance scores of shape (n_intervals, n_features).
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        feature_names: Names of features. If None, uses generic names.
                     Defaults to None.
        top_n: Number of top features to plot. Defaults to 5.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(importance.shape[1])]
    
    plt.figure(figsize=(12, 6))
    for t in range(len(time_grid)-1):
        sorted_idx = np.argsort(importance[t])[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:top_n]]
        top_importance = importance[t, sorted_idx[:top_n]]
        
        plt.subplot(1, len(time_grid)-1, t+1)
        plt.bar(range(top_n), top_importance)
        plt.xticks(range(top_n), top_features, rotation=45, ha='right')
        plt.title(f'Time {time_grid[t]:.1f}-{time_grid[t+1]:.1f}')
    
    plt.tight_layout()
    plt.show()

def plot_survival_curves(survival_probs: np.ndarray,
                        time_grid: np.ndarray,
                        labels: Optional[List[str]] = None) -> None:
    """Plot survival curves.
    
    This function creates a step plot of survival probabilities over time.
    It can plot either a single curve or multiple curves with labels.
    
    Args:
        survival_probs: Survival probabilities of shape (n_samples, n_intervals)
                      or (n_intervals,) for a single curve.
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        labels: Labels for different curves. Required if plotting multiple curves.
               Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    if len(survival_probs.shape) == 1:
        plt.step(time_grid[:-1], survival_probs, where='post')
    else:
        for i in range(survival_probs.shape[0]):
            label = labels[i] if labels else f'Curve {i}'
            plt.step(time_grid[:-1], survival_probs[i], where='post', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Survival Curves')
    if labels:
        plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_survival_curves_with_uncertainty(survival_probs: np.ndarray,
                                        lower_bound: np.ndarray,
                                        upper_bound: np.ndarray,
                                        time_grid: np.ndarray,
                                        labels: Optional[List[str]] = None) -> None:
    """Plot survival curves with uncertainty bands.
    
    This function creates a step plot of survival probabilities over time with
    uncertainty bands representing the 95% confidence interval.
    
    Args:
        survival_probs: Mean survival probabilities of shape (n_samples, n_intervals)
                      or (n_intervals,) for a single curve.
        lower_bound: Lower bound of 95% confidence interval, same shape as survival_probs.
        upper_bound: Upper bound of 95% confidence interval, same shape as survival_probs.
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        labels: Labels for different curves. Required if plotting multiple curves.
               Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    if len(survival_probs.shape) == 1:
        # Plot single curve with uncertainty
        plt.step(time_grid[:-1], survival_probs, where='post', color='blue', label='Mean prediction')
        plt.fill_between(time_grid[:-1], lower_bound, upper_bound, alpha=0.3, color='blue', step='post',
                        label='95% confidence interval')
    else:
        # Plot multiple curves with uncertainty
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(survival_probs.shape[0]):
            color = colors[i % len(colors)]
            label = labels[i] if labels else f'Curve {i}'
            
            plt.step(time_grid[:-1], survival_probs[i], where='post', color=color, label=label)
            plt.fill_between(time_grid[:-1], lower_bound[i], upper_bound[i], 
                           alpha=0.3, color=color, step='post', label=f'95% CI - {label}')
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Survival Curves with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cumulative_incidence(cif: Dict[int, np.ndarray],
                            time_grid: np.ndarray,
                            risk_names: Optional[Dict[int, str]] = None) -> None:
    """Plot cumulative incidence functions for competing risks.
    
    This function creates a step plot of cumulative incidence functions
    for each competing risk over time.
    
    Args:
        cif: Dictionary mapping risk types to their cumulative incidence
            functions. Each function should be of shape (n_intervals,).
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        risk_names: Dictionary mapping risk types to their names.
                  If None, uses generic names. Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    for risk, curve in cif.items():
        label = risk_names[risk] if risk_names else f'Risk {risk}'
        plt.step(time_grid[:-1], curve, where='post', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Cumulative Incidence Functions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_cumulative_incidence_with_uncertainty(cif: Dict[int, np.ndarray],
                                             lower_bounds: Dict[int, np.ndarray],
                                             upper_bounds: Dict[int, np.ndarray],
                                             time_grid: np.ndarray,
                                             risk_names: Optional[Dict[int, str]] = None) -> None:
    """Plot cumulative incidence functions with uncertainty bands.
    
    This function creates a step plot of cumulative incidence functions
    for each competing risk over time with uncertainty bands representing
    the 95% confidence interval.
    
    Args:
        cif: Dictionary mapping risk types to their mean cumulative incidence
            functions. Each function should be of shape (n_intervals,).
        lower_bounds: Dictionary mapping risk types to their lower bound estimates.
        upper_bounds: Dictionary mapping risk types to their upper bound estimates.
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        risk_names: Dictionary mapping risk types to their names.
                  If None, uses generic names. Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    color_idx = 0
    
    for risk, curve in cif.items():
        color = colors[color_idx % len(colors)]
        label = risk_names[risk] if risk_names else f'Risk {risk}'
        
        # Plot mean prediction
        plt.step(time_grid[:-1], curve, where='post', color=color, label=label)
        
        # Plot uncertainty bands
        lower_curve = lower_bounds[risk]
        upper_curve = upper_bounds[risk]
        plt.fill_between(time_grid[:-1], lower_curve, upper_curve,
                        alpha=0.3, color=color, step='post',
                        label=f'95% CI - {label}')
        
        color_idx += 1
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.title('Cumulative Incidence Functions with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_state_occupation(state_probs: np.ndarray,
                         time_grid: np.ndarray,
                         state_names: Optional[List[str]] = None) -> None:
    """Plot state occupation probabilities for multi-state model.
    
    This function creates a step plot of state occupation probabilities
    over time for each state in a multi-state model.
    
    Args:
        state_probs: State occupation probabilities of shape
                   (n_intervals, n_states).
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        state_names: Names for each state. If None, uses generic names.
                   Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    n_states = state_probs.shape[-1]
    for state in range(n_states):
        label = state_names[state] if state_names else f'State {state}'
        plt.step(time_grid[:-1], state_probs[:, state], where='post', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('State Occupation Probability')
    plt.title('State Occupation Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_state_occupation_with_uncertainty(state_probs: np.ndarray,
                                         lower_bounds: np.ndarray,
                                         upper_bounds: np.ndarray,
                                         time_grid: np.ndarray,
                                         state_names: Optional[List[str]] = None) -> None:
    """Plot state occupation probabilities with uncertainty bands.
    
    This function creates a step plot of state occupation probabilities
    over time for each state in a multi-state model with uncertainty bands
    representing the 95% confidence interval.
    
    Args:
        state_probs: Mean state occupation probabilities of shape
                   (n_intervals, n_states).
        lower_bounds: Lower bounds of 95% confidence interval, same shape as state_probs.
        upper_bounds: Upper bounds of 95% confidence interval, same shape as state_probs.
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        state_names: Names for each state. If None, uses generic names.
                   Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
    """
    plt.figure(figsize=(10, 6))
    
    n_states = state_probs.shape[-1]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for state in range(n_states):
        color = colors[state % len(colors)]
        label = state_names[state] if state_names else f'State {state}'
        
        # Plot mean prediction
        plt.step(time_grid[:-1], state_probs[:, state], where='post', color=color, label=label)
        
        # Plot uncertainty bands
        plt.fill_between(time_grid[:-1], 
                        lower_bounds[:, state], 
                        upper_bounds[:, state],
                        alpha=0.3, color=color, step='post',
                        label=f'95% CI - {label}')
    
    plt.xlabel('Time')
    plt.ylabel('State Occupation Probability')
    plt.title('State Occupation Probabilities with Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_time_varying_effects(model,
                            X: np.ndarray,
                            time_grid: np.ndarray,
                            feature_idx: int,
                            feature_name: Optional[str] = None) -> None:
    """Plot how a feature's effect varies across time intervals.
    
    This function analyzes how changing a feature's value affects model
    predictions at different time points. It creates a step plot showing
    the effect size over time.
    
    Args:
        model: Trained DeepTimeR model instance.
        X: Input features of shape (n_samples, n_features).
        time_grid: Time grid used for discretization of shape (n_intervals + 1,).
        feature_idx: Index of the feature to analyze.
        feature_name: Name of the feature. If None, uses generic name.
                    Defaults to None.
    
    Note:
        The plot is displayed using matplotlib and not returned.
        The effect size is calculated by perturbing the feature value by 1.0
        and measuring the change in predictions.
    """
    if feature_name is None:
        feature_name = f'Feature {feature_idx}'
    
    # Create perturbed inputs
    X_perturbed = X.copy()
    X_perturbed[:, feature_idx] += 1.0
    
    # Get predictions for original and perturbed inputs
    preds_original = model.predict(X)
    preds_perturbed = model.predict(X_perturbed)
    
    # Calculate effect size
    effect = preds_perturbed - preds_original
    
    plt.figure(figsize=(10, 6))
    plt.step(time_grid[:-1], effect, where='post')
    plt.xlabel('Time')
    plt.ylabel('Effect Size')
    plt.title(f'Time-Varying Effect of {feature_name}')
    plt.grid(True)
    plt.show() 