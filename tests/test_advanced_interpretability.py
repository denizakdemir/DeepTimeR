"""Tests for advanced interpretability features."""

import numpy as np
import pytest
from deeptimer import DeepTimeR, AdvancedInterpreter
from deeptimer.data import SurvivalData

def test_shap_values():
    """Test SHAP values computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=10,
        task_type='survival'
    )
    model.compile(optimizer='adam')
    model.fit(X, np.column_stack((times, events)), epochs=1)
    
    # Initialize interpreter
    interpreter = AdvancedInterpreter(model, data)
    
    # Test SHAP values computation
    shap_values = interpreter.compute_shap_values()
    
    assert shap_values.shape == (n_samples, n_features)
    assert not np.isnan(shap_values).any()
    assert not np.isinf(shap_values).any()

def test_lime_explanation():
    """Test LIME explanation computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=10,
        task_type='survival'
    )
    model.compile(optimizer='adam')
    model.fit(X, np.column_stack((times, events)), epochs=1)
    
    # Initialize interpreter
    interpreter = AdvancedInterpreter(model, data)
    
    # Test LIME explanation for a single instance
    instance_idx = 0
    explanation = interpreter.compute_lime_explanation(instance_idx)
    
    assert 'feature_importance' in explanation
    assert 'prediction' in explanation
    assert 'local_prediction' in explanation
    assert len(explanation['feature_importance']) == n_features

def test_partial_dependence():
    """Test partial dependence computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=10,
        task_type='survival'
    )
    model.compile(optimizer='adam')
    model.fit(X, np.column_stack((times, events)), epochs=1)
    
    # Initialize interpreter
    interpreter = AdvancedInterpreter(model, data)
    
    # Test partial dependence computation
    feature_idx = 0
    grid_points = 10
    pd_data = interpreter.compute_partial_dependence(feature_idx, grid_points=grid_points)
    
    assert 'feature_values' in pd_data
    assert 'average_predictions' in pd_data
    assert len(pd_data['feature_values']) == 1  # One array for the specified feature
    assert len(pd_data['average_predictions']) == 1  # One array for the specified feature
    assert len(pd_data['feature_values'][0]) == grid_points  # Check actual grid point count
    assert len(pd_data['average_predictions'][0]) == grid_points

def test_feature_importance():
    """Test feature importance computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=10,
        task_type='survival'
    )
    model.compile(optimizer='adam')
    model.fit(X, np.column_stack((times, events)), epochs=1)
    
    # Initialize interpreter
    interpreter = AdvancedInterpreter(model, data)
    
    # Test feature importance computation
    importance = interpreter.get_feature_importance()
    
    assert len(importance) == n_features
    assert all(0 <= imp <= 1 for imp in importance)
    assert abs(sum(importance) - 1.0) < 1e-6

def test_plotting_functions():
    """Test plotting functions."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=10,
        task_type='survival'
    )
    model.compile(optimizer='adam')
    model.fit(X, np.column_stack((times, events)), epochs=1)
    
    # Initialize interpreter
    interpreter = AdvancedInterpreter(model, data)
    
    # Test SHAP summary plot
    interpreter.plot_shap_summary()
    
    # Test LIME explanation plot
    interpreter.plot_lime_explanation(0)
    
    # Test partial dependence plot
    interpreter.plot_partial_dependence(0)
    
    # Test feature importance plot
    interpreter.plot_feature_importance()
    
    # Test saving plots
    interpreter.plot_shap_summary(save_path='test_shap.png')
    interpreter.plot_lime_explanation(0, save_path='test_lime.png')
    interpreter.plot_partial_dependence(0, save_path='test_pd.png')
    interpreter.plot_feature_importance(save_path='test_importance.png') 