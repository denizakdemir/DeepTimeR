"""Tests for advanced evaluation metrics."""

import numpy as np
import pytest
from deeptimer import DeepTimeR, AdvancedEvaluator
from deeptimer.data import SurvivalData

def test_time_dependent_roc():
    """Test time-dependent ROC curve computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(input_dim=n_features)
    model.build_model(task_type='survival')
    model.compile(task_type='survival')
    model.fit(X, data.prepare_data(X, times, events)[1])
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(model, data)
    
    # Test ROC computation
    times = np.linspace(0, max(times), 5)
    roc_data = evaluator.compute_time_dependent_roc(times)
    
    assert 'tpr' in roc_data
    assert 'fpr' in roc_data
    assert 'auc' in roc_data
    assert len(roc_data['auc']) == len(times)
    assert all(0 <= auc <= 1 for auc in roc_data['auc'])

def test_calibration():
    """Test calibration computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(input_dim=n_features)
    model.build_model(task_type='survival')
    model.compile(task_type='survival')
    model.fit(X, data.prepare_data(X, times, events)[1])
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(model, data)
    
    # Test calibration computation
    times = np.linspace(0, max(times), 5)
    cal_data = evaluator.compute_calibration(times)
    
    assert 'mean_pred' in cal_data
    assert 'mean_true' in cal_data
    assert 'bin_edges' in cal_data
    assert len(cal_data['mean_pred']) == len(times)
    assert all(0 <= pred <= 1 for pred in cal_data['mean_pred'][0])
    assert all(0 <= true <= 1 for true in cal_data['mean_true'][0])

def test_dynamic_metrics():
    """Test dynamic prediction metrics computation."""
    # Create synthetic data
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1.0, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    data = SurvivalData(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(input_dim=n_features)
    model.build_model(task_type='survival')
    model.compile(task_type='survival')
    model.fit(X, data.prepare_data(X, times, events)[1])
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(model, data)
    
    # Test dynamic metrics computation
    times = np.linspace(0, max(times), 5)
    metrics = evaluator.compute_dynamic_metrics(times)
    
    assert 'brier_scores' in metrics
    assert 'c_indices' in metrics
    assert 'auc_scores' in metrics
    assert len(metrics['brier_scores']) == len(times)
    assert len(metrics['c_indices']) == len(times)
    assert len(metrics['auc_scores']) == len(times)
    assert all(0 <= score <= 1 for score in metrics['brier_scores'])
    assert all(0 <= c_index <= 1 for c_index in metrics['c_indices'])
    assert all(0 <= auc <= 1 for auc in metrics['auc_scores'])

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
    model = DeepTimeR(input_dim=n_features)
    model.build_model(task_type='survival')
    model.compile(task_type='survival')
    model.fit(X, data.prepare_data(X, times, events)[1])
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(model, data)
    
    # Test plotting functions
    times = np.linspace(0, max(times), 5)
    
    # Test ROC plotting
    evaluator.plot_time_dependent_roc(times)
    
    # Test calibration plotting
    evaluator.plot_calibration(times)
    
    # Test dynamic metrics plotting
    evaluator.plot_dynamic_metrics(times)
    
    # Test saving plots
    evaluator.plot_time_dependent_roc(times, save_path='test_roc.png')
    evaluator.plot_calibration(times, save_path='test_calibration.png')
    evaluator.plot_dynamic_metrics(times, save_path='test_metrics.png') 