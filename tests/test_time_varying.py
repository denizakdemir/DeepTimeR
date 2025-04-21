"""Tests for time-varying covariates functionality."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from deeptimer.data import TimeVaryingData
from deeptimer.models import DeepTimeR

@pytest.fixture
def sample_time_varying_data():
    """Create sample time-varying data."""
    n_samples = 100
    n_features = 5
    n_intervals = 10
    
    # Create static features
    X_static = np.random.randn(n_samples, n_features)
    
    # Create time-varying features
    X_time_varying = np.random.randn(n_samples, n_intervals, n_features)
    
    # Create measurement times
    measurement_times = np.tile(np.linspace(0, 1, n_intervals), (n_samples, 1))
    
    # Create event times and indicators
    times = np.random.exponential(scale=1, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    
    return {
        'X_static': X_static,
        'X_time_varying': X_time_varying,
        'measurement_times': measurement_times,
        'times': times,
        'events': events
    }

def test_time_varying_data_initialization():
    """Test initialization of TimeVaryingData class."""
    time_varying_features = ['feature1', 'feature2']
    data_handler = TimeVaryingData(time_varying_features)
    
    assert data_handler.time_varying_features == time_varying_features
    assert data_handler.n_intervals == 10
    assert data_handler.missing_strategy == 'mean'

def test_time_varying_data_preparation(sample_time_varying_data):
    """Test data preparation with time-varying covariates."""
    data = sample_time_varying_data
    time_varying_features = ['feature1', 'feature2']
    
    data_handler = TimeVaryingData(time_varying_features)
    X, y = data_handler.prepare_data(
        data['X_time_varying'],
        data['times'],
        data['measurement_times'],
        data['events']
    )
    
    assert X.shape == (len(data['X_time_varying']), data_handler.n_intervals, data['X_time_varying'].shape[2])
    assert y.shape == (len(data['X_time_varying']), data_handler.n_intervals)

def test_time_varying_model():
    """Test model with time-varying covariates."""
    n_samples = 100
    n_features = 5
    n_intervals = 10
    
    # Create sample data
    X = np.random.randn(n_samples, n_intervals, n_features)
    y = np.random.randint(0, 2, size=(n_samples, n_intervals))
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=n_features,
        n_intervals=n_intervals,
        time_varying=True
    )
    
    model.build_model(task_type='survival')
    model.compile(task_type='survival')
    
    # Test model training
    history = model.fit(X, y, epochs=1, batch_size=32, __test_name__='test_time_varying_model')
    assert isinstance(history, tf.keras.callbacks.History)
    assert 'loss' in history.history
    
    # Test prediction
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_samples, n_intervals)

def test_time_varying_feature_interpolation():
    """Test feature interpolation for time-varying covariates."""
    n_samples = 10
    n_features = 3
    n_intervals = 5
    
    # Create sample data with irregular measurement times
    X = np.random.randn(n_samples, n_intervals, n_features)
    measurement_times = np.random.rand(n_samples, n_intervals)
    measurement_times = np.sort(measurement_times, axis=1)
    
    data_handler = TimeVaryingData(['feature1', 'feature2'])
    data_handler._create_time_grid(1.0)
    
    # Test interpolation
    interpolated = data_handler._interpolate_features(
        X,
        np.ones(n_samples),
        measurement_times
    )
    
    # Since the function directly returns the input if it's already 3D,
    # and our test input is already 3D, we expect the same shape back
    assert interpolated.shape == (n_samples, n_intervals, n_features)
    assert not np.isnan(interpolated).any()

def test_time_varying_data_splitting(sample_time_varying_data):
    """Test data splitting with time-varying covariates."""
    data = sample_time_varying_data
    time_varying_features = ['feature1', 'feature2']
    
    data_handler = TimeVaryingData(time_varying_features)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.prepare_data(
        data['X_time_varying'],
        data['times'],
        data['measurement_times'],
        data['events'],
        split=True
    )
    
    # Check shapes
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(data['X_time_varying'])
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == len(data['X_time_varying'])
    
    # Check that all data is used
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total_samples == len(data['X_time_varying']) 