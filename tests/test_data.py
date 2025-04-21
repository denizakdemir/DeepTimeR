"""Comprehensive tests for data handling functionality."""

import numpy as np
import pytest
from deeptimer.data import (
    SurvivalData,
    CompetingRisksData,
    MultiStateData,
    TimeVaryingData
)

@pytest.fixture
def sample_survival_data():
    """Create sample survival data."""
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    
    return X, times, events

@pytest.fixture
def sample_competing_risks_data():
    """Create sample competing risks data."""
    n_samples = 100
    n_features = 5
    n_risks = 3
    
    X = np.random.randn(n_samples, n_features)
    times = np.random.exponential(scale=1, size=n_samples)
    event_types = np.random.randint(0, n_risks + 1, size=n_samples)
    events = (event_types > 0).astype(int)
    
    return X, times, event_types, events

@pytest.fixture
def sample_multistate_data():
    """Create sample multi-state data."""
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    transitions = []
    for _ in range(n_samples):
        time = np.random.exponential(scale=1)
        from_state = 'state1'
        to_state = np.random.choice(['state2', 'state3'])
        transitions.append((time, from_state, to_state))
    
    return X, transitions

@pytest.fixture
def sample_time_varying_data():
    """Create sample time-varying data."""
    n_samples = 100
    n_features = 5
    n_intervals = 10
    
    X = np.random.randn(n_samples, n_intervals, n_features)
    times = np.random.exponential(scale=1, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    measurement_times = np.tile(np.linspace(0, 1, n_intervals), (n_samples, 1))
    
    return X, times, events, measurement_times

def test_survival_data_initialization():
    """Test initialization of SurvivalData class."""
    data_handler = SurvivalData()
    assert data_handler.n_intervals == 10
    assert data_handler.missing_strategy == 'mean'
    
    # Test with custom parameters
    data_handler = SurvivalData(n_intervals=20, missing_strategy='median')
    assert data_handler.n_intervals == 20
    assert data_handler.missing_strategy == 'median'

def test_survival_data_preparation(sample_survival_data):
    """Test data preparation for survival analysis."""
    X, times, events = sample_survival_data
    
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    assert X_processed.shape[0] == len(X)
    assert y.shape[0] == len(X)
    assert y.shape[1] == data_handler.n_intervals
    assert np.all(np.isfinite(X_processed))
    assert np.all(np.isfinite(y))

def test_survival_data_splitting(sample_survival_data):
    """Test data splitting for survival analysis."""
    X, times, events = sample_survival_data
    
    data_handler = SurvivalData()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.prepare_data(
        X, times, events, split=True
    )
    
    # Check shapes
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(X)
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == len(X)
    
    # Check that all data is used
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total_samples == len(X)

def test_competing_risks_data_initialization():
    """Test initialization of CompetingRisksData class."""
    data_handler = CompetingRisksData()
    assert data_handler.n_intervals == 10
    assert data_handler.missing_strategy == 'mean'
    
    # Test with custom parameters
    data_handler = CompetingRisksData(n_intervals=20, missing_strategy='median')
    assert data_handler.n_intervals == 20
    assert data_handler.missing_strategy == 'median'

def test_competing_risks_data_preparation(sample_competing_risks_data):
    """Test data preparation for competing risks analysis."""
    X, times, event_types, events = sample_competing_risks_data
    
    data_handler = CompetingRisksData()
    X_processed, y = data_handler.prepare_data(X, times, event_types, events)
    
    assert X_processed.shape[0] == len(X)
    assert y.shape[0] == len(X)
    assert y.shape[1] == data_handler.n_intervals
    assert y.shape[2] == len(np.unique(event_types[event_types > 0]))
    assert np.all(np.isfinite(X_processed))
    assert np.all(np.isfinite(y))

def test_multistate_data_initialization():
    """Test initialization of MultiStateData class."""
    state_structure = {
        'states': ['state1', 'state2', 'state3'],
        'transitions': [('state1', 'state2'), ('state1', 'state3')]
    }
    
    data_handler = MultiStateData(state_structure)
    assert data_handler.state_structure == state_structure
    assert data_handler.n_intervals == 10
    assert data_handler.missing_strategy == 'mean'
    
    # Test with custom parameters
    data_handler = MultiStateData(
        state_structure,
        n_intervals=20,
        missing_strategy='median'
    )
    assert data_handler.n_intervals == 20
    assert data_handler.missing_strategy == 'median'

def test_multistate_data_preparation(sample_multistate_data):
    """Test data preparation for multi-state analysis."""
    X, transitions = sample_multistate_data
    
    state_structure = {
        'states': ['state1', 'state2', 'state3'],
        'transitions': [('state1', 'state2'), ('state1', 'state3')]
    }
    
    data_handler = MultiStateData(state_structure)
    X_processed, y = data_handler.prepare_data(X, transitions)
    
    assert X_processed.shape[0] == len(X)
    assert y.shape[0] == len(X)
    assert y.shape[1] == data_handler.n_intervals
    assert y.shape[2] == len(state_structure['states'])
    assert y.shape[3] == len(state_structure['states'])
    assert np.all(np.isfinite(X_processed))
    assert np.all(np.isfinite(y))

def test_time_varying_data_initialization():
    """Test initialization of TimeVaryingData class."""
    time_varying_features = ['feature1', 'feature2']
    data_handler = TimeVaryingData(time_varying_features)
    
    assert data_handler.time_varying_features == time_varying_features
    assert data_handler.n_intervals == 10
    assert data_handler.missing_strategy == 'mean'
    
    # Test with custom parameters
    data_handler = TimeVaryingData(
        time_varying_features,
        n_intervals=20,
        missing_strategy='median'
    )
    assert data_handler.n_intervals == 20
    assert data_handler.missing_strategy == 'median'

def test_time_varying_data_preparation(sample_time_varying_data):
    """Test data preparation for time-varying covariates."""
    X, times, events, measurement_times = sample_time_varying_data
    
    time_varying_features = ['feature1', 'feature2']
    data_handler = TimeVaryingData(time_varying_features)
    X_processed, y = data_handler.prepare_data(
        X, times, measurement_times, events
    )
    
    assert X_processed.shape[0] == len(X)
    assert X_processed.shape[1] == data_handler.n_intervals
    assert X_processed.shape[2] == X.shape[2]
    assert y.shape[0] == len(X)
    assert y.shape[1] == data_handler.n_intervals
    assert np.all(np.isfinite(X_processed))
    assert np.all(np.isfinite(y))

def test_data_error_handling():
    """Test error handling in data classes."""
    # Test invalid n_intervals
    with pytest.raises(ValueError):
        SurvivalData(n_intervals=0)
    
    # Test invalid missing_strategy
    with pytest.raises(ValueError):
        SurvivalData(missing_strategy='invalid')
    
    # Test invalid state structure
    with pytest.raises(ValueError):
        MultiStateData({})
    
    # Test invalid time-varying features
    with pytest.raises(ValueError):
        TimeVaryingData([])
    
    # Test invalid data shapes
    X = np.random.randn(10, 5)
    times = np.random.exponential(scale=1, size=5)  # Wrong length
    events = np.random.binomial(1, 0.7, size=10)
    
    data_handler = SurvivalData()
    with pytest.raises(ValueError):
        data_handler.prepare_data(X, times, events)

def test_data_validation():
    """Test data validation functionality."""
    # Test missing values handling
    X = np.random.randn(10, 5)
    X[0, 0] = np.nan
    times = np.random.exponential(scale=1, size=10)
    events = np.random.binomial(1, 0.7, size=10)
    
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    assert not np.isnan(X_processed).any()
    
    # Test negative times
    times[0] = -1
    with pytest.raises(ValueError):
        data_handler.prepare_data(X, times, events)
    
    # Test invalid event values
    events[0] = 2
    with pytest.raises(ValueError):
        data_handler.prepare_data(X, times, events)

def test_data_scaling():
    """Test feature scaling functionality."""
    X = np.random.randn(100, 5)
    times = np.random.exponential(scale=1, size=100)
    events = np.random.binomial(1, 0.7, size=100)
    
    data_handler = SurvivalData()
    X_processed, _ = data_handler.prepare_data(X, times, events)
    
    # Check that features are scaled
    assert np.allclose(np.mean(X_processed, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(X_processed, axis=0), 1, atol=1e-6)

def test_data_time_discretization():
    """Test time discretization functionality."""
    X = np.random.randn(100, 5)
    times = np.random.exponential(scale=1, size=100)
    events = np.random.binomial(1, 0.7, size=100)
    
    data_handler = SurvivalData(n_intervals=20)
    _, y = data_handler.prepare_data(X, times, events)
    
    # Check that time is properly discretized
    assert y.shape[1] == 20
    assert np.all(np.isin(y, [-1, 0, 1]))  # -1: censored, 0: survived, 1: event 