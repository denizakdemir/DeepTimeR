import numpy as np
import tensorflow as tf
from deeptimer.evaluation import ModelEvaluator, cross_validate
from sklearn.model_selection import KFold
import pytest

class MockModel:
    def __init__(self, time_grid=None, **kwargs):
        if time_grid is None:
            time_grid = np.linspace(0, 10, 11)
        self.time_grid = time_grid
        self.n_intervals = len(time_grid) - 1
    
    def predict(self, X):
        # Return mock predictions for testing
        n_samples = X.shape[0]
        n_intervals = self.n_intervals
        return np.random.uniform(0, 1, size=(n_samples, n_intervals))
    
    def fit(self, X, y, **kwargs):
        # Mock fit method that returns a dummy history dict
        return {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.6, 0.7, 0.8]
        }

@pytest.fixture
def mock_model():
    time_grid = np.linspace(0, 10, 11)
    return MockModel(time_grid)

@pytest.fixture
def mock_data():
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    times = np.random.uniform(0, 10, size=n_samples)
    events = np.random.binomial(1, 0.7, size=n_samples)
    return X, times, events

def test_concordance_index(mock_model, mock_data):
    X, times, events = mock_data
    evaluator = ModelEvaluator(mock_model)
    
    # Test basic functionality
    c_index = evaluator.concordance_index(X, times, events)
    assert 0 <= c_index <= 1
    
    # Test with competing risks
    event_type = np.random.randint(1, 3, size=len(times))
    c_index_cr = evaluator.concordance_index(X, times, events, event_type)
    assert 0 <= c_index_cr <= 1

def test_brier_score(mock_model, mock_data):
    X, times, events = mock_data
    evaluator = ModelEvaluator(mock_model)
    
    # Specify custom time points to match test expectations
    time_points = np.linspace(0, 10, 11)
    brier_scores = evaluator.brier_score(X, times, events, time_points=time_points)
    
    # Check that brier scores are calculated for the specified time points
    assert len(brier_scores) == len(time_points)
    assert np.all(0 <= brier_scores) and np.all(brier_scores <= 1)

def test_integrated_brier_score(mock_model, mock_data):
    X, times, events = mock_data
    evaluator = ModelEvaluator(mock_model)
    
    ibs = evaluator.integrated_brier_score(X, times, events)
    assert 0 <= ibs <= 1

def test_calibration_curve(mock_model, mock_data):
    X, times, events = mock_data
    evaluator = ModelEvaluator(mock_model)
    
    time_point = 5.0
    pred_probs, obs_probs = evaluator.calibration_curve(X, times, events, time_point)
    
    assert len(pred_probs) == len(obs_probs)
    assert np.all(np.logical_and(pred_probs >= 0, pred_probs <= 1))
    assert np.all(np.logical_and(obs_probs >= 0, obs_probs <= 1))
    
    # Test with custom number of bins
    pred_probs_custom, obs_probs_custom = evaluator.calibration_curve(
        X, times, events, time_point, n_bins=5
    )
    assert len(pred_probs_custom) == 5
    assert len(obs_probs_custom) == 5

def test_cross_validate(mock_data):
    X, times, events = mock_data
    y = np.column_stack((times, events))
    
    # Test basic functionality
    metrics = cross_validate(
        MockModel,
        X,
        y,
        n_splits=3,
        random_state=42,
        time_grid=np.linspace(0, 10, 11)
    )
    
    assert 'c_index' in metrics
    assert 'integrated_brier_score' in metrics
    assert len(metrics['c_index']) == 3
    assert len(metrics['integrated_brier_score']) == 3
    
    # Test with custom metrics
    def custom_metric(model, X, y):
        return 0.5
    
    metrics_custom = cross_validate(
        MockModel,
        X,
        y,
        n_splits=3,
        random_state=42,
        time_grid=np.linspace(0, 10, 11),
        custom_metrics={'custom': custom_metric}
    )
    
    assert 'custom' in metrics_custom
    assert len(metrics_custom['custom']) == 3
    assert np.all(np.array(metrics_custom['custom']) == 0.5) 