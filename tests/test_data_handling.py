"""Tests for data handling functionality."""

import numpy as np
import pytest
from deeptimer.data import TimeVaryingData, MultiStateData

class TestDataHandling:
    """Test suite for data handling functionality."""
    
    @pytest.fixture
    def sample_time_varying_features(self):
        """Create sample time-varying features."""
        n_samples = 100
        n_intervals = 10
        n_features = 5
        return np.random.randn(n_samples, n_intervals, n_features)
    
    @pytest.fixture
    def sample_multistate_transitions(self):
        """Create sample multi-state transitions."""
        return [
            (0.5, 'healthy', 'sick'),
            (1.0, 'sick', 'dead'),
            (0.8, 'healthy', 'dead')
        ]
    
    def test_time_varying_data_initialization(self):
        """Test initialization of TimeVaryingData."""
        time_varying_features = ['feature1', 'feature2']
        data_handler = TimeVaryingData(
            time_varying_features=time_varying_features,
            n_intervals=10
        )
        assert data_handler.time_varying_features == time_varying_features
        assert data_handler.n_intervals == 10
    
    def test_time_varying_data_preprocessing(self, sample_time_varying_features):
        """Test preprocessing of time-varying data."""
        time_varying_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        data_handler = TimeVaryingData(
            time_varying_features=time_varying_features,
            n_intervals=10
        )
        
        # Test preprocessing
        X_processed = data_handler._preprocess_features(sample_time_varying_features)
        assert X_processed is not None
        assert X_processed.shape == sample_time_varying_features.shape
    
    def test_time_varying_data_interpolation(self):
        """Test interpolation of time-varying features."""
        time_varying_features = ['feature1', 'feature2']
        data_handler = TimeVaryingData(
            time_varying_features=time_varying_features,
            n_intervals=10
        )
        
        # Create sample data with missing measurements
        n_samples = 100
        n_features = 2
        features = np.random.randn(n_samples, n_features)
        times = np.random.uniform(0, 100, n_samples)
        measurement_times = np.random.uniform(0, 100, (n_samples, 5))
        
        # Test interpolation
        interpolated = data_handler._interpolate_features(
            features=features,
            times=times,
            measurement_times=measurement_times
        )
        assert interpolated is not None
        assert interpolated.shape == (n_samples, 10, n_features)
    
    def test_multistate_data_initialization(self):
        """Test initialization of MultiStateData."""
        state_structure = {
            'states': ['healthy', 'sick', 'dead'],
            'transitions': [('healthy', 'sick'), ('sick', 'dead'), ('healthy', 'dead')]
        }
        data_handler = MultiStateData(
            state_structure=state_structure,
            n_intervals=10
        )
        assert data_handler.state_structure == state_structure
        assert data_handler.n_intervals == 10
    
    def test_multistate_data_transition_validation(self, sample_multistate_transitions):
        """Test validation of multi-state transitions."""
        state_structure = {
            'states': ['healthy', 'sick', 'dead'],
            'transitions': [('healthy', 'sick'), ('sick', 'dead'), ('healthy', 'dead')]
        }
        data_handler = MultiStateData(
            state_structure=state_structure,
            n_intervals=10
        )
        
        # Test transition validation
        data_handler._validate_transitions(sample_multistate_transitions)
    
    def test_multistate_data_preparation(self):
        """Test preparation of multi-state data."""
        state_structure = {
            'states': ['healthy', 'sick', 'dead'],
            'transitions': [('healthy', 'sick'), ('sick', 'dead'), ('healthy', 'dead')]
        }
        data_handler = MultiStateData(
            state_structure=state_structure,
            n_intervals=10
        )
        
        # Create sample data
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        transitions = [
            (0.5, 'healthy', 'sick'),
            (1.0, 'sick', 'dead'),
            (0.8, 'healthy', 'dead')
        ]
        
        # Test data preparation
        X_processed, y = data_handler.prepare_data(
            X=X,
            transitions=transitions
        )
        assert X_processed is not None
        assert y is not None
        assert X_processed.shape[0] == n_samples
        assert y.shape[1] == 10  # n_intervals
        assert y.shape[2] == 3  # n_transitions
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        time_varying_features = ['feature1', 'feature2']
        data_handler = TimeVaryingData(
            time_varying_features=time_varying_features,
            n_intervals=10,
            missing_strategy='mean'
        )
        
        # Create sample data with missing values
        n_samples = 100
        n_features = 2
        X = np.random.randn(n_samples, n_features)
        X[0, 0] = np.nan  # Add missing value
        
        # Test missing value handling
        X_processed = data_handler._handle_missing_values(X)
        assert X_processed is not None
        assert not np.isnan(X_processed).any()
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        time_varying_features = ['feature1', 'feature2']
        data_handler = TimeVaryingData(
            time_varying_features=time_varying_features,
            n_intervals=10
        )
        
        # Create sample data
        n_samples = 100
        n_features = 2
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Test data splitting
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_handler.split_data(
            X=X,
            y=y,
            test_size=0.2,
            val_size=0.2
        )
        assert X_train.shape[0] == 64  # 80% of 80%
        assert X_val.shape[0] == 16   # 20% of 80%
        assert X_test.shape[0] == 20  # 20% of total 