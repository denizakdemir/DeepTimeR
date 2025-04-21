"""Tests for time-varying covariates and multi-state modeling functionality."""

import numpy as np
import tensorflow as tf
import pytest
from deeptimer.models import DeepTimeR
from deeptimer.data import TimeVaryingData, MultiStateData

class TestTimeVaryingMultistate:
    """Test suite for time-varying covariates and multi-state modeling."""
    
    @pytest.fixture
    def sample_time_varying_data(self):
        """Create sample time-varying data."""
        n_samples = 100
        n_intervals = 10
        n_features = 5
        X = np.random.randn(n_samples, n_intervals, n_features)
        times = np.random.uniform(0, 100, n_samples)
        events = np.random.binomial(1, 0.5, n_samples)
        return X, times, events
    
    @pytest.fixture
    def sample_multistate_data(self):
        """Create sample multi-state data."""
        n_samples = 100
        n_intervals = 10
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        transitions = [
            (0.3, 'state0', 'state1'),  # State 0 to State 1
            (0.2, 'state1', 'state2'),  # State 1 to State 2
            (0.1, 'state0', 'state2')   # State 0 to State 2
        ]
        return X, transitions
    
    def test_time_varying_model_initialization(self):
        """Test initialization of time-varying model."""
        model = DeepTimeR(
            input_dim=5,
            n_intervals=10,
            task_type='survival',
            time_varying=True
        )
        assert model.input_dim == 5
        assert model.n_intervals == 10
        assert model.time_varying == True
    
    def test_time_varying_model_building(self):
        """Test building of time-varying model."""
        model = DeepTimeR(
            input_dim=5,
            n_intervals=10,
            task_type='survival',
            time_varying=True
        )
        assert isinstance(model.model, tf.keras.Model)
    
    def test_time_varying_model_fit(self, sample_time_varying_data):
        """Test fitting time-varying model."""
        X, times, events = sample_time_varying_data
        model = DeepTimeR(
            input_dim=X.shape[-1],  # Use last dimension as feature dimension
            n_intervals=10,
            task_type='survival',
            time_varying=True
        )
        model.compile(optimizer='adam')
        model.fit(X, np.column_stack((times, events)), epochs=1)
    
    def test_time_varying_model_predict(self, sample_time_varying_data):
        """Test prediction with time-varying model."""
        X, times, events = sample_time_varying_data
        model = DeepTimeR(
            input_dim=X.shape[-1],  # Use last dimension as feature dimension
            n_intervals=10,
            task_type='survival',
            time_varying=True
        )
        model.compile(optimizer='adam')
        predictions = model.predict(X)
        assert predictions.shape[1] == model.n_intervals
    
    def test_multistate_model_initialization(self):
        """Test initialization of multi-state model."""
        state_structure = {'states': ['state0', 'state1', 'state2'], 
                          'transitions': [('state0', 'state1'), ('state0', 'state2'), ('state1', 'state2')]}
        model = DeepTimeR(
            input_dim=5,
            n_intervals=10,
            task_type='multistate',
            state_structure=state_structure
        )
        assert model.input_dim == 5
        assert model.state_structure['states'] == state_structure['states']
        assert model.state_structure['transitions'] == state_structure['transitions']
    
    def test_multistate_model_building(self):
        """Test building of multistate model."""
        state_structure = {'states': ['state0', 'state1', 'state2'], 
                          'transitions': [('state0', 'state1'), ('state0', 'state2'), ('state1', 'state2')]}
        model = DeepTimeR(
            input_dim=5,
            n_intervals=10,
            task_type='multistate',
            state_structure=state_structure
        )
        assert isinstance(model.model, tf.keras.Model)
    
    def test_multistate_model_fit(self, sample_multistate_data):
        """Test fitting multistate model."""
        X, transitions = sample_multistate_data
        state_structure = {'states': ['state0', 'state1', 'state2'], 
                          'transitions': [('state0', 'state1'), ('state0', 'state2'), ('state1', 'state2')]}
        model = DeepTimeR(
            input_dim=X.shape[1],
            n_intervals=10,
            task_type='multistate',
            state_structure=state_structure
        )
        model.compile(optimizer='adam')
        model.fit(X, transitions, epochs=1)
    
    def test_multistate_model_predict(self, sample_multistate_data):
        """Test prediction with multistate model."""
        X, transitions = sample_multistate_data
        state_structure = {'states': ['state0', 'state1', 'state2'], 
                          'transitions': [('state0', 'state1'), ('state0', 'state2'), ('state1', 'state2')]}
        model = DeepTimeR(
            input_dim=X.shape[1],
            n_intervals=10,
            task_type='multistate',
            state_structure=state_structure
        )
        model.compile(optimizer='adam')
        predictions = model.predict(X)
        assert predictions.shape[1] == model.n_intervals
    
    def test_time_varying_multistate_combined(self):
        """Test combined time-varying and multistate functionality."""
        # Generate sample time-varying data
        n_samples = 100
        n_features = 5
        n_timesteps = 10  # Change to match n_intervals
        X = np.random.randn(n_samples, n_timesteps, n_features)
        
        # Generate multistate transitions
        transitions = []
        for i in range(n_samples):
            time_val = np.random.uniform(0, 1.0)
            from_state = 'state0'
            to_state = np.random.choice(['state1', 'state2'])
            transitions.append((time_val, from_state, to_state))
        
        state_structure = {'states': ['state0', 'state1', 'state2'], 
                          'transitions': [('state0', 'state1'), ('state0', 'state2'), ('state1', 'state2')]}
        
        model = DeepTimeR(
            input_dim=n_features,
            n_intervals=10,
            task_type='multistate',
            state_structure=state_structure,
            time_varying=True
        )
        model.compile(optimizer='adam')
        model.fit(X, transitions, epochs=1)
        predictions = model.predict(X)
        assert predictions.shape[1] == model.n_intervals 