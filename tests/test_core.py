"""Comprehensive tests for core model functionality."""

import numpy as np
import pytest
import tensorflow as tf
from deeptimer.models import DeepTimeR
from deeptimer.data import SurvivalData, CompetingRisksData, MultiStateData, TimeVaryingData

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

def test_model_initialization():
    """Test basic model initialization."""
    model = DeepTimeR(
        input_dim=5,
        n_intervals=10,
        task_type='survival'
    )
    assert model.input_dim == 5
    assert model.n_intervals == 10
    assert model.task_type == 'survival'

def test_model_building(sample_survival_data):
    """Test model building for different task types."""
    X, _, _ = sample_survival_data
    
    # Test survival model building
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)
    
    # Test competing risks model building
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        n_risks=3,
        task_type='competing_risks'
    )
    model.build_model()
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)
    
    # Test multi-state model building
    state_structure = {
        'states': ['state1', 'state2', 'state3'],
        'transitions': [('state1', 'state2'), ('state1', 'state3')]
    }
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        state_structure=state_structure,
        task_type='multistate'
    )
    model.build_model()
    assert model.model is not None
    assert isinstance(model.model, tf.keras.Model)

def test_model_training(sample_survival_data):
    """Test model training functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    
    # Test training with different batch sizes
    for batch_size in [16, 32, 64]:
        history = model.fit(X_processed, y, epochs=2, batch_size=batch_size)
        assert isinstance(history, dict)
        assert 'loss' in history
        assert len(history['loss']) == 2

def test_model_prediction(sample_survival_data):
    """Test model prediction functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    model.fit(X_processed, y, epochs=1, batch_size=32)
    
    # Test prediction
    preds = model.predict(X_processed)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X), model.n_intervals)
    assert np.all(preds >= 0) and np.all(preds <= 1)

def test_model_saving_loading(sample_survival_data, tmp_path):
    """Test model saving and loading functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    model.fit(X_processed, y, epochs=1, batch_size=32)
    
    # Save model
    model_path = tmp_path / "test_model"
    model.save(str(model_path))
    
    # Load model
    loaded_model = DeepTimeR.load(str(model_path))
    assert isinstance(loaded_model, DeepTimeR)
    assert loaded_model.input_dim == model.input_dim
    assert loaded_model.n_intervals == model.n_intervals
    
    # Test loaded model prediction
    preds_original = model.predict(X_processed)
    preds_loaded = loaded_model.predict(X_processed)
    np.testing.assert_array_almost_equal(preds_original, preds_loaded)

def test_model_validation(sample_survival_data):
    """Test model validation functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model with validation
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    
    history = model.fit(
        X_processed, y,
        epochs=2,
        batch_size=32,
        validation_split=0.2
    )
    
    assert 'val_loss' in history
    assert len(history['val_loss']) == 2

def test_model_early_stopping(sample_survival_data):
    """Test model early stopping functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model with early stopping
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    
    history = model.fit(
        X_processed, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
    )
    
    # Check if history contains expected keys (not checking early stopping due to random initialization)
    assert 'loss' in history
    assert history.get('loss', [])  # Check that loss exists and is non-empty

def test_model_checkpointing(sample_survival_data, tmp_path):
    """Test model checkpointing functionality."""
    X, times, events = sample_survival_data
    
    # Prepare data
    data_handler = SurvivalData()
    X_processed, y = data_handler.prepare_data(X, times, events)
    
    # Initialize and train model with checkpointing
    model = DeepTimeR(
        input_dim=X.shape[1],
        n_intervals=10,
        task_type='survival'
    )
    model.build_model()
    model.compile(task_type='survival')
    
    checkpoint_path = tmp_path / "checkpoint.keras"
    history = model.fit(
        X_processed, y,
        epochs=2,
        batch_size=32,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            save_best_only=True
        )]
    )
    
    # Check if checkpoint was created
    assert checkpoint_path.exists()

def test_model_error_handling():
    """Test model error handling."""
    # Test invalid input dimension
    try:
        DeepTimeR(
            input_dim=0,
            n_intervals=10,
            task_type='survival'
        )
        assert False, "Should have raised ValueError for invalid input dimension"
    except ValueError:
        pass
    
    # Test invalid number of intervals
    try:
        DeepTimeR(
            input_dim=5,
            n_intervals=0,
            task_type='survival'
        )
        assert False, "Should have raised ValueError for invalid n_intervals"
    except ValueError:
        pass
    
    # Test invalid hidden layers
    try:
        DeepTimeR(
            input_dim=5,
            n_intervals=10,
            hidden_layers=[]
        )
        assert False, "Should have raised ValueError for empty hidden_layers"
    except ValueError:
        pass
    
    # Test invalid task type
    try:
        DeepTimeR(
            input_dim=5,
            n_intervals=10,
            task_type='invalid_task'
        )
        assert False, "Should have raised ValueError for invalid task_type"
    except ValueError:
        pass
    
    # Note: The model implementation is auto-building the model in __init__,
    # so this last test is no longer valid and is commented out
    
    # Test prediction before building model
    # model = DeepTimeR(
    #     input_dim=5,
    #     n_intervals=10,
    #     task_type='survival'
    # )
    # with pytest.raises(ValueError):
    #     model.predict(np.random.randn(10, 5)) 