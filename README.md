# DeepTimeR

DeepTimeR is a multitask deep learning model for time-to-event analysis, supporting survival analysis, competing risks, and multi-state modeling. The package provides a unified framework for handling different types of time-to-event data while maintaining interpretability through attention mechanisms and rule extraction.

## Installation

```bash
pip install deeptimer
```

## Features

- Support for multiple time-to-event analysis tasks:
  - Survival analysis
  - Competing risks
  - Multi-state modeling
- Feature attention mechanism for interpretability
- Uncertainty quantification using Monte Carlo Dropout
- Rule extraction from trained models
- Visualization tools for model outputs with uncertainty bounds
- Standardized data preprocessing

## Usage

### Survival Analysis

```python
import numpy as np
from deeptimer import DeepTimeR, SurvivalData

# Prepare data
data_handler = SurvivalData()
X, y = data_handler.prepare_data(features, time, event)

# Initialize and train model
model = DeepTimeR(input_dim=X.shape[1], time_points=np.linspace(0, 100, 100))
model.build_model(task_type='survival')
model.compile(task_type='survival')
model.fit(X, y)

# Make predictions
survival_probs, attention_weights = model.predict(X)

# Get feature importance
importance = get_feature_importance(model, X)
plot_feature_importance(importance, feature_names)

# Extract interpretable rules
rules = extract_rules(model, X, feature_names)
```

### Competing Risks

```python
from deeptimer import DeepTimeR, CompetingRisksData

# Prepare data
data_handler = CompetingRisksData()
X, y = data_handler.prepare_data(features, time, event_type, event)

# Initialize and train model
model = DeepTimeR(input_dim=X.shape[1], 
                 time_points=np.linspace(0, 100, 100),
                 n_risks=2)
model.build_model(task_type='competing_risks')
model.compile(task_type='competing_risks')
model.fit(X, y)

# Make predictions
cif, attention_weights = model.predict(X)

# Plot cumulative incidence functions
plot_cumulative_incidence(cif, time_points, risk_names)
```

### Multi-State Modeling

```python
from deeptimer import DeepTimeR, MultiStateData

# Define state structure
state_structure = {
    'states': ['healthy', 'disease', 'death'],
    'transitions': [('healthy', 'disease'), 
                   ('healthy', 'death'),
                   ('disease', 'death')]
}

# Prepare data
data_handler = MultiStateData(state_structure)
X, y = data_handler.prepare_data(features, transitions)

# Initialize and train model
model = DeepTimeR(input_dim=X.shape[1],
                 time_points=np.linspace(0, 100, 100),
                 state_structure=state_structure)
model.build_model(task_type='multistate')
model.compile(task_type='multistate')
model.fit(X, y)

# Make predictions
state_probs, attention_weights = model.predict(X)
```

### Uncertainty Quantification

```python
import numpy as np
from deeptimer import DeepTimeR
from deeptimer.utils import plot_survival_curves_with_uncertainty

# Initialize and train model (as shown in examples above)
model = DeepTimeR(input_dim=X.shape[1], n_intervals=100)
model.build_model(task_type='survival')
model.compile(task_type='survival')
model.fit(X, times, events)

# Make predictions with uncertainty using Monte Carlo Dropout
mean_preds, lower_bounds, upper_bounds = model.predict_with_uncertainty(
    X_new, n_samples=100
)

# Plot survival curves with uncertainty bands
plot_survival_curves_with_uncertainty(
    mean_preds[0], lower_bounds[0], upper_bounds[0], 
    np.linspace(0, 10, 101)
)
```

For competing risks analysis:

```python
from deeptimer.utils import plot_cumulative_incidence_with_uncertainty

# Get predictions with uncertainty for competing risks
mean_preds, lower_bounds, upper_bounds = model.predict_with_uncertainty(
    X_new, n_samples=100
)

# Convert to dictionary format for plotting
risk_predictions = {
    1: mean_preds[0, :, 0],
    2: mean_preds[0, :, 1]
}
lower_bound_dict = {
    1: lower_bounds[0, :, 0],
    2: lower_bounds[0, :, 1]
}
upper_bound_dict = {
    1: upper_bounds[0, :, 0],
    2: upper_bounds[0, :, 1]
}

# Plot with uncertainty bands
plot_cumulative_incidence_with_uncertainty(
    risk_predictions, lower_bound_dict, upper_bound_dict, 
    np.linspace(0, 10, 101)
)
```

## Model Architecture

The model consists of three main components:

1. **Shared Encoder**: A neural network that learns a shared representation of the input features, with an attention mechanism for interpretability.

2. **Task-Specific Decoders**: Separate decoders for each type of time-to-event analysis:
   - Survival analysis decoder
   - Competing risks decoder
   - Multi-state decoder

3. **Interpretability Features**:
   - Feature attention weights
   - Uncertainty quantification with Monte Carlo Dropout
   - Rule extraction from decision trees
   - Visualization tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 