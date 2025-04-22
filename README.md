# DeepTimeR

DeepTimeR is a sophisticated deep learning framework for time-to-event analysis, offering a unified approach to survival analysis, competing risks, and multi-state modeling with a strong focus on interpretability and uncertainty quantification.

**Author:** Deniz Akdemir (deniz.akdemir.work@gmail.com)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/denizakdemir/DeepTimeR)

## 🌟 Key Features

- **Unified Framework**: One integrated approach for multiple time-to-event analysis tasks
  - Traditional survival analysis
  - Competing risks analysis
  - Multi-state modeling
  - Time-varying covariates support

- **Robust Uncertainty Quantification**: Reliable confidence intervals with Monte Carlo Dropout
  - Mathematically constrained prediction bounds
  - Valid probability outputs guaranteed
  - Visualization with uncertainty bands

- **Advanced Interpretability Tools**:
  - SHAP-based feature importance analysis
  - LIME explanations for individual predictions
  - Partial dependence plots for feature effects

- **Comprehensive Evaluation Metrics**:
  - Concordance index for discrimination
  - Calibration curves and metrics
  - Prediction error curves
  - Time-dependent ROC curves

- **Principled Constraint Framework**:
  - Ensures valid probabilities and transitions
  - Based on isotonic regression
  - Mathematically optimal projections

## 📦 Installation

```bash
pip install deeptimer
```

## 🚀 Quick Start

```python
import numpy as np
from deeptimer import DeepTimeR, SurvivalData
from deeptimer.evaluation import ModelEvaluator

# Prepare your data
data_handler = SurvivalData()
X, y = data_handler.prepare_data(features, times, events)

# Create and train model
model = DeepTimeR(input_dim=X.shape[1], n_intervals=100)
model.build_model(task_type='survival')
model.compile(task_type='survival')
model.fit(X, y, epochs=100, batch_size=32)

# Make predictions with uncertainty
mean_preds, lower_bounds, upper_bounds = model.predict_with_uncertainty(X_test, n_samples=100)

# Evaluate model performance
evaluator = ModelEvaluator(model)
c_index = evaluator.concordance_index(X_test, times_test, events_test)
print(f"C-index: {c_index:.3f}")
```

## Usage

### Survival Analysis

```python
import numpy as np
from deeptimer import DeepTimeR, SurvivalData
from deeptimer.evaluation import ModelEvaluator
from deeptimer.advanced_interpretability import AdvancedInterpreter

# Prepare data
data_handler = SurvivalData()
X, y = data_handler.prepare_data(features, time, event)

# Initialize and train model
model = DeepTimeR(input_dim=X.shape[1], n_intervals=100)
model.build_model(task_type='survival')
model.compile(task_type='survival')
model.fit(X, y)

# Make predictions
survival_probs = model.predict(X)

# Evaluate model performance
evaluator = ModelEvaluator(model, task_type='survival')
c_index = evaluator.concordance_index(X, time, event)
ibs = evaluator.integrated_brier_score(X, time, event)
calibration_metrics = evaluator.calibration_metrics(X, time, event, time_point=5.0)

print(f"C-index: {c_index:.3f}")
print(f"Integrated Brier Score: {ibs:.3f}")
print(f"Calibration metrics: {calibration_metrics}")

# Plot prediction error curve
evaluator.plot_prediction_error_curve(X, time, event, save_path="prediction_error.png")

# Plot calibration curve
time_points = [2.0, 5.0, 8.0]
evaluator.plot_calibration_curve(X, time, event, time_points, save_path="calibration.png")

# Model interpretation
interpreter = AdvancedInterpreter(model, data={"X": X, "feature_names": feature_names})

# Get feature importance using SHAP values
importance = interpreter.get_feature_importance()
interpreter.plot_feature_importance(save_path="feature_importance.png")

# Get LIME explanation for a specific instance
lime_exp = interpreter.compute_lime_explanation(instance_idx=0)
interpreter.plot_lime_explanation(instance_idx=0, save_path="lime_explanation.png")

# Create partial dependence plot for important features
interpreter.plot_partial_dependence(
    features=["age", "biomarker"], 
    save_path="partial_dependence.png"
)
```

### Competing Risks

```python
from deeptimer import DeepTimeR, CompetingRisksData
from deeptimer.evaluation import ModelEvaluator
from deeptimer.advanced_evaluation import AdvancedEvaluator

# Prepare data
data_handler = CompetingRisksData()
X, y = data_handler.prepare_data(features, time, event_type, event)

# Initialize and train model with custom loss function
model = DeepTimeR(input_dim=X.shape[1], 
                 n_intervals=100,
                 n_risks=2)

# Use gradient clipping to prevent exploding gradients
model.build_model(task_type='competing_risks')
model.compile(
    task_type='competing_risks',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
)
model.fit(X, y)

# Make predictions
cif = model.predict(X)

# Advanced evaluation
evaluator = AdvancedEvaluator(model, data={"X": X, "times": time, "events": event, "event_types": event_type})

# Time-dependent ROC curves
time_points = [2.0, 5.0, 8.0]
evaluator.plot_time_dependent_roc(time_points, save_path="time_dependent_roc.png")

# Dynamic prediction metrics
evaluator.plot_dynamic_metrics(time_points, save_path="dynamic_metrics.png")

# Make predictions with uncertainty using Monte Carlo Dropout
mean_preds, lower_bounds, upper_bounds = model.predict_with_uncertainty(
    X_new, n_samples=100
)

# Plot cumulative incidence functions with uncertainty
from deeptimer.utils import plot_cumulative_incidence_with_uncertainty

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

plot_cumulative_incidence_with_uncertainty(
    risk_predictions, lower_bound_dict, upper_bound_dict, 
    np.linspace(0, 10, 101),
    save_path="cif_with_uncertainty.png"
)
```

### Multi-State Modeling

```python
from deeptimer import DeepTimeR, MultiStateData
from deeptimer.constraints import ConstraintHandler

# Define state structure
state_structure = {
    'states': ['healthy', 'disease', 'death'],
    'transitions': [('healthy', 'disease'), 
                   ('healthy', 'death'),
                   ('disease', 'death')],
    'absorbing_states': ['death']
}

# Prepare data
data_handler = MultiStateData(state_structure)
X, y = data_handler.prepare_data(features, transitions)

# Initialize and train model
model = DeepTimeR(input_dim=X.shape[1],
                 n_intervals=100,
                 state_structure=state_structure)
model.build_model(task_type='multistate')
model.compile(task_type='multistate')
model.fit(X, y)

# Make predictions
state_probs = model.predict(X)

# Apply constraints to ensure valid transition probabilities
constraint_handler = ConstraintHandler()
constrained_probs = constraint_handler.apply_constraints(
    state_probs, model_type='multistate', state_structure=state_structure
)

# Calculate state occupation probabilities over time
from deeptimer.utils import calculate_state_occupation

state_occupation = calculate_state_occupation(constrained_probs, state_structure)

# Plot state occupation probabilities
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
time_points = np.linspace(0, 10, 100)
for i, state in enumerate(state_structure['states']):
    plt.plot(time_points, state_occupation[0, :, i], label=state)
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('State Occupation Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('state_occupation.png')
```

### Uncertainty Quantification

```python
import numpy as np
from deeptimer import DeepTimeR
from deeptimer.utils import plot_survival_curves_with_uncertainty

# Initialize and train model
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
    np.linspace(0, 10, 101),
    save_path='survival_with_uncertainty.png'
)
```

### Cross-Validation

```python
from deeptimer.evaluation import cross_validate

# Define custom metrics
def custom_metric(model, X, y):
    preds = model.predict(X)
    # Custom calculation here
    return score

# Cross-validate model
results = cross_validate(
    DeepTimeR,  
    X, y, 
    n_splits=5,
    random_state=42,
    custom_metrics={'my_metric': custom_metric},
    # Model parameters
    input_dim=X.shape[1],
    n_intervals=100
)

print(f"Average C-index: {np.mean(results['c_index']):.3f}")
print(f"Average IBS: {np.mean(results['integrated_brier_score']):.3f}")
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