# Prediction and Uncertainty Quantification in DeepTimeR

This document explains how survival probabilities and cumulative incidence functions (CIFs) are predicted and how uncertainty is quantified and constrained in DeepTimeR.

## How Predictions Work in DeepTimeR

The prediction process for new data follows these steps:

1. **Model Prediction**: The `predict` method in the `DeepTimeR` class takes input features and returns model predictions:
   ```python
   def predict(self, X: np.ndarray) -> np.ndarray:
       # Create time input spanning all intervals
       time_input = np.arange(self.n_intervals)
       time_input = np.tile(time_input, (len(X), 1))
       
       # Handle input features
       if len(X.shape) > 2:  # Handle time-varying features
           X_input = X if self.time_varying else X[:, 0, :]
       else:
           X_input = X
       
       return self.model.predict([X_input, time_input])
   ```

2. **Output Processing**: 
   - For **survival analysis**, the model outputs hazard probabilities which are converted to survival probabilities using the `SurvivalProbabilityLayer` with:
     ```python
     return tf.math.cumprod(1 - hazard_probs, axis=1)
     ```
   
   - For **competing risks analysis**, the `CompetingRisksProbabilityLayer` computes cause-specific probabilities:
     ```python
     overall_survival = tf.math.cumprod(1 - tf.reduce_sum(risk_hazards, axis=2), axis=1)
     return tf.expand_dims(overall_survival, axis=2) * risk_hazards
     ```

3. **Plotting**: The utility functions in `utils.py` create visualizations:
   ```python
   def plot_survival_curves(survival_probs, time_grid, labels=None):
       # Create step plots of survival probabilities
   
   def plot_cumulative_incidence(cif, time_grid, risk_names=None):
       # Plot step functions for each competing risk
   ```

## Implemented Uncertainty Quantification Using Monte Carlo Dropout

DeepTimeR now includes uncertainty quantification using Monte Carlo Dropout. This approach works by:

1. **Keeping dropout active during inference**:
   - Dropout layers added to the network remain active during prediction
   - This introduces randomness in each forward pass of the model

2. **Multiple forward passes**:
   - Running the model multiple times with different dropout patterns
   - Each pass produces a slightly different prediction
   
3. **Statistical aggregation**:
   - Computing the mean prediction across all samples
   - Calculating percentiles (e.g., 2.5% and 97.5%) for confidence bounds

The implementation is through the `predict_with_uncertainty` method:
   ```python
   def predict_with_uncertainty(self, X, n_samples=100, alpha=0.05):
       """Generate predictions with uncertainty bounds using Monte Carlo Dropout.
       
       Args:
           X: Input features
           n_samples: Number of Monte Carlo samples to generate
           alpha: Significance level for confidence intervals (default: 0.05)
           
       Returns:
           Tuple of (mean_pred, lower_bound, upper_bound)
       """
       # Make multiple predictions with dropout enabled
       predictions = []
       for _ in range(n_samples):
           # Keep dropout active during inference
           preds = K.function([self.model.input, K.learning_phase()], 
                             [self.model.output])(
                                 [X, 1])[0]  # 1 = training phase active
           predictions.append(preds)
       
       # Calculate statistics
       predictions = np.array(predictions)
       mean_pred = np.mean(predictions, axis=0)
       lower_bound = np.percentile(predictions, alpha/2 * 100, axis=0)
       upper_bound = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
       
       return mean_pred, lower_bound, upper_bound
   ```

4. **Visualization functions** have been updated to display uncertainty:
   ```python
   def plot_survival_curves_with_uncertainty(survival_probs, lower_bound, upper_bound, 
                                           time_grid, labels=None):
       # Plot step functions with confidence bands
       plt.step(time_grid, survival_probs)
       plt.fill_between(time_grid, lower_bound, upper_bound, alpha=0.3)
   ```

## Unified Constraint Framework for Uncertainty Bounds

A key innovation in DeepTimeR is the unified constraint framework that ensures all model outputs - including uncertainty bounds - satisfy mathematical properties required for valid probabilities:

### General Approach

1. **Unified Mathematical Framework**:
   - Multi-state models serve as the most general formulation
   - Survival analysis is treated as a 2-state special case
   - Competing risks is treated as a (k+1)-state special case

2. **Constraint Types Applied to Uncertainty Bounds**:
   - **Value range constraints**: 0 ≤ P(t) ≤ 1 for all probabilities and bounds
   - **Monotonicity constraints**: 
     - Survival probabilities are non-increasing
     - Cumulative incidence functions are non-decreasing
     - Transitions to absorbing states are non-decreasing
   - **Containment constraints**: lower_bound ≤ mean_prediction ≤ upper_bound
   - **Sum constraints**: 
     - Transition probabilities from each state sum to 1
     - Sum of competing risks CIFs ≤ 1
   - **Absorbing state properties**: Once in an absorbing state, you stay there

3. **Implementation via Projection**:
   - Isotonic regression is used for optimal projection onto the monotone cone
   - Implemented using the Pool-Adjacent-Violators Algorithm (PAVA)
   - This provides L2-optimal projections that minimally distort the original predictions

### Handling Model-Specific Constraints

The `ConstraintHandler` class provides specialized methods for each model type:

1. **Multi-state Models** (most general case):
   ```python
   def _apply_multistate_constraints(self, mean_pred, lower_bound, upper_bound, 
                                   state_structure):
       # Apply constraints to transition probability matrices with uncertainty
   ```

2. **Survival Analysis** (2-state special case):
   ```python
   def _apply_survival_constraints(self, mean_pred, lower_bound, upper_bound):
       # Apply constraints to survival probabilities with uncertainty
   ```

3. **Competing Risks** ((k+1)-state special case):
   ```python
   def _apply_competing_risks_constraints(self, mean_pred, lower_bound, upper_bound, 
                                        n_risks):
       # Apply constraints to cumulative incidence functions with uncertainty
   ```

### Special Handling for Uncertainty

The framework includes specialized handling for uncertainty:

1. **Uncertainty narrowing** at boundary points:
   - Uncertainty narrows to zero at t=0 for CIFs (where CIF(0)=0)
   - Uncertainty narrows to zero at t=0 for survival (where S(0)=1)
   - Uncertainty narrows as states become absorbing

2. **Prevention of crossing intervals** between different competing risks

3. **Smooth uncertainty bands** that respect all constraints

## Benefits of This Approach

1. **Theoretical soundness**: Constraints are applied using principled mathematical optimization
2. **Unified handling**: All model types use the same underlying constraint machinery
3. **Minimal distortion**: The L2-optimal projection preserves original predictions as much as possible
4. **Comprehensive constraints**: Ensures all mathematical properties are maintained across all models
5. **Uncertainty quantification**: Provides valid confidence bounds that respect all constraints

## Alternative Approaches

Other approaches to uncertainty quantification that could be implemented:

1. **Bootstrap Confidence Intervals**:
   - Train multiple models on bootstrap samples of the data
   - Generate predictions from each model for new data
   - Compute quantiles to create confidence bands

2. **Bayesian Neural Networks**:
   - Replace deterministic weights with distributions
   - Use variational inference or MCMC for training
   - Sample from posterior during prediction

3. **Direct Prediction of Parameters**:
   - Predict parameters of a distribution (e.g., Weibull)
   - Use these parameters to compute confidence intervals analytically