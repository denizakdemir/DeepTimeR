# Constraints for Survival Probabilities and CIFs

To ensure that the survival probabilities and cumulative incidence functions (CIFs) in DeepTimeR are mathematically valid and interpretable, they should satisfy several key constraints:

## Survival Probability Constraints

1. **Monotonicity**:
   - Survival probabilities must be monotonically decreasing over time
   - For any time points t₁ < t₂, S(t₁) ≥ S(t₂)

2. **Value Range**:
   - Must always be between 0 and 1: 0 ≤ S(t) ≤ 1

3. **Starting Value**:
   - At time t=0, survival probability must equal 1: S(0) = 1

4. **Endpoint Behavior**:
   - As t approaches the maximum follow-up time, survival probability should approach a non-negative value, potentially 0 for complete follow-up

5. **Step Function Behavior**:
   - In discrete-time models, survival curves should be right-continuous step functions

## Cumulative Incidence Function Constraints

1. **Monotonicity**:
   - Each cause-specific CIF must be monotonically increasing
   - For any time points t₁ < t₂, F_j(t₁) ≤ F_j(t₂)

2. **Value Range**:
   - Each CIF must be between 0 and 1: 0 ≤ F_j(t) ≤ 1

3. **Starting Value**:
   - At time t=0, all CIFs must equal 0: F_j(0) = 0

4. **Sum Constraint**:
   - The sum of all cause-specific CIFs must never exceed 1
   - ∑_j F_j(t) ≤ 1 for all t

5. **Relationship with Survival**:
   - In competing risks, overall survival plus sum of all CIFs must equal 1
   - S(t) + ∑_j F_j(t) = 1

## Uncertainty Interval Constraints

When adding uncertainty quantification, these additional constraints should apply:

1. **Interval Containment**:
   - Uncertainty intervals (lower bound L(t) and upper bound U(t)) must contain the point estimate
   - L(t) ≤ S(t) ≤ U(t) or L(t) ≤ F_j(t) ≤ U(t)

2. **Bound Monotonicity**:
   - Lower and upper bounds must maintain the same monotonicity properties as the point estimates

3. **Bound Ranges**:
   - 0 ≤ L(t) ≤ U(t) ≤ 1 for both survival probabilities and CIFs

4. **Crossing Prevention**:
   - Uncertainty intervals for different risks should ideally not cross in ways that violate logical constraints

5. **Narrowing at Extremes**:
   - Uncertainty intervals should narrow at t=0 (where S(0)=1 and F_j(0)=0)

6. **Widening with Prediction Distance**:
   - Uncertainty should generally increase as predictions extend further from observed data

These constraints ensure that both point estimates and uncertainty measures maintain mathematical validity while providing meaningful clinical interpretation for time-to-event outcomes.


# Issues with Monotonicity in DeepTimeR: Theoretical Review

Upon examining the DeepTimeR implementation, I've identified theoretical issues that could lead to monotonicity violations in survival probabilities and CIFs.

## Theoretical Framework and Issues

### Survival Probability Calculation

The implementation uses a discrete-time formulation where:

```python
return tf.math.cumprod(1 - hazard_probs, axis=1)
```

In theory, survival probability S(t) = ∏ᵢ₌₁ᵗ (1 - h(i)) should be monotonically decreasing. However, this implementation has a fundamental issue:

1. **Unconstrained Hazard Predictions**: While hazards are constrained between 0-1 using sigmoid activation, there's no constraint on the *pattern* of hazards across time intervals.

2. **Independent Interval Prediction**: Each time interval's hazard is predicted independently, allowing for predictions where h(t₂) << h(t₁) for t₂ > t₁.

3. **Temporal Smoothness Limitation**: The temporal smoothness regularization encourages smoothness but doesn't enforce monotonicity:
   ```python
   diff = y_pred[:, 1:] - y_pred[:, :-1]
   smoothness_loss = tf.reduce_mean(tf.square(diff))
   ```
   This penalizes large changes between consecutive points but doesn't ensure hazards follow a pattern that produces monotonic survival curves.

### CIF Calculation Issues

For competing risks, the implementation computes:
```python
overall_survival = tf.math.cumprod(1 - tf.reduce_sum(risk_hazards, axis=2), axis=1)
return tf.expand_dims(overall_survival, axis=2) * risk_hazards
```

This approach:
1. Doesn't ensure CIFs are monotonically increasing
2. Doesn't guarantee that individual risk-specific CIFs are properly monotonic

## Correction Approaches

To enforce the constraints we identified earlier, the implementation needs to be modified:

1. **Cumulative Hazard Modeling**:
   - Model cumulative hazards H(t) instead of interval-specific hazards
   - Ensure H(t) is monotonically increasing using constraints
   - Derive S(t) = exp(-H(t))

2. **Isotonic Neural Networks**:
   - Use specialized neural network architectures with monotonicity constraints
   - Apply increasing constraints for hazards/CIFs and decreasing constraints for survival

3. **Post-processing**:
   - Apply isotonic regression to enforce monotonicity after prediction
   - For CIFs, ensure their sum never exceeds 1 while maintaining monotonicity

4. **Structured Time Dependence**:
   - Instead of predicting each time interval independently, model time sequentially
   - Use recurrent neural networks (RNNs) with positivity constraints on outputs

5. **Parametric Approach**:
   - Have the network predict parameters of established survival distributions
   - For example, predict shape and scale parameters of a Weibull distribution
   - This inherently satisfies monotonicity constraints

The current temporal smoothness loss could be replaced with a directional penalty that specifically enforces monotonicity rather than just smoothness.

Implementing these changes would ensure that DeepTimeR produces predictions that satisfy the mathematical constraints required for valid survival and competing risks analysis.