# DeepTimeR TODO List

## Completed Tasks

### Unified Probabilistic Constraint Framework
- [x] Refactor constraint handling to use multi-state models as the unified framework
- [x] Implement a general constraint solver based on constrained optimization principles
- [x] Express survival and competing risks constraints as special cases of MSM constraints
- [x] Implement proper isotonic regression as a projection onto the monotone cone
- [x] Add theoretical documentation on the unified constraint framework
- [x] Create comprehensive tests for the unified constraint system

### Core Features
- [x] Implement basic time-varying covariate support
- [x] Add temporal smoothness regularization
- [x] Implement multi-state modeling
- [x] Add competing risks analysis
- [x] Create base data handling classes
- [x] Create time-varying data tests
- [x] Create multi-state model tests
- [x] Create data handling tests
- [x] Implement basic test infrastructure

### Model Improvements
- [x] Add support for custom loss functions in model compilation
- [x] Implement gradient clipping to prevent exploding gradients

### Evaluation & Interpretability
- [x] Implement concordance index calculation
- [x] Add calibration metrics
- [x] Implement prediction error curves

### Documentation
- [x] Create usage examples for each analysis type
- [x] Document model architecture and design decisions
- [x] Add MIT license for public release
- [x] Update package metadata for PyPI distribution
- [x] Create CHANGELOG.md to track version changes

## Next Steps

### Performance Improvements
- [ ] Implement custom training loop with early stopping
- [ ] Add TensorFlow profiling for performance optimization
- [ ] Implement batch caching for better memory usage

### Advanced Features
- [ ] Add support for landmark analysis 
- [ ] Implement joint modeling for longitudinal and time-to-event data
- [ ] Add recurrent neural network architecture for time-varying covariates
- [ ] Implement transfer learning capabilities from pretrained models

### Documentation & Examples
- [ ] Create detailed documentation website with Sphinx
- [ ] Create step-by-step tutorials for major analysis types
- [ ] Add case studies with real-world datasets
- [ ] Create interactive visualizations for model outputs

### Distribution & Integration
- [ ] Add continuous integration with GitHub Actions
- [ ] Create Docker container for easy deployment
- [ ] Implement API for integration with other frameworks
- [ ] Add compatibility with scikit-survival and lifelines for model comparison