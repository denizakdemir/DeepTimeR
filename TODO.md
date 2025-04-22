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

## Next Steps

### Model Improvements
- [ ] Add support for custom loss functions in model compilation
- [ ] Implement gradient clipping to prevent exploding gradients

### Evaluation & Interpretability
- [ ] Implement concordance index calculation
- [ ] Add calibration metrics
- [ ] Implement prediction error curves

### Documentation
- [ ] Create usage examples for each analysis type
- [ ] Document model architecture and design decisions