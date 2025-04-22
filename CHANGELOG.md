# Changelog

## 0.1.0 (2025-04-22)

### Added
- Initial release of DeepTimeR
- Added support for survival analysis, competing risks, and multi-state modeling
- Implemented feature attention mechanism for interpretability
- Added uncertainty quantification using Monte Carlo Dropout
- Implemented unified probabilistic constraint framework
- Added advanced evaluation metrics (concordance index, Brier score, calibration metrics)
- Added advanced interpretability features (SHAP values, LIME explanations, partial dependence plots)
- Added gradient clipping to prevent exploding gradients
- Added support for custom loss functions
- Added comprehensive documentation and examples

### Changed
- Standardized model interfaces for different time-to-event analysis tasks
- Unified constraint handling for different analysis types
- Improved model evaluation with prediction error curves and calibration metrics

### Fixed
- Fixed issues with time-varying covariate handling
- Addressed numerical stability issues in loss functions
- Improved constraints for valid probability predictions