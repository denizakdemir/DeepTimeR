"""DeepTimeR: A deep learning framework for time-to-event analysis.

This package provides tools for:
- Deep learning models for time-to-event analysis
- Data preprocessing and handling
- Model evaluation and visualization
- Hyperparameter tuning
- Advanced evaluation metrics
- Advanced interpretability features
"""

__version__ = "0.1.0"

from .models import DeepTimeR
from .data import BaseData, SurvivalData, CompetingRisksData, MultiStateData
from .evaluation import ModelEvaluator, cross_validate
from .advanced_evaluation import AdvancedEvaluator
from .advanced_interpretability import AdvancedInterpreter
from .utils import (
    get_feature_importance,
    plot_feature_importance,
    plot_survival_curves,
    plot_cumulative_incidence,
    plot_state_occupation,
    plot_time_varying_effects
)
from .tuning import HyperparameterTuner

__all__ = [
    'DeepTimeR',
    'BaseData',
    'SurvivalData',
    'CompetingRisksData',
    'MultiStateData',
    'ModelEvaluator',
    'AdvancedEvaluator',
    'AdvancedInterpreter',
    'cross_validate',
    'get_feature_importance',
    'plot_feature_importance',
    'plot_survival_curves',
    'plot_cumulative_incidence',
    'plot_state_occupation',
    'plot_time_varying_effects',
    'HyperparameterTuner'
] 