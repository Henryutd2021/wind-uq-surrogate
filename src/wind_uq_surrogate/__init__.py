"""UQ metrics and surrogate models for plant-level wind power uncertainty (framework).

Modules: ``metrics_single``, ``metrics_pair``, ``metrics_ensemble``, ``metrics_spatial`` (UQ metrics);
``probabilistic`` (predictive distributions and scores); ``calibration`` (Step A); ``features`` (Step B);
``surrogate`` (Step C); ``validation`` (Step D); ``workflow`` (Steps A-D end to end); ``synthetic``, ``io``,
``cli``. See README.md and docs/.
"""

from .calibration import (IsotonicRecalibrator, block_labels, crps_optimal_scale, optimal_scale,
                          rolling_optimal_scale)
from .features import (FeatureSpec, PlantSeries, SpatialSpec, build_features, calibration_targets, feature_columns,
                       prediction_table, training_table)
from .io import read_plants, split_observed, write_plants
from .metrics_ensemble import (crps_ensemble, ensemble_metrics, predictability_index, rank_histogram,
                               rank_histogram_summary, spread_index, total_spread_index)
from .metrics_pair import (agreement_metrics, moving_correlation, moving_mape, moving_nmae, moving_nrmse)
from .metrics_single import (moving_spectral_entropy, moving_std, normalized_range, single_series_metrics,
                             spectral_entropy, turbulence_intensity)
from .metrics_spatial import empirical_variogram, fit_variogram, pairwise_distances, variogram_model, variogram_over_time
from .probabilistic import (available_coverages, crps, crps_gaussian, crps_laplace, default_levels, interval_scores,
                            pinball_loss, quantiles, reliability)
from .surrogate import Surrogate, select_gpr_kernel
from .synthetic import synthetic_fleet
from .validation import cross_validate, plant_splits, score_predictive, score_scale, time_split
from .workflow import WorkflowConfig, run_workflow, save_result

__version__ = "1.0.0"

__all__ = [
    "FeatureSpec",
    "IsotonicRecalibrator",
    "PlantSeries",
    "SpatialSpec",
    "Surrogate",
    "WorkflowConfig",
    "agreement_metrics",
    "available_coverages",
    "block_labels",
    "build_features",
    "calibration_targets",
    "cross_validate",
    "crps",
    "crps_ensemble",
    "crps_gaussian",
    "crps_laplace",
    "crps_optimal_scale",
    "default_levels",
    "empirical_variogram",
    "ensemble_metrics",
    "feature_columns",
    "fit_variogram",
    "interval_scores",
    "moving_correlation",
    "moving_mape",
    "moving_nmae",
    "moving_nrmse",
    "moving_spectral_entropy",
    "moving_std",
    "normalized_range",
    "optimal_scale",
    "pairwise_distances",
    "pinball_loss",
    "plant_splits",
    "predictability_index",
    "prediction_table",
    "quantiles",
    "rank_histogram",
    "rank_histogram_summary",
    "read_plants",
    "reliability",
    "rolling_optimal_scale",
    "run_workflow",
    "save_result",
    "score_predictive",
    "score_scale",
    "select_gpr_kernel",
    "single_series_metrics",
    "spectral_entropy",
    "split_observed",
    "spread_index",
    "synthetic_fleet",
    "time_split",
    "total_spread_index",
    "training_table",
    "turbulence_intensity",
    "variogram_model",
    "variogram_over_time",
    "write_plants",
]
