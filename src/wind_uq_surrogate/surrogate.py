"""Step C of the workflow: surrogate regressors from UQ-metric features to the calibrated uncertainty scale.

All models share one interface (:class:`Surrogate`): inputs are standardized, the target is the per-unit scale,
by default regressed on its logarithm (the scale is positive and right-skewed). Gaussian process regression
(Rasmussen and Williams, 2006) is the main surrogate because it also returns its own predictive uncertainty;
support vector regression, random forests, ridge regression and a constant baseline are comparators. All are
scikit-learn estimators (Pedregosa et al., 2011). Details: ``docs/surrogate_workflow.md``, Section 3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SURROGATES = ("gpr", "svr", "random_forest", "ridge", "constant")
GPR_KERNELS = ("rbf", "matern32", "matern52", "rational_quadratic")


def gpr_kernel(n_features: int, kind: str = "matern52", ard: bool = True):
    """Kernel ``C * k(x, x') + white noise``; ``ard=True`` gives one length scale per feature (not available for
    the rational quadratic kernel)."""
    ls = np.ones(n_features) if ard else 1.0
    bounds = (1e-2, 1e3)
    if kind == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=bounds)
    elif kind == "matern32":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5)
    elif kind == "matern52":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5)
    elif kind == "rational_quadratic":
        base = RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=bounds, alpha_bounds=(1e-2, 1e3))
    else:
        raise ValueError(f"kind must be one of {GPR_KERNELS}")
    return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(0.1, (1e-6, 1e1))


def _estimator(name: str, n_features: int, random_state, params: dict):
    p = dict(params)
    if name == "gpr":
        kernel = p.pop("kernel", None) or gpr_kernel(n_features, p.pop("kernel_kind", "matern52"), p.pop("ard", True))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=p.pop("n_restarts", 0),
                                        random_state=random_state, **p)
    if name == "svr":
        return SVR(**{"C": 1.0, "epsilon": 0.1, "gamma": "scale", **p})
    if name == "random_forest":
        return RandomForestRegressor(**{"n_estimators": 200, "min_samples_leaf": 5, "random_state": random_state, **p})
    if name == "ridge":
        return Ridge(**{"alpha": 1.0, **p})
    if name == "constant":
        return DummyRegressor(**{"strategy": "median", **p})
    raise ValueError(f"unknown surrogate {name!r}; choose from {SURROGATES}")


class Surrogate:
    """A surrogate from features to the uncertainty scale.

    Parameters
    ----------
    name : one of :data:`SURROGATES`.
    log_target : regress ``log(scale)``; predictions are ``exp`` of the predicted log (the median of a log-normal
        predictive distribution of the scale).
    max_train_samples : fit on a random subset of at most this many rows (exact GP regression costs O(n^3)).
    random_state : seed of the subset and of the estimators.
    **params : passed to the scikit-learn estimator (for ``"gpr"`` also ``kernel``, ``kernel_kind``, ``ard``,
        ``n_restarts``).
    """

    def __init__(self, name: str = "gpr", log_target: bool = True, max_train_samples: int | None = None,
                 random_state=0, **params):
        if name not in SURROGATES:
            raise ValueError(f"unknown surrogate {name!r}; choose from {SURROGATES}")
        self.name = name
        self.log_target = log_target
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.params = params
        self.pipeline_ = None
        self.feature_names_ = None
        self.fit_warnings_: list[str] = []

    @staticmethod
    def _matrix(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if not np.isfinite(arr).all():
            raise ValueError("X must be finite (drop rows with missing features first)")
        return arr

    def fit(self, X, y) -> "Surrogate":
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        Xa = self._matrix(X)
        ya = np.asarray(y, dtype=float)
        if ya.shape != (Xa.shape[0],):
            raise ValueError("y must have one value per row of X")
        if not np.isfinite(ya).all() or (self.log_target and np.any(ya <= 0)):
            raise ValueError("y must be finite (and positive with log_target)")
        if self.max_train_samples is not None and Xa.shape[0] > self.max_train_samples:
            rows = np.random.default_rng(self.random_state).choice(Xa.shape[0], self.max_train_samples, replace=False)
            Xa, ya = Xa[np.sort(rows)], ya[np.sort(rows)]
        target = np.log(ya) if self.log_target else ya
        est = _estimator(self.name, Xa.shape[1], self.random_state, self.params)
        self.pipeline_ = Pipeline([("scale", StandardScaler()), ("model", est)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.pipeline_.fit(Xa, target)
        self.fit_warnings_ = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for w in caught:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return self

    def _check_fitted(self):
        if self.pipeline_ is None:
            raise RuntimeError("fit the surrogate first")

    def predict(self, X) -> np.ndarray:
        """Predicted scale (same unit as the training target)."""
        self._check_fitted()
        out = self.pipeline_.predict(self._matrix(X))
        return np.exp(out) if self.log_target else out

    def predict_log_std(self, X) -> np.ndarray:
        """GP regression only: posterior standard deviation of the (log-)target, i.e. the surrogate's own
        uncertainty about the scale."""
        self._check_fitted()
        if self.name != "gpr":
            raise ValueError("only the GP surrogate returns a predictive standard deviation")
        Xs = self.pipeline_.named_steps["scale"].transform(self._matrix(X))
        _, std = self.pipeline_.named_steps["model"].predict(Xs, return_std=True)
        return std

    @property
    def log_marginal_likelihood_(self) -> float:
        self._check_fitted()
        if self.name != "gpr":
            raise ValueError("only the GP surrogate has a marginal likelihood")
        return float(self.pipeline_.named_steps["model"].log_marginal_likelihood_value_)

    @property
    def kernel_(self):
        self._check_fitted()
        if self.name != "gpr":
            raise ValueError("only the GP surrogate has a kernel")
        return self.pipeline_.named_steps["model"].kernel_


def select_gpr_kernel(X, y, kinds=GPR_KERNELS, ard: bool = True, **kwargs) -> tuple[str, pd.DataFrame]:
    """Fit a GP surrogate with each kernel in ``kinds`` and rank them by log marginal likelihood (same data and
    target transform, so the values are comparable). Returns the best kind and the ranking table."""
    rows = []
    for kind in kinds:
        model = Surrogate("gpr", kernel_kind=kind, ard=ard, **kwargs).fit(X, y)
        rows.append({"kernel_kind": kind, "log_marginal_likelihood": model.log_marginal_likelihood_,
                     "fitted_kernel": str(model.kernel_), "warnings": len(model.fit_warnings_)})
    table = pd.DataFrame(rows).sort_values("log_marginal_likelihood", ascending=False).reset_index(drop=True)
    return str(table.loc[0, "kernel_kind"]), table
