"""Post-processing of the raw result tables: model selection and summary statistics.

Everything here operates on the ``sweep`` table written by
``scripts/run_experiments.py`` (one row per configuration and candidate kernel).
Model selection is repeated from the stored cross-validation scores, so the figures
and tables can never pick a model by its test accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Columns identifying one configuration (one training subsample).
CONFIG_COLUMNS: tuple[str, ...] = ("n_features", "n_train", "seed")

CLASSICAL_FAMILIES: tuple[str, ...] = ("Linear", "Polynomial", "RBF")
QUANTUM_FAMILIES: tuple[str, ...] = ("Z", "ZZ-linear", "ZZ-full")
FAMILIES: tuple[str, ...] = CLASSICAL_FAMILIES + QUANTUM_FAMILIES


def select_by_cv(sweep: pd.DataFrame, within: Sequence[str] = ("family",)) -> pd.DataFrame:
    """Keep, per configuration and per group, the candidate with the best CV accuracy.

    Parameters
    ----------
    sweep
        The raw sweep table.
    within
        Columns defining the groups a winner is chosen in, e.g. ``("family",)`` to
        tune depth and bandwidth jointly or ``("family", "reps")`` to tune the
        bandwidth separately at every depth.

    Returns
    -------
    DataFrame
        One row per ``(configuration, group)``. Ties keep the first candidate in grid
        order, matching :func:`qksvm.model_selection.select_best`.
    """
    keys = [*CONFIG_COLUMNS, *within]
    winners = sweep.groupby(keys, sort=False, dropna=False)["cv_accuracy"].idxmax()
    return sweep.loc[winners].reset_index(drop=True)


def mean_and_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval of the mean.

    The interval describes variability across training subsamples (seeds). It does
    not include the sampling error of the fixed test set, which is shared by, and
    therefore cancels in comparisons between, kernels.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float("nan")
    half_width = stats.sem(values) * stats.t.ppf(0.5 + confidence / 2.0, len(values) - 1)
    return float(values.mean()), float(half_width)


def summarise(frame: pd.DataFrame, by: Sequence[str], value: str = "test_accuracy") -> pd.DataFrame:
    """Aggregate ``value`` over seeds: mean, standard deviation, 95% CI and count."""
    rows = []
    for key, group in frame.groupby(list(by), sort=False, dropna=False):
        mean, half_width = mean_and_ci(group[value])
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "mean": mean,
                "std": float(group[value].std(ddof=1)) if len(group) > 1 else float("nan"),
                "ci95": half_width,
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def paired_difference(
    selected: pd.DataFrame, family: str, baseline: str = "RBF", value: str = "test_accuracy"
) -> pd.DataFrame:
    """Per-seed difference ``family - baseline`` on identical data, summarised.

    Both models are trained on the same subsample and scored on the same test set, so
    the difference removes the (large) seed-to-seed variation common to all kernels.
    The p-value is from a two-sided Wilcoxon signed-rank test across seeds.
    """
    index = list(CONFIG_COLUMNS)
    pivot = selected.pivot(index=index, columns="family", values=value)
    difference = (pivot[family] - pivot[baseline]).rename("difference").reset_index()

    rows = []
    for (n_features, n_train), group in difference.groupby(["n_features", "n_train"]):
        delta = group["difference"].to_numpy()
        mean, half_width = mean_and_ci(delta)
        p_value = stats.wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
        rows.append(
            {
                "n_features": n_features,
                "n_train": n_train,
                "family": family,
                "baseline": baseline,
                "mean_difference": mean,
                "ci95": half_width,
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "p_wilcoxon": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def haar_fidelity_moments(n_qubits: int) -> tuple[float, float]:
    r"""Mean and variance of :math:`|\langle\psi|\phi\rangle|^2` for Haar-random states.

    For dimension :math:`d = 2^n` the fidelity follows a Beta(1, d - 1) distribution,
    so the mean is :math:`1/d` and the variance :math:`(d-1) / (d^2 (d+1))`. A feature
    map whose kernel values reach these moments is as expressive, and as concentrated,
    as a random circuit.
    """
    dim = 2.0**n_qubits
    return 1.0 / dim, (dim - 1.0) / (dim**2 * (dim + 1.0))
