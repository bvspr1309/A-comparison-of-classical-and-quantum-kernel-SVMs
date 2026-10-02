"""Experiment definitions: hyper-parameter grids and the per-configuration runners.

A *configuration* is one ``(n_features, n_train, seed)`` triple. :func:`run_config` is
a pure function of the configuration and the raw data: it returns plain records
(dicts) and holds no global state, so configurations can be executed in parallel and
re-run in isolation.

Three tables are produced per configuration:

``sweep``
    Every candidate kernel: cross-validated, training and test accuracy plus
    diagnostics of its training Gram matrix.
``geometry``
    Geometric difference to the tuned RBF kernel and model complexity, for the
    bandwidth-tuned quantum kernel at each depth.
``shots``
    Test accuracy when the selected quantum kernels are estimated from a finite
    number of measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qksvm.data import BinaryTask, Preprocessor, RawSplit, make_binary_task
from qksvm.feature_maps import FeatureMap
from qksvm.kernels import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    QuantumKernel,
    RBFKernel,
    project_to_psd,
    sample_shot_noise,
)
from qksvm.metrics import (
    geometric_difference,
    kernel_target_alignment,
    model_complexity,
    off_diagonal_statistics,
)
from qksvm.model_selection import (
    DEFAULT_C_GRID,
    Evaluation,
    GramTransform,
    evaluate_gram,
    select_best,
)

Record = dict[str, Any]

QUANTUM_FAMILIES: tuple[str, ...] = ("Z", "ZZ-linear", "ZZ-full")
_ENTANGLEMENT = {"Z": "none", "ZZ-linear": "linear", "ZZ-full": "full"}


@dataclass(frozen=True)
class Grids:
    """Hyper-parameter grids. Every kernel family is tuned with the same procedure.

    The RBF, polynomial and (per depth) quantum grids have 9, 8 and 8 kernel
    candidates respectively. When depth is selected as well, a quantum family has
    ``len(reps)`` times more candidates than a classical one, i.e. the protocol gives
    the quantum kernels the larger tuning budget.
    """

    C: tuple[float, ...] = DEFAULT_C_GRID
    rbf_gamma: tuple[float, ...] = tuple(float(g) for g in np.logspace(-3, 1, 9))
    poly_degree: tuple[int, ...] = (2, 3)
    poly_gamma: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)
    bandwidth: tuple[float, ...] = tuple(float(c) for c in 2.0 ** np.arange(-7, 1))
    reps: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def candidates(self) -> list[Kernel]:
        """Enumerate every kernel that is evaluated in the sweep.

        Quantum candidates are ordered by depth and then bandwidth, so ties in
        cross-validation accuracy are resolved towards the shallower circuit.
        """
        kernels: list[Kernel] = [LinearKernel()]
        kernels += [RBFKernel(gamma) for gamma in self.rbf_gamma]
        kernels += [
            PolynomialKernel(degree, gamma)
            for degree in self.poly_degree
            for gamma in self.poly_gamma
        ]
        kernels += [
            QuantumKernel(FeatureMap(_ENTANGLEMENT[family], reps), bandwidth)
            for family in QUANTUM_FAMILIES
            for reps in self.reps
            for bandwidth in self.bandwidth
        ]
        return kernels


@dataclass(frozen=True)
class Config:
    """One experimental configuration."""

    n_features: int
    n_train: int
    seed: int
    classes: tuple[int, int] = (3, 6)  # Dress vs. Shirt, as in Huang et al. (2021)
    n_folds: int = 5
    grids: Grids = field(default_factory=Grids)

    def tags(self) -> Record:
        """Columns that identify this configuration in a results table."""
        return {"n_features": self.n_features, "n_train": self.n_train, "seed": self.seed}


@dataclass(frozen=True)
class Prepared:
    """A task after PCA/standardisation, ready to be fed to kernels."""

    Z_train: np.ndarray
    Z_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    explained_variance: float


def prepare(config: Config, train: RawSplit, test: RawSplit) -> Prepared:
    """Draw the training subsample for ``config`` and fit the preprocessing on it."""
    task: BinaryTask = make_binary_task(train, test, config.classes, config.n_train, config.seed)
    preprocessor = Preprocessor(config.n_features, seed=config.seed).fit(task.X_train)
    return Prepared(
        Z_train=preprocessor.transform(task.X_train),
        Z_test=preprocessor.transform(task.X_test),
        y_train=task.y_train,
        y_test=task.y_test,
        explained_variance=preprocessor.explained_variance_ratio,
    )


def _evaluate(
    kernel: Kernel,
    K_train: np.ndarray,
    K_test: np.ndarray,
    data: Prepared,
    config: Config,
    fit_transform: GramTransform | None = None,
) -> Evaluation:
    return evaluate_gram(
        kernel,
        K_train,
        K_test,
        data.y_train,
        data.y_test,
        config.grids.C,
        config.n_folds,
        config.seed,
        fit_transform,
    )


def kernel_from_record(record: Mapping[str, Any]) -> Kernel:
    """Rebuild the kernel described by one row of the sweep table."""
    family = record["family"]
    if family == "Linear":
        return LinearKernel()
    if family == "RBF":
        return RBFKernel(float(record["gamma"]))
    if family == "Polynomial":
        return PolynomialKernel(int(record["degree"]), float(record["gamma"]))
    feature_map = FeatureMap(_ENTANGLEMENT[family], int(record["reps"]))
    return QuantumKernel(feature_map, float(record["bandwidth"]))


def run_sweep(config: Config, data: Prepared) -> tuple[list[Evaluation], list[Record]]:
    """Evaluate every candidate kernel on one configuration.

    Returns the evaluations (used afterwards for model selection) and one record per
    kernel with its hyper-parameters, accuracies and Gram-matrix diagnostics.
    """
    evaluations, records = [], []
    for kernel in config.grids.candidates():
        K_train = kernel.gram(data.Z_train)
        evaluation = _evaluate(
            kernel, K_train, kernel.gram(data.Z_test, data.Z_train), data, config
        )
        mean, variance = off_diagonal_statistics(K_train)
        evaluations.append(evaluation)
        records.append(
            {
                **config.tags(),
                **evaluation.as_record(),
                "alignment": kernel_target_alignment(K_train, data.y_train),
                "offdiag_mean": mean,
                "offdiag_var": variance,
                "explained_variance": data.explained_variance,
            }
        )
    return evaluations, records


def select_family(
    evaluations: Sequence[Evaluation], family: str, reps: int | None = None
) -> Evaluation:
    """Cross-validation winner within one family, optionally at a fixed depth."""
    return select_best(
        [
            evaluation
            for evaluation in evaluations
            if evaluation.kernel.family == family
            and (reps is None or evaluation.kernel.params().get("reps") == reps)
        ]
    )


def run_geometry(
    config: Config,
    data: Prepared,
    evaluations: Sequence[Evaluation],
    regs: Sequence[float] = (1e-3, 1e-2, 1e-1),
) -> list[Record]:
    r"""Geometric difference to the tuned RBF kernel, and model complexity, per depth.

    For every quantum family and depth, the cross-validated bandwidth is used; the
    resulting Gram matrix is compared with the cross-validated RBF Gram matrix via
    :math:`g(K_{\mathrm{RBF}} \Vert K_Q)`.
    """
    rbf = select_family(evaluations, "RBF").kernel
    K_rbf = rbf.gram(data.Z_train)
    base = {**config.tags(), "sqrt_n": float(np.sqrt(config.n_train))}

    records: list[Record] = [
        {
            **base,
            "family": "RBF",
            "reps": 0,
            "reg": reg,
            "geometric_difference": 1.0,
            "model_complexity": model_complexity(K_rbf, data.y_train, reg),
        }
        for reg in regs
    ]
    for family in QUANTUM_FAMILIES:
        for reps in config.grids.reps:
            kernel = select_family(evaluations, family, reps).kernel
            K_quantum = kernel.gram(data.Z_train)
            records += [
                {
                    **base,
                    "family": family,
                    "reps": reps,
                    "bandwidth": kernel.params()["bandwidth"],
                    "reg": reg,
                    "geometric_difference": geometric_difference(K_rbf, K_quantum, reg),
                    "model_complexity": model_complexity(K_quantum, data.y_train, reg),
                }
                for reg in regs
            ]
    return records


def run_shots(
    config: Config,
    data: Prepared,
    evaluations: Sequence[Evaluation],
    shots_grid: Sequence[int] = (10, 30, 100, 300, 1000, 3000, 10000),
) -> list[Record]:
    """Accuracy of each quantum family when kernel entries are estimated from shots.

    Depth and bandwidth are fixed to the values selected with the exact kernel. For
    each shot budget, both the training Gram matrix and the test-train kernel are
    replaced by finite-shot estimates and ``C`` is re-tuned by cross-validation on the
    noisy matrix, i.e. the learner only ever sees what a (gate-noise-free) device
    would return. Matrices an SVM is fitted on are projected onto the PSD cone;
    validation and test rows are used as measured.
    """
    rng = np.random.default_rng([config.seed, config.n_features, config.n_train])
    records: list[Record] = []
    for family in QUANTUM_FAMILIES:
        exact = select_family(evaluations, family)
        kernel = exact.kernel
        K_train = kernel.gram(data.Z_train)
        K_test = kernel.gram(data.Z_test, data.Z_train)
        base = {**config.tags(), "family": family, **kernel.params()}
        records.append(
            {**base, "shots": np.inf, "C": exact.C, "test_accuracy": exact.test_accuracy}
        )
        for shots in shots_grid:
            noisy = _evaluate(
                kernel,
                sample_shot_noise(K_train, shots, rng, symmetric=True),
                sample_shot_noise(K_test, shots, rng, symmetric=False),
                data,
                config,
                fit_transform=project_to_psd,
            )
            records.append(
                {**base, "shots": shots, "C": noisy.C, "test_accuracy": noisy.test_accuracy}
            )
    return records


def run_config(config: Config, train: RawSplit, test: RawSplit) -> dict[str, list[Record]]:
    """Run the sweep and both follow-up analyses for one configuration."""
    data = prepare(config, train, test)
    evaluations, sweep = run_sweep(config, data)
    return {
        "sweep": sweep,
        "geometry": run_geometry(config, data, evaluations),
        "shots": run_shots(config, data, evaluations),
    }
