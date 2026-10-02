"""One model-selection procedure for every kernel, classical or quantum.

Protocol
--------
1. Each candidate kernel (one point of its hyper-parameter grid) produces a training
   Gram matrix.
2. The SVM regularisation ``C`` is chosen for that kernel by stratified k-fold
   cross-validation *on the training set*, by slicing the Gram matrix.
3. Within a kernel family the candidate with the best cross-validation accuracy is
   selected. Ties are broken towards the earlier grid entry, i.e. deterministically.
4. Only then is the selected model scored on the held-out test set.

Test accuracy is also recorded for the non-selected candidates so that sensitivity
plots (accuracy vs. bandwidth or depth) can be drawn, but it is never used to make a
choice.

Note on preprocessing: PCA and standardisation are fitted once on the whole training
subsample and shared by the CV folds. They use no labels and never see test data, so
the held-out estimate is unbiased; the CV score itself is very slightly optimistic,
identically for every kernel, and is only used for ranking candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from qksvm.kernels import Kernel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: A map applied to a square training Gram matrix before the SVM is fitted on it.
GramTransform = Callable[[FloatArray], FloatArray]

#: Regularisation strengths searched for every kernel.
DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class Evaluation:
    """Outcome of tuning ``C`` for one kernel and scoring the refitted model."""

    kernel: Kernel
    C: float
    cv_accuracy: float
    train_accuracy: float
    test_accuracy: float
    n_support: int

    def as_record(self) -> dict[str, Any]:
        """Flatten into a dict suitable for a results table."""
        return {
            "family": self.kernel.family,
            **self.kernel.params(),
            "C": self.C,
            "cv_accuracy": self.cv_accuracy,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "n_support": self.n_support,
        }


def _svc(C: float) -> SVC:
    return SVC(kernel="precomputed", C=C, cache_size=500)


def cross_validate_C(
    K_train: FloatArray,
    y_train: IntArray,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    n_folds: int = 5,
    seed: int = 0,
    fit_transform: GramTransform | None = None,
) -> tuple[float, float]:
    """Choose ``C`` by stratified k-fold cross-validation on a precomputed Gram matrix.

    Parameters
    ----------
    K_train, y_train
        Training Gram matrix and labels.
    C_grid
        Candidate regularisation strengths.
    n_folds, seed
        Number of stratified folds and the seed of their shuffle.
    fit_transform
        Optional map applied to the fitting block of every fold (see
        :func:`evaluate_gram`). The validation rows are left untouched, exactly as
        test rows are.

    Returns
    -------
    best_C, best_cv_accuracy
        The first ``C`` in ``C_grid`` attaining the highest mean validation accuracy.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.zeros(len(C_grid))
    for fit_idx, val_idx in folds.split(K_train, y_train):
        K_fit = K_train[np.ix_(fit_idx, fit_idx)]
        if fit_transform is not None:
            K_fit = fit_transform(K_fit)
        K_val = K_train[np.ix_(val_idx, fit_idx)]
        for position, C in enumerate(C_grid):
            model = _svc(C).fit(K_fit, y_train[fit_idx])
            scores[position] += model.score(K_val, y_train[val_idx])
    scores /= n_folds
    best = int(np.argmax(scores))
    return float(C_grid[best]), float(scores[best])


def evaluate_gram(
    kernel: Kernel,
    K_train: FloatArray,
    K_test: FloatArray,
    y_train: IntArray,
    y_test: IntArray,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    n_folds: int = 5,
    seed: int = 0,
    fit_transform: GramTransform | None = None,
) -> Evaluation:
    """Tune ``C`` on given Gram matrices, refit on all training data, score on test.

    Taking the Gram matrices as arguments (rather than computing them from ``kernel``)
    lets the finite-shot experiments reuse this exact procedure on noisy estimates.

    ``fit_transform`` is applied to every square matrix an SVM is fitted on - the
    fitting block of each fold and the full training matrix - and never to
    validation or test rows. For finite-shot kernels it is the PSD projection: with
    this convention a validation row is a raw noisy estimate, like a test row, so the
    cross-validation score reflects how strongly the fitted model amplifies shot
    noise at prediction time.
    """
    C, cv_accuracy = cross_validate_C(K_train, y_train, C_grid, n_folds, seed, fit_transform)
    K_fit = K_train if fit_transform is None else fit_transform(K_train)
    model = _svc(C).fit(K_fit, y_train)
    return Evaluation(
        kernel=kernel,
        C=C,
        cv_accuracy=cv_accuracy,
        train_accuracy=float(model.score(K_train, y_train)),
        test_accuracy=float(model.score(K_test, y_test)),
        n_support=int(model.n_support_.sum()),
    )


def evaluate_kernel(
    kernel: Kernel,
    X_train: FloatArray,
    y_train: IntArray,
    X_test: FloatArray,
    y_test: IntArray,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    n_folds: int = 5,
    seed: int = 0,
) -> Evaluation:
    """Compute a kernel's Gram matrices and evaluate it with :func:`evaluate_gram`."""
    return evaluate_gram(
        kernel,
        kernel.gram(X_train),
        kernel.gram(X_test, X_train),
        y_train,
        y_test,
        C_grid,
        n_folds,
        seed,
    )


def select_best(evaluations: Sequence[Evaluation]) -> Evaluation:
    """Return the evaluation with the highest cross-validation accuracy.

    The test accuracy is deliberately ignored. Ties go to the earliest candidate.
    """
    if not evaluations:
        raise ValueError("No evaluations to select from.")
    return max(evaluations, key=lambda evaluation: evaluation.cv_accuracy)
