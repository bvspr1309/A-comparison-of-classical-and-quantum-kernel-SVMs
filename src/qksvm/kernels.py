r"""Classical and quantum kernels behind one interface.

Every kernel is a small immutable object that maps two sets of inputs to a Gram
matrix. The SVM is always trained on the *precomputed* Gram matrix, so classical and
quantum models share exactly the same optimiser and model-selection code path; the
kernel function is the only thing that differs between them.

The quantum kernel is the fidelity kernel of Havlicek et al. (2019),

.. math::

    k(x, x') = \bigl| \langle \phi(c\,x) \,|\, \phi(c\,x') \rangle \bigr|^{2},

where :math:`c` is a bandwidth that rescales the inputs before encoding. Treating
:math:`c` as a tunable hyper-parameter (Shaydulin & Wild 2022; Canatar et al. 2022) is
the quantum analogue of tuning :math:`\gamma` for the RBF kernel and is required for
a fair comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

from qksvm.feature_maps import ComplexArray, FeatureMap, statevectors

FloatArray = NDArray[np.float64]


class Kernel(Protocol):
    """Anything that can produce a Gram matrix and describe itself."""

    @property
    def family(self) -> str:
        """Name of the kernel family, e.g. ``"RBF"`` or ``"ZZ-full"``."""
        ...

    def gram(self, XA: FloatArray, XB: FloatArray | None = None) -> FloatArray:
        """Return ``K[i, j] = k(XA[i], XB[j])``; ``XB=None`` means ``XB = XA``."""
        ...

    def params(self) -> dict[str, Any]:
        """Hyper-parameters of this kernel, for logging."""
        ...


@dataclass(frozen=True)
class LinearKernel:
    r""":math:`k(x, x') = x^\top x'`. A sanity baseline: no feature map at all."""

    family: str = "Linear"

    def gram(self, XA: FloatArray, XB: FloatArray | None = None) -> FloatArray:
        """Return the Gram matrix between ``XA`` and ``XB``."""
        return linear_kernel(XA, XB)

    def params(self) -> dict[str, Any]:
        """Return the (empty) hyper-parameter set."""
        return {}


@dataclass(frozen=True)
class RBFKernel:
    r""":math:`k(x, x') = \exp(-\gamma \lVert x - x' \rVert^2)`."""

    gamma: float
    family: str = "RBF"

    def gram(self, XA: FloatArray, XB: FloatArray | None = None) -> FloatArray:
        """Return the Gram matrix between ``XA`` and ``XB``."""
        return rbf_kernel(XA, XB, gamma=self.gamma)

    def params(self) -> dict[str, Any]:
        """Return the hyper-parameters."""
        return {"gamma": self.gamma}


@dataclass(frozen=True)
class PolynomialKernel:
    r""":math:`k(x, x') = (\gamma\, x^\top x' + 1)^{p}`."""

    degree: int
    gamma: float
    family: str = "Polynomial"

    def gram(self, XA: FloatArray, XB: FloatArray | None = None) -> FloatArray:
        """Return the Gram matrix between ``XA`` and ``XB``."""
        return polynomial_kernel(XA, XB, degree=self.degree, gamma=self.gamma, coef0=1.0)

    def params(self) -> dict[str, Any]:
        """Return the hyper-parameters."""
        return {"degree": self.degree, "gamma": self.gamma}


@dataclass(frozen=True)
class QuantumKernel:
    """Fidelity kernel of a Pauli-Z feature map, computed by exact simulation.

    Parameters
    ----------
    feature_map
        Circuit family and depth.
    bandwidth
        Scale ``c`` applied to the inputs before encoding (``x -> c * x``).
    """

    feature_map: FeatureMap
    bandwidth: float = 1.0

    @property
    def family(self) -> str:
        """Name of the feature-map family."""
        return self.feature_map.name

    def states(self, X: FloatArray) -> ComplexArray:
        """Return the feature states of ``X`` after bandwidth scaling."""
        return statevectors(self.bandwidth * np.asarray(X, dtype=np.float64), self.feature_map)

    def gram(self, XA: FloatArray, XB: FloatArray | None = None) -> FloatArray:
        """Return the exact (infinite-shot) Gram matrix between ``XA`` and ``XB``."""
        states_a = self.states(XA)
        if XB is None:
            K = fidelity(states_a, states_a)
            np.fill_diagonal(K, 1.0)  # remove ~1e-16 round-off on k(x, x) = 1
            return (K + K.T) / 2.0
        return fidelity(states_a, self.states(XB))

    def params(self) -> dict[str, Any]:
        """Return the hyper-parameters."""
        return {"bandwidth": self.bandwidth, **asdict(self.feature_map)}


def fidelity(states_a: ComplexArray, states_b: ComplexArray) -> FloatArray:
    r"""Pairwise state fidelities :math:`|\langle a_i | b_j \rangle|^2`."""
    overlaps = states_a.conj() @ states_b.T
    return overlaps.real**2 + overlaps.imag**2


def sample_shot_noise(
    K: FloatArray,
    shots: int,
    rng: np.random.Generator,
    symmetric: bool,
) -> FloatArray:
    r"""Simulate estimating a fidelity kernel with a finite number of measurements.

    On hardware, :math:`k(x, x')` is estimated with the compute-uncompute circuit
    :math:`U^\dagger(x')\,U(x)\,|0\rangle` as the fraction of ``shots`` that return the
    all-zeros bit string. That count is exactly ``Binomial(shots, k(x, x'))``, so
    sampling from this distribution reproduces the measurement statistics of an ideal
    (gate-noise-free) device without simulating each circuit. ``tests/`` checks this
    against Qiskit's sampler.

    Parameters
    ----------
    K
        Exact kernel values in ``[0, 1]``.
    shots
        Measurements per kernel entry.
    rng
        Random generator (pass a seeded one for reproducibility).
    symmetric
        If ``True``, ``K`` is a square training Gram matrix: each pair is estimated
        once and mirrored, and the diagonal is left at its exact value of 1 (the
        compute-uncompute circuit for ``x = x'`` is the identity).
    """
    probabilities = np.clip(K, 0.0, 1.0)
    estimate = rng.binomial(shots, probabilities) / shots
    if symmetric:
        upper = np.triu(estimate, k=1)
        estimate = upper + upper.T
        np.fill_diagonal(estimate, 1.0)
    return estimate


def project_to_psd(K: FloatArray) -> FloatArray:
    """Return the nearest (Frobenius-norm) positive-semidefinite matrix to ``K``.

    A finite-shot Gram matrix is symmetric but generally not PSD, which makes the SVM
    dual non-convex. Clipping negative eigenvalues to zero is the standard remedy.
    """
    eigenvalues, eigenvectors = np.linalg.eigh((K + K.T) / 2.0)
    return (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
