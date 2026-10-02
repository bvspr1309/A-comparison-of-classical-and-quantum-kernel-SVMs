r"""Batched, exact statevector simulation of Pauli-Z feature maps.

The feature maps studied here are the ones introduced by Havlicek et al. (2019) and
shipped in Qiskit as ``z_feature_map`` / ``zz_feature_map``:

.. math::

    |\phi(x)\rangle = \bigl[\, D(x)\, H^{\otimes n} \bigr]^{L} |0\rangle^{\otimes n},

where :math:`L` is the number of repetitions ("depth") and :math:`D(x)` is diagonal in
the computational basis. Writing a basis state as a bit string :math:`b`,

.. math::

    D(x)\,|b\rangle = e^{\,i\,\theta_b(x)}\,|b\rangle, \qquad
    \theta_b(x) = 2\sum_i x_i b_i
        + 2\!\!\sum_{(i,j) \in E}\!\! (\pi - x_i)(\pi - x_j)\,(b_i \oplus b_j),

with :math:`E` the set of entangled qubit pairs (empty for the Z map).

Because :math:`D(x)` is diagonal, one repetition costs a Walsh-Hadamard transform plus
an element-wise multiplication. This lets us simulate *all* data points at once with
NumPy, roughly two orders of magnitude faster than building one circuit per point,
which is what makes nested cross-validation over bandwidth, depth and seeds affordable.
The simulator is verified amplitude-by-amplitude against Qiskit in ``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

Entanglement = Literal["none", "linear", "full"]


@dataclass(frozen=True)
class FeatureMap:
    """A Pauli-Z feature map on ``n`` qubits, one qubit per input feature.

    Parameters
    ----------
    entanglement
        ``"none"``   - Z feature map (product state, no two-qubit gates).
        ``"linear"`` - ZZ feature map with nearest-neighbour pairs ``(i, i + 1)``.
        ``"full"``   - ZZ feature map with all pairs ``(i, j)``, ``i < j``.
    reps
        Number of repetitions :math:`L` of the ``H``-then-diagonal block.
    """

    entanglement: Entanglement
    reps: int = 1

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.entanglement not in ("none", "linear", "full"):
            raise ValueError(f"Unknown entanglement pattern: {self.entanglement!r}")
        if self.reps < 1:
            raise ValueError("reps must be a positive integer.")

    @property
    def name(self) -> str:
        """Short label used in result tables, e.g. ``"ZZ-full"``."""
        return "Z" if self.entanglement == "none" else f"ZZ-{self.entanglement}"

    def pairs(self, n_qubits: int) -> tuple[tuple[int, int], ...]:
        """Return the entangled qubit pairs for a register of ``n_qubits``."""
        if self.entanglement == "none":
            return ()
        if self.entanglement == "linear":
            return tuple((i, i + 1) for i in range(n_qubits - 1))
        return tuple(combinations(range(n_qubits), 2))


@cache
def _bit_table(n_qubits: int) -> FloatArray:
    """Return ``B[k, i]`` = value of qubit ``i`` in basis state ``k`` (little-endian).

    Qiskit orders basis states so that qubit 0 is the least-significant bit; we follow
    the same convention so amplitudes can be compared index-by-index.
    """
    indices = np.arange(2**n_qubits)[:, None]
    return ((indices >> np.arange(n_qubits)) & 1).astype(np.float64)


@cache
def _parity_table(n_qubits: int, pairs: tuple[tuple[int, int], ...]) -> FloatArray:
    """Return ``P[k, p]`` = ``b_i XOR b_j`` of basis state ``k`` for pair ``p = (i, j)``."""
    bits = _bit_table(n_qubits)
    left = bits[:, [i for i, _ in pairs]]
    right = bits[:, [j for _, j in pairs]]
    return np.abs(left - right)


def diagonal_phases(X: FloatArray, feature_map: FeatureMap) -> FloatArray:
    """Phase angles of :math:`D(x)` for every sample.

    Returns
    -------
    ndarray of shape ``(n_samples, 2**n_features)``
        ``theta[s, k]`` such that ``D(x_s)|k> = exp(1j * theta[s, k]) |k>``.
    """
    n_qubits = X.shape[1]
    theta = 2.0 * X @ _bit_table(n_qubits).T
    pairs = feature_map.pairs(n_qubits)
    if pairs:
        first = np.array([i for i, _ in pairs])
        second = np.array([j for _, j in pairs])
        couplings = (np.pi - X[:, first]) * (np.pi - X[:, second])
        theta += 2.0 * couplings @ _parity_table(n_qubits, pairs).T
    return theta


def _hadamard_all_qubits(states: ComplexArray, n_qubits: int) -> ComplexArray:
    """Apply ``H`` to every qubit of every state (a normalised Walsh-Hadamard transform)."""
    n_samples = states.shape[0]
    tensor = states.reshape((n_samples,) + (2,) * n_qubits)
    for axis in range(1, n_qubits + 1):
        zero = np.take(tensor, 0, axis=axis)
        one = np.take(tensor, 1, axis=axis)
        tensor = np.stack((zero + one, zero - one), axis=axis)
    return tensor.reshape(n_samples, -1) / np.sqrt(2.0**n_qubits)


def statevectors(X: FloatArray, feature_map: FeatureMap) -> ComplexArray:
    r"""Exact feature states :math:`|\phi(x)\rangle` for a batch of inputs.

    Parameters
    ----------
    X
        Array of shape ``(n_samples, n_features)``; one qubit is used per feature.
    feature_map
        The circuit family and its depth.

    Returns
    -------
    ndarray of shape ``(n_samples, 2**n_features)``
        Unit-norm statevectors in Qiskit's little-endian basis ordering.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n_samples, n_qubits = X.shape
    phases = np.exp(1j * diagonal_phases(X, feature_map))

    states = np.zeros((n_samples, 2**n_qubits), dtype=np.complex128)
    states[:, 0] = 1.0  # |0...0>
    for _ in range(feature_map.reps):
        states = _hadamard_all_qubits(states, n_qubits) * phases
    return states
