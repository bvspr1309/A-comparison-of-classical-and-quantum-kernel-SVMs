import numpy as np
import pytest

from qksvm.kernels import RBFKernel
from qksvm.metrics import (
    geometric_difference,
    kernel_target_alignment,
    model_complexity,
    off_diagonal_statistics,
)


@pytest.fixture
def labelled(rng):
    X = rng.normal(size=(40, 3))
    y = np.where(X[:, 0] > 0, 1, -1)
    return X, y


@pytest.mark.parametrize("centered", [True, False])
def test_ideal_kernel_has_alignment_one(labelled, centered):
    _, y = labelled
    assert kernel_target_alignment(np.outer(y, y), y, centered) == pytest.approx(1.0)


def test_alignment_is_bounded_and_scale_invariant(labelled):
    X, y = labelled
    K = RBFKernel(0.5).gram(X)
    alignment = kernel_target_alignment(K, y)
    assert -1.0 <= alignment <= 1.0
    assert kernel_target_alignment(7.0 * K, y) == pytest.approx(alignment)


def test_uninformative_kernels_have_zero_centered_alignment(labelled):
    _, y = labelled
    y = np.array([1, -1] * 20)  # exactly balanced
    assert kernel_target_alignment(np.ones((40, 40)), y) == 0.0
    assert kernel_target_alignment(np.eye(40), y) == pytest.approx(1 / np.sqrt(39), abs=1e-12)


def test_geometric_difference_of_a_kernel_with_itself_is_one(labelled):
    X, _ = labelled
    K = RBFKernel(2.0).gram(X) + 1e-3 * np.eye(len(X))
    assert geometric_difference(K, K) == pytest.approx(1.0, abs=1e-6)
    assert geometric_difference(K, 5.0 * K) == pytest.approx(1.0, abs=1e-6)  # trace-normalised


def test_geometric_difference_bounds_the_complexity_ratio(labelled, rng):
    """Defining property (Huang et al. 2021): s_1 <= g(K_1 || K_2)^2 * s_2 for any labels."""
    X, _ = labelled
    K_1 = RBFKernel(2.0).gram(X) + 1e-3 * np.eye(len(X))
    K_2 = RBFKernel(0.2).gram(X) + 1e-3 * np.eye(len(X))
    g = geometric_difference(K_1, K_2)
    for _ in range(20):
        y = rng.choice([-1, 1], size=len(X))
        assert model_complexity(K_1, y) <= g**2 * model_complexity(K_2, y) * (1 + 1e-8)


def test_regularised_geometric_difference_converges_and_guards_singularity(labelled):
    X, _ = labelled
    K = RBFKernel(2.0).gram(X) + 1e-3 * np.eye(len(X))
    other = RBFKernel(0.3).gram(X)
    exact = geometric_difference(K, other)
    assert geometric_difference(K, other, reg=1e-10) == pytest.approx(exact, rel=1e-4)
    assert geometric_difference(K, other, reg=1.0) < exact
    with pytest.raises(np.linalg.LinAlgError):
        geometric_difference(np.ones((5, 5)), np.eye(5))


def test_model_complexity_limits(labelled):
    _, y = labelled
    n = len(y)
    assert model_complexity(np.eye(n), y) == pytest.approx(n)  # memorising kernel
    ideal = np.outer(y, y) + 1e-9 * np.eye(n)
    assert model_complexity(ideal, y) == pytest.approx(1.0, rel=1e-4)  # labels are 1 direction


def test_off_diagonal_statistics():
    K = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    mean, variance = off_diagonal_statistics(K)
    assert mean == pytest.approx(0.4)
    assert variance == pytest.approx(np.var([0.2, 0.4, 0.6]))


def test_haar_moments_match_random_states(rng):
    """Closed-form Haar moments agree with sampled random states."""
    from qksvm.analysis import haar_fidelity_moments

    n_qubits, n_states = 4, 600
    states = rng.normal(size=(n_states, 2**n_qubits)) + 1j * rng.normal(
        size=(n_states, 2**n_qubits)
    )
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    fidelities = np.abs(states.conj() @ states.T) ** 2
    mean, variance = off_diagonal_statistics(fidelities)
    expected_mean, expected_variance = haar_fidelity_moments(n_qubits)
    assert mean == pytest.approx(expected_mean, rel=0.02)
    assert variance == pytest.approx(expected_variance, rel=0.05)
