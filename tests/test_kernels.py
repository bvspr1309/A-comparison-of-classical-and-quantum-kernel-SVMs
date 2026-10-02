import numpy as np
import pytest

from qksvm.feature_maps import FeatureMap
from qksvm.kernels import (
    LinearKernel,
    PolynomialKernel,
    QuantumKernel,
    RBFKernel,
    project_to_psd,
    sample_shot_noise,
)

MAPS = [FeatureMap("none", 1), FeatureMap("linear", 2), FeatureMap("full", 3)]


@pytest.mark.parametrize("feature_map", MAPS)
def test_gram_matrix_is_a_valid_kernel(feature_map, rng):
    K = QuantumKernel(feature_map, bandwidth=0.7).gram(rng.normal(size=(30, 4)))
    np.testing.assert_array_equal(K, K.T)
    np.testing.assert_array_equal(np.diag(K), 1.0)
    assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-10  # positive semidefinite


def test_cross_gram_is_consistent_with_square_gram(rng):
    kernel = QuantumKernel(FeatureMap("full", 2), bandwidth=0.3)
    X = rng.normal(size=(12, 4))
    np.testing.assert_allclose(kernel.gram(X, X), kernel.gram(X), atol=1e-12)
    assert kernel.gram(X[:5], X).shape == (5, 12)


def test_product_map_has_closed_form(rng):
    """For the depth-1 Z map the kernel factorises: k = prod_i cos^2(c (x_i - y_i))."""
    c = 0.4
    XA, XB = rng.normal(size=(6, 5)), rng.normal(size=(7, 5))
    expected = np.prod(np.cos(c * (XA[:, None, :] - XB[None, :, :])) ** 2, axis=-1)
    actual = QuantumKernel(FeatureMap("none", 1), bandwidth=c).gram(XA, XB)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zero_bandwidth_gives_constant_kernel(rng):
    K = QuantumKernel(FeatureMap("full", 2), bandwidth=0.0).gram(rng.normal(size=(5, 3)))
    np.testing.assert_allclose(K, 1.0, atol=1e-12)


@pytest.mark.parametrize("feature_map", MAPS)
def test_matches_qiskit_machine_learning(feature_map, rng):
    """Agreement with the library implementation researchers would otherwise use."""
    pytest.importorskip("qiskit_machine_learning")
    from qiskit.circuit.library import z_feature_map, zz_feature_map
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    n_qubits, bandwidth = 4, 0.6
    if feature_map.entanglement == "none":
        circuit = z_feature_map(n_qubits, reps=feature_map.reps)
    else:
        circuit = zz_feature_map(
            n_qubits, reps=feature_map.reps, entanglement=feature_map.entanglement
        )
    XA, XB = rng.normal(size=(5, n_qubits)), rng.normal(size=(6, n_qubits))
    reference = FidelityStatevectorKernel(feature_map=circuit).evaluate(
        bandwidth * XA, bandwidth * XB
    )
    actual = QuantumKernel(feature_map, bandwidth).gram(XA, XB)
    np.testing.assert_allclose(actual, reference, atol=1e-10)


def test_classical_kernels_match_their_formulas(rng):
    XA, XB = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    squared_distance = ((XA[:, None] - XB[None]) ** 2).sum(-1)
    np.testing.assert_allclose(RBFKernel(0.3).gram(XA, XB), np.exp(-0.3 * squared_distance))
    np.testing.assert_allclose(PolynomialKernel(3, 0.2).gram(XA, XB), (0.2 * XA @ XB.T + 1) ** 3)
    np.testing.assert_allclose(LinearKernel().gram(XA, XB), XA @ XB.T)


def test_shot_noise_is_unbiased_with_binomial_variance(rng):
    K = np.full((200, 200), 0.3)
    shots = 50
    estimate = sample_shot_noise(K, shots, rng, symmetric=False)
    assert abs(estimate.mean() - 0.3) < 0.005
    assert abs(estimate.var() - 0.3 * 0.7 / shots) < 5e-4


def test_symmetric_shot_noise_keeps_structure(rng):
    K = QuantumKernel(FeatureMap("full", 1), 0.5).gram(rng.normal(size=(20, 3)))
    estimate = sample_shot_noise(K, 100, rng, symmetric=True)
    np.testing.assert_array_equal(estimate, estimate.T)
    np.testing.assert_array_equal(np.diag(estimate), 1.0)


def test_shot_noise_vanishes_for_many_shots(rng):
    K = QuantumKernel(FeatureMap("linear", 1), 0.5).gram(rng.normal(size=(15, 3)))
    estimate = sample_shot_noise(K, 10**8, rng, symmetric=True)
    np.testing.assert_allclose(estimate, K, atol=1e-3)


def test_binomial_model_matches_compute_uncompute_circuit(rng):
    """P(all zeros) of U(y)^dagger U(x)|0> equals the kernel value we sample from."""
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import zz_feature_map
    from qiskit.primitives import StatevectorSampler

    n_qubits, shots = 3, 20_000
    x, y = rng.normal(scale=0.5, size=(2, n_qubits))
    encode = zz_feature_map(n_qubits, reps=1, entanglement="full")
    circuit = encode.assign_parameters(x).compose(encode.assign_parameters(y).inverse())
    circuit.measure_all()
    counts = (
        StatevectorSampler(seed=7).run([circuit], shots=shots).result()[0].data.meas.get_counts()
    )
    frequency = counts.get("0" * n_qubits, 0) / shots

    exact = QuantumKernel(FeatureMap("full", 1)).gram(x[None], y[None])[0, 0]
    standard_error = np.sqrt(exact * (1 - exact) / shots)
    assert abs(frequency - exact) < 5 * standard_error + 1e-9


def test_psd_projection():
    K = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])  # indefinite
    assert np.linalg.eigvalsh(K).min() < 0
    projected = project_to_psd(K)
    assert np.linalg.eigvalsh(projected).min() > -1e-12
    np.testing.assert_allclose(project_to_psd(projected), projected, atol=1e-12)  # idempotent
