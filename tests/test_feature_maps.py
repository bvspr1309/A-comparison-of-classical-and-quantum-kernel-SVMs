"""The batched simulator must reproduce Qiskit's circuits amplitude-by-amplitude."""

import numpy as np
import pytest

from qksvm.feature_maps import FeatureMap, statevectors

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit.library import z_feature_map, zz_feature_map  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402


def qiskit_statevector(x: np.ndarray, feature_map: FeatureMap) -> np.ndarray:
    """Reference state from Qiskit's own feature-map circuits."""
    n_qubits = len(x)
    if feature_map.entanglement == "none":
        circuit = z_feature_map(n_qubits, reps=feature_map.reps)
    else:
        circuit = zz_feature_map(
            n_qubits, reps=feature_map.reps, entanglement=feature_map.entanglement
        )
    return Statevector(circuit.assign_parameters(x)).data


@pytest.mark.parametrize("entanglement", ["none", "linear", "full"])
@pytest.mark.parametrize("reps", [1, 2, 3])
@pytest.mark.parametrize("n_qubits", [2, 3, 5])
def test_matches_qiskit_amplitudes(entanglement, reps, n_qubits, rng):
    feature_map = FeatureMap(entanglement, reps)
    X = rng.normal(scale=1.5, size=(4, n_qubits))
    ours = statevectors(X, feature_map)
    for row, x in zip(ours, X, strict=True):
        np.testing.assert_allclose(row, qiskit_statevector(x, feature_map), atol=1e-12)


@pytest.mark.parametrize("entanglement", ["none", "linear", "full"])
def test_states_are_normalised(entanglement, rng):
    states = statevectors(rng.normal(size=(10, 6)), FeatureMap(entanglement, reps=4))
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)


def test_entangling_pairs():
    assert FeatureMap("none").pairs(4) == ()
    assert FeatureMap("linear").pairs(4) == ((0, 1), (1, 2), (2, 3))
    assert len(FeatureMap("full").pairs(5)) == 10


def test_single_sample_is_accepted(rng):
    assert statevectors(rng.normal(size=3), FeatureMap("full")).shape == (1, 8)


@pytest.mark.parametrize(
    "kwargs", [{"entanglement": "circular"}, {"entanglement": "full", "reps": 0}]
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FeatureMap(**kwargs)
