"""End-to-end smoke test of one configuration on synthetic images."""

import numpy as np
import pytest

from qksvm.data import RawSplit
from qksvm.experiments import Config, Grids, kernel_from_record, run_config

GRIDS = Grids(
    C=(0.1, 10.0),
    rbf_gamma=(0.1, 1.0),
    poly_degree=(2,),
    poly_gamma=(0.1,),
    bandwidth=(0.1, 1.0),
    reps=(1, 2),
)


def synthetic_split(rng, n_per_class):
    labels = np.repeat(np.arange(10), n_per_class)
    prototypes = rng.normal(size=(10, 30))
    images = prototypes[labels] + rng.normal(scale=0.8, size=(len(labels), 30))
    return RawSplit(images=images, labels=labels)


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(0)
    train, test = synthetic_split(rng, 60), synthetic_split(rng, 40)
    config = Config(n_features=3, n_train=60, seed=0, n_folds=3, grids=GRIDS)
    return run_config(config, train, test), run_config(config, train, test)


def test_sweep_has_one_row_per_candidate(tables):
    sweep = tables[0]["sweep"]
    assert len(sweep) == len(GRIDS.candidates()) == 1 + 2 + 1 + 3 * 2 * 2
    assert {row["family"] for row in sweep} == {
        "Linear",
        "RBF",
        "Polynomial",
        "Z",
        "ZZ-linear",
        "ZZ-full",
    }
    for row in sweep:
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert -1.0 <= row["alignment"] <= 1.0
        assert row["C"] in GRIDS.C


def test_geometry_and_shots_tables(tables):
    geometry, shots = tables[0]["geometry"], tables[0]["shots"]
    quantum = [row for row in geometry if row["family"] != "RBF"]
    assert len(quantum) == 3 * len(GRIDS.reps) * 3  # families x depths x ridge values
    assert all(row["geometric_difference"] > 0 for row in quantum)
    assert all(row["model_complexity"] > 0 for row in geometry)
    assert {row["shots"] for row in shots} == {10, 30, 100, 300, 1000, 3000, 10000, np.inf}


def test_run_is_reproducible(tables):
    first, second = tables
    assert first == second


def test_kernels_can_be_rebuilt_from_their_records(tables):
    rebuilt = [kernel_from_record(row) for row in tables[0]["sweep"]]
    assert rebuilt == GRIDS.candidates()
