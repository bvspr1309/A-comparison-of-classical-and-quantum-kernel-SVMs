import numpy as np
import pandas as pd
import pytest

from qksvm.analysis import mean_and_ci, paired_difference, select_by_cv, summarise


@pytest.fixture
def sweep():
    """Two seeds, two families, two candidates each; the test-best is never the CV-best."""
    rows = []
    for seed in (0, 1):
        for family, shift in (("RBF", 0.0), ("Z", 0.01)):
            rows.append(
                dict(seed=seed, family=family, reps=1, cv_accuracy=0.90, test_accuracy=0.80 + shift)
            )
            rows.append(
                dict(seed=seed, family=family, reps=2, cv_accuracy=0.85, test_accuracy=0.99)
            )
    return pd.DataFrame(rows).assign(n_features=4, n_train=64)


def test_selection_ignores_test_accuracy(sweep):
    selected = select_by_cv(sweep)
    assert len(selected) == 4  # one winner per (seed, family)
    assert (selected["reps"] == 1).all()
    assert selected["test_accuracy"].max() < 0.99


def test_selection_within_depth_keeps_every_depth(sweep):
    assert len(select_by_cv(sweep, within=("family", "reps"))) == len(sweep)


def test_ties_keep_first_candidate_in_grid_order(sweep):
    tied = sweep.assign(cv_accuracy=0.9)
    assert (select_by_cv(tied)["reps"] == 1).all()


def test_mean_and_ci_matches_textbook_value():
    mean, half_width = mean_and_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    # t_{0.975, 3} = 3.1824; standard error = sqrt(5/3) / 2
    assert half_width == pytest.approx(3.1824 * np.sqrt(5 / 3) / 2, rel=1e-4)
    assert np.isnan(mean_and_ci([1.0])[1])


def test_summarise_and_paired_difference(sweep):
    selected = select_by_cv(sweep)
    summary = summarise(selected, by=("family",))
    assert summary.set_index("family").loc["Z", "mean"] == pytest.approx(0.81)
    assert (summary["n"] == 2).all()

    paired = paired_difference(selected, "Z").iloc[0]
    assert paired["mean_difference"] == pytest.approx(0.01)
    assert (paired["wins"], paired["ties"], paired["losses"]) == (2, 0, 0)
