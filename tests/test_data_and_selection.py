import gzip

import numpy as np
import pytest

from qksvm.data import Preprocessor, RawSplit, make_binary_task, read_idx
from qksvm.feature_maps import FeatureMap
from qksvm.kernels import LinearKernel, QuantumKernel, RBFKernel
from qksvm.model_selection import cross_validate_C, evaluate_gram, evaluate_kernel, select_best


def fake_split(rng, n_per_class=60, n_pixels=20):
    labels = np.repeat(np.arange(10), n_per_class)
    images = rng.random((len(labels), n_pixels)) + labels[:, None] * 0.1
    return RawSplit(images=images, labels=labels)


def test_read_idx_round_trip(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = bytes([0, 0, 8, 3]) + b"".join(int(s).to_bytes(4, "big") for s in array.shape)
    path = tmp_path / "toy-idx3-ubyte.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(header + array.tobytes())
    np.testing.assert_array_equal(read_idx(path), array)


def test_binary_task_is_balanced_deterministic_and_disjoint(rng):
    train, test = fake_split(rng), fake_split(rng, n_per_class=30)
    task = make_binary_task(train, test, classes=(3, 6), n_train=40, seed=0)
    assert task.X_train.shape == (40, 20)
    assert set(np.unique(task.y_train)) == {-1, 1} and task.y_train.sum() == 0
    assert len(task.y_test) == 60 and task.y_test.sum() == 0  # all test images of both classes

    again = make_binary_task(train, test, classes=(3, 6), n_train=40, seed=0)
    np.testing.assert_array_equal(task.X_train, again.X_train)
    other = make_binary_task(train, test, classes=(3, 6), n_train=40, seed=1)
    assert not np.array_equal(task.X_train, other.X_train)
    np.testing.assert_array_equal(task.X_test, other.X_test)  # test set does not depend on seed

    with pytest.raises(ValueError):
        make_binary_task(train, test, classes=(3, 6), n_train=41, seed=0)


def test_preprocessor_is_fitted_on_training_data_only(rng):
    X_train, X_test = rng.normal(size=(50, 10)), rng.normal(size=(20, 10)) + 3.0
    preprocessor = Preprocessor(n_components=4).fit(X_train)
    Z_train = preprocessor.transform(X_train)
    np.testing.assert_allclose(Z_train.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z_train.std(axis=0), 1.0, atol=1e-10)
    # A test point is transformed identically whatever other test points accompany it.
    np.testing.assert_allclose(
        preprocessor.transform(X_test)[:1], preprocessor.transform(X_test[:1]), atol=1e-12
    )
    assert 0.0 < preprocessor.explained_variance_ratio <= 1.0


@pytest.fixture
def circles(rng):
    """Two concentric rings: not linearly separable, easy for an RBF kernel."""
    n = 240
    radius = np.where(np.arange(n) % 2 == 0, 0.5, 1.5)
    angle = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[radius * np.cos(angle), radius * np.sin(angle)] + rng.normal(scale=0.05, size=(n, 2))
    y = np.where(radius < 1.0, -1, 1)
    return X[:120], y[:120], X[120:], y[120:]


def test_pipeline_solves_a_nonlinear_problem(circles):
    X_train, y_train, X_test, y_test = circles
    rbf = evaluate_kernel(RBFKernel(1.0), X_train, y_train, X_test, y_test)
    linear = evaluate_kernel(LinearKernel(), X_train, y_train, X_test, y_test)
    quantum = evaluate_kernel(
        QuantumKernel(FeatureMap("none", 1), bandwidth=0.5), X_train, y_train, X_test, y_test
    )
    assert rbf.test_accuracy > 0.97
    assert quantum.test_accuracy > 0.97
    assert linear.test_accuracy < 0.75


def test_selection_uses_cross_validation_not_test_accuracy(circles):
    X_train, y_train, X_test, y_test = circles
    evaluations = [
        evaluate_kernel(kernel, X_train, y_train, X_test, y_test)
        for kernel in (LinearKernel(), RBFKernel(1.0))
    ]
    assert select_best(evaluations).kernel.family == "RBF"
    # Corrupting the test labels must not change which model is selected.
    corrupted = [
        evaluate_kernel(kernel, X_train, y_train, X_test, -y_test)
        for kernel in (LinearKernel(), RBFKernel(1.0))
    ]
    assert select_best(corrupted).kernel.family == "RBF"
    assert corrupted[1].cv_accuracy == evaluations[1].cv_accuracy


def test_cross_validation_is_deterministic(circles):
    X_train, y_train, _, _ = circles
    K = RBFKernel(1.0).gram(X_train)
    assert cross_validate_C(K, y_train, seed=3) == cross_validate_C(K, y_train, seed=3)


def test_fit_transform_touches_only_matrices_the_svm_is_fitted_on(circles):
    X_train, y_train, X_test, y_test = circles
    kernel = RBFKernel(1.0)
    K_train, K_test = kernel.gram(X_train), kernel.gram(X_test, X_train)
    seen = []

    def spy(K):
        seen.append(K.shape)
        return K

    plain = evaluate_gram(kernel, K_train, K_test, y_train, y_test, n_folds=5)
    spied = evaluate_gram(kernel, K_train, K_test, y_train, y_test, n_folds=5, fit_transform=spy)
    # Five fitting blocks of 96 points, then the full training matrix; all square.
    assert seen == [(96, 96)] * 5 + [(120, 120)]
    assert spied == plain  # an identity transform changes nothing
