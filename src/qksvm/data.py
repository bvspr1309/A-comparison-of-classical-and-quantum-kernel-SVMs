"""Fashion-MNIST loading, binary-task construction and leakage-free preprocessing.

The dataset is downloaded once from the official Zalando Research repository and every
file is verified against a pinned SHA-256 digest, so a run can never silently use
different data.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

#: File name -> SHA-256 digest of the gzip archive.
_FILES: dict[str, str] = {
    "train-images-idx3-ubyte.gz": (
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"
    ),
    "train-labels-idx1-ubyte.gz": (
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"
    ),
    "t10k-images-idx3-ubyte.gz": (
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5"
    ),
}

CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class RawSplit:
    """One official Fashion-MNIST split: flattened images in [0, 1] and integer labels."""

    images: FloatArray  # shape (n_samples, 784)
    labels: IntArray  # shape (n_samples,), values in 0..9


@dataclass(frozen=True)
class BinaryTask:
    """A two-class problem with a training subsample and a held-out test set.

    Labels are encoded as -1 / +1, the convention used by the kernel-alignment and
    model-complexity formulas.
    """

    X_train: FloatArray
    y_train: IntArray
    X_test: FloatArray
    y_test: IntArray


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(name: str, root: Path) -> Path:
    """Return the local path of ``name``, downloading and verifying it if necessary."""
    path = root / name
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BASE_URL + name, path)
    digest = _sha256(path)
    if digest != _FILES[name]:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {_FILES[name]}, got {digest}. "
            "Delete the file and re-run to download it again."
        )
    return path


def read_idx(path: Path) -> NDArray[np.uint8]:
    """Parse a gzip-compressed IDX file (the MNIST container format) into an array.

    The header is a big-endian magic number whose last byte is the number of
    dimensions, followed by one 4-byte big-endian size per dimension.
    """
    with gzip.open(path, "rb") as handle:
        buffer = handle.read()
    n_dims = buffer[3]
    shape = tuple(int.from_bytes(buffer[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims))
    return np.frombuffer(buffer, dtype=np.uint8, offset=4 + 4 * n_dims).reshape(shape)


def load_fashion_mnist(root: str | Path = "data/raw") -> tuple[RawSplit, RawSplit]:
    """Load the official train (60 000) and test (10 000) splits of Fashion-MNIST."""
    root = Path(root)
    splits = []
    for prefix in ("train", "t10k"):
        images = read_idx(_fetch(f"{prefix}-images-idx3-ubyte.gz", root))
        labels = read_idx(_fetch(f"{prefix}-labels-idx1-ubyte.gz", root))
        splits.append(
            RawSplit(
                images=images.reshape(len(images), -1).astype(np.float64) / 255.0,
                labels=labels.astype(np.int64),
            )
        )
    return splits[0], splits[1]


def make_binary_task(
    train: RawSplit,
    test: RawSplit,
    classes: tuple[int, int],
    n_train: int,
    seed: int,
) -> BinaryTask:
    """Build a balanced two-class task.

    The training set is a class-balanced random subsample of the official training
    split, drawn with ``seed``. The test set is *every* image of the two classes in the
    official test split, so it is identical across seeds and never used for fitting.

    Parameters
    ----------
    train, test
        The official splits returned by :func:`load_fashion_mnist`.
    classes
        ``(negative, positive)`` Fashion-MNIST class indices, mapped to -1 and +1.
    n_train
        Total number of training points (must be even; half per class).
    seed
        Seed of the subsampling generator.
    """
    if n_train % 2:
        raise ValueError("n_train must be even so that the two classes are balanced.")
    rng = np.random.default_rng(seed)
    negative, positive = classes

    chosen = np.concatenate(
        [
            rng.choice(np.flatnonzero(train.labels == c), size=n_train // 2, replace=False)
            for c in (negative, positive)
        ]
    )
    rng.shuffle(chosen)
    test_mask = np.isin(test.labels, classes)

    def encode(labels: IntArray) -> IntArray:
        return np.where(labels == positive, 1, -1).astype(np.int64)

    return BinaryTask(
        X_train=train.images[chosen],
        y_train=encode(train.labels[chosen]),
        X_test=test.images[test_mask],
        y_test=encode(test.labels[test_mask]),
    )


class Preprocessor:
    """PCA to ``n_components`` followed by per-component standardisation.

    Both steps are *fitted on training data only* and then applied unchanged to test
    data. After the transform each feature has zero mean and unit variance on the
    training set, which gives every kernel (classical or quantum) inputs on the same
    scale; kernel-specific length scales are handled by each kernel's own bandwidth.
    """

    def __init__(self, n_components: int, seed: int = 0) -> None:
        self.n_components = n_components
        self._pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
        self._scaler = StandardScaler()

    def fit(self, X: FloatArray) -> Preprocessor:
        """Fit PCA and the scaler on training images."""
        self._scaler.fit(self._pca.fit_transform(X))
        return self

    def transform(self, X: FloatArray) -> FloatArray:
        """Project images onto the fitted principal components and standardise."""
        return self._scaler.transform(self._pca.transform(X))

    @property
    def explained_variance_ratio(self) -> float:
        """Fraction of training-set pixel variance retained by the projection."""
        return float(self._pca.explained_variance_ratio_.sum())
