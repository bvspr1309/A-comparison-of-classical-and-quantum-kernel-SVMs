r"""Label-aware and label-free diagnostics of a Gram matrix.

These quantities explain *why* a kernel generalises (or fails to), independently of
the SVM that is trained on it:

* **kernel-target alignment** - how well the kernel's geometry matches the labels;
* **geometric difference** and **model complexity** (Huang et al. 2021) - whether a
  quantum kernel even has room to outperform a classical one on some labelling, and
  how complex a function the given labels are under each kernel;
* **concentration** (Thanasilp et al. 2024) - whether off-diagonal kernel values
  collapse to a constant, which makes the Gram matrix uninformative and, on hardware,
  exponentially expensive to resolve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _center(K: FloatArray) -> FloatArray:
    """Center a Gram matrix in feature space: ``H K H`` with ``H = I - 11^T / n``."""
    row_mean = K.mean(axis=0, keepdims=True)
    col_mean = K.mean(axis=1, keepdims=True)
    return K - row_mean - col_mean + K.mean()


def kernel_target_alignment(K: FloatArray, y: NDArray[np.int64], centered: bool = True) -> float:
    r"""Alignment between a Gram matrix and the ideal kernel :math:`y y^\top`.

    .. math::

        A(K, y) = \frac{\langle K, y y^\top \rangle_F}
                       {\lVert K \rVert_F \, \lVert y y^\top \rVert_F}

    With ``centered=True`` both matrices are centered first (Cortes et al. 2012), which
    removes the spurious alignment that any kernel with a large constant offset has
    with balanced labels. Values lie in ``[-1, 1]``; higher means the kernel already
    separates the classes.

    Parameters
    ----------
    K
        Square Gram matrix of the training set.
    y
        Labels in ``{-1, +1}``.
    centered
        Use centered alignment (recommended) instead of the original definition of
        Cristianini et al. (2001).
    """
    y = np.asarray(y, dtype=np.float64)
    target = np.outer(y, y)
    if centered:
        K, target = _center(K), _center(target)
    denominator = np.linalg.norm(K) * np.linalg.norm(target)
    if denominator == 0.0:  # e.g. a constant kernel after centering
        return 0.0
    return float(np.sum(K * target) / denominator)


def _trace_normalise(K: FloatArray) -> FloatArray:
    """Rescale ``K`` so that ``Tr(K) = n``, the convention of Huang et al. (2021)."""
    return K * (K.shape[0] / np.trace(K))


def _psd_eigh(K: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Eigendecomposition of a PSD matrix with round-off negatives clipped to zero."""
    eigenvalues, eigenvectors = np.linalg.eigh((K + K.T) / 2.0)
    return np.clip(eigenvalues, 0.0, None), eigenvectors


def geometric_difference(K_1: FloatArray, K_2: FloatArray, reg: float = 0.0) -> float:
    r"""Geometric difference :math:`g(K^1 \,\Vert\, K^2)` of Huang et al. (2021), Eq. 5.

    .. math::

        g(K^1 \Vert K^2) = \sqrt{\bigl\lVert \sqrt{K^2}\,(K^1)^{-1}\sqrt{K^2}
                                 \bigr\rVert_\infty},

    with both kernels normalised to :math:`\mathrm{Tr}(K) = N` and
    :math:`\lVert\cdot\rVert_\infty` the spectral norm. It bounds the ratio of model
    complexities for *any* labelling, :math:`s_{K^1} \le g^2\, s_{K^2}`. With
    :math:`K^1` classical and :math:`K^2` quantum, a small :math:`g` guarantees the
    classical kernel is competitive; :math:`g \sim \sqrt{N}` is necessary (not
    sufficient) for a quantum prediction advantage.

    For ``reg > 0`` the inverse is replaced by
    :math:`(K^1 + \lambda I)^{-1} K^1 (K^1 + \lambda I)^{-1}`, the operator whose
    quadratic form in the labels is the squared norm of the ridge regressor (see
    :func:`model_complexity`), so the bound above continues to hold between the
    regularised complexity of :math:`K^1` and the unregularised one of :math:`K^2`.
    It reduces to the formula above as :math:`\lambda \to 0` and discards directions
    of :math:`K^1` with eigenvalue far below :math:`\lambda`, which a regularised
    learner cannot use and which would otherwise dominate the inverse numerically.

    Parameters
    ----------
    K_1, K_2
        Square Gram matrices on the same ``N`` training points.
    reg
        Ridge parameter :math:`\lambda \ge 0` (after trace normalisation).
    """
    K_1, K_2 = _trace_normalise(K_1), _trace_normalise(K_2)

    eigenvalues_1, vectors_1 = _psd_eigh(K_1)
    if reg == 0.0:
        if eigenvalues_1.min() <= 1e-12 * eigenvalues_1.max():
            raise np.linalg.LinAlgError("K_1 is numerically singular; pass reg > 0.")
        spectrum = 1.0 / eigenvalues_1
    else:
        spectrum = eigenvalues_1 / (eigenvalues_1 + reg) ** 2
    inverse_1 = (vectors_1 * spectrum) @ vectors_1.T

    eigenvalues_2, vectors_2 = _psd_eigh(K_2)
    sqrt_2 = (vectors_2 * np.sqrt(eigenvalues_2)) @ vectors_2.T

    return float(np.sqrt(np.linalg.norm(sqrt_2 @ inverse_1 @ sqrt_2, ord=2)))


def model_complexity(K: FloatArray, y: NDArray[np.int64], reg: float = 0.0) -> float:
    r"""Model complexity :math:`s_K(N)` of Huang et al. (2021), Eq. 4.

    .. math::

        s_K(N) = y^\top (K + \lambda I)^{-1} K\, (K + \lambda I)^{-1} y
        \;\xrightarrow{\lambda \to 0}\; y^\top K^{-1} y ,

    i.e. the squared norm of the (ridge) kernel regressor fitted to the labels. The
    prediction error of that regressor is bounded by :math:`c\sqrt{s_K(N)/N}`, so a
    smaller value means the labels are a "simpler" function under this kernel. ``K``
    is normalised to :math:`\mathrm{Tr}(K) = N` before the computation.
    """
    y = np.asarray(y, dtype=np.float64)
    eigenvalues, eigenvectors = _psd_eigh(_trace_normalise(K))
    projections = eigenvectors.T @ y
    if reg == 0.0:
        spectrum = 1.0 / eigenvalues
    else:
        spectrum = eigenvalues / (eigenvalues + reg) ** 2
    return float(np.sum(spectrum * projections**2))


def off_diagonal_statistics(K: FloatArray) -> tuple[float, float]:
    """Mean and variance of the off-diagonal entries of a square Gram matrix.

    A variance that shrinks towards zero as the circuit grows is the signature of
    (exponential) kernel concentration: all pairs of distinct points look equally
    similar, so the Gram matrix carries no information about the data.
    """
    off_diagonal = K[~np.eye(K.shape[0], dtype=bool)]
    return float(off_diagonal.mean()), float(off_diagonal.var())
