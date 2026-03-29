"""Shared Ledoit-Wolf shrinkage utilities used by MIQP, Layered, and LASSO."""

import numpy as np


def ledoit_wolf(R_np):
    """Return OAS covariance estimate Sigma_LW (N x N)."""
    try:
        from sklearn.covariance import OAS
        return OAS().fit(R_np).covariance_
    except ImportError:
        pass

    T, N  = R_np.shape
    S     = np.cov(R_np, rowvar=False, bias=False)
    mu    = np.trace(S) / N
    s2    = np.sum(S ** 2)
    b2    = 0.0
    m     = R_np.mean(0)
    for t in range(T):
        x  = R_np[t] - m
        b2 += np.sum((np.outer(x, x) - S) ** 2)
    b2 /= T ** 2
    b2    = min(b2, s2)
    delta = b2 / s2 if s2 > 0 else 0.0
    return (1 - delta) * S + delta * mu * np.eye(N)


def shrunk_quadratic_form(R_np, idx_np, use_lw=True):
    """
    Return (Q, c) for the MIP objective  min w'Qw + c'w.

        Q = T * Sigma_LW   if use_lw=True  (full-rank, fixes overfitting)
        Q = R'R            if use_lw=False  (raw, rank-deficient when T<N)
        c = -2 * R' @ I   always from raw R_np — never shrunk

    Usage in every model:
        Q, c = shrunk_quadratic_form(R_np, idx_np, self.use_lw_shrinkage)
    """
    c = -2.0 * (R_np.T @ idx_np)   # always raw

    if use_lw:
        try:
            Sigma = ledoit_wolf(R_np)
            return R_np.shape[0] * Sigma, c
        except Exception:
            pass

    return R_np.T @ R_np, c


def whitened_return_matrix(R_np):
    """
    Return R_white = R @ Sigma_LW^{-1/2} for LASSO OMP scoring only.
    Falls back to raw R on failure.
    """
    try:
        Sigma = ledoit_wolf(R_np)
        eigv, eigvec = np.linalg.eigh(Sigma)
        eigv = np.maximum(eigv, 1e-10)
        W    = eigvec @ np.diag(1.0 / np.sqrt(eigv)) @ eigvec.T
        return R_np @ W
    except Exception:
        return R_np