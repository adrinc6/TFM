"""Portfolio optimizers: HRP, risk parity, and robust Markowitz."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover
    linkage = None
    squareform = None


def _to_cov(returns: pd.DataFrame) -> pd.DataFrame:
    R = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)
    if R.empty:
        return pd.DataFrame()
    return R.cov()


def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1.0 - corr).clip(lower=0.0) / 2.0) ** 0.5


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    n = link[-1, 3]
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values - n
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
    sub = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub.values)
    ivp /= ivp.sum()
    w = np.array(ivp).reshape(-1, 1)
    return float((w.T @ sub.values @ w).ravel()[0])


def hrp_weights(returns: pd.DataFrame) -> Dict[str, float]:
    """Compute Hierarchical Risk Parity weights."""
    cov = _to_cov(returns)
    if cov.empty:
        return {}

    cols = list(cov.columns)
    n = len(cols)
    if n == 1:
        return {cols[0]: 1.0}

    corr = returns[cols].corr().fillna(0.0)
    if linkage is None or squareform is None:
        w = np.ones(n) / n
        return {c: float(x) for c, x in zip(cols, w)}

    dist = _correl_dist(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, "single")
    sort_ix = _get_quasi_diag(link)
    ordered = [cols[i] for i in sort_ix]

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            alpha = 1.0 - v0 / max(v0 + v1, 1e-12)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha

    w = w / max(float(w.sum()), 1e-12)
    return {k: float(v) for k, v in w.to_dict().items()}


def risk_parity_weights(returns: pd.DataFrame) -> Dict[str, float]:
    cov = _to_cov(returns)
    if cov.empty:
        return {}
    inv_var = 1.0 / np.diag(cov.values)
    inv_var = np.where(np.isfinite(inv_var), inv_var, 0.0)
    if inv_var.sum() <= 0:
        inv_var = np.ones(len(cov.columns))
    w = inv_var / inv_var.sum()
    return {c: float(x) for c, x in zip(cov.columns, w)}


def robust_markowitz_weights(returns: pd.DataFrame, risk_aversion: float = 4.0) -> Dict[str, float]:
    cov = _to_cov(returns)
    if cov.empty:
        return {}
    mu = returns[cov.columns].mean().fillna(0.0).values
    cov_mat = cov.values

    # Ridge regularization improves stability under noisy estimates.
    lam = 1e-3
    cov_reg = cov_mat + lam * np.eye(cov_mat.shape[0])
    try:
        inv = np.linalg.pinv(cov_reg)
        raw = inv @ mu
    except Exception:
        raw = np.ones(cov_mat.shape[0])

    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        raw = np.ones_like(raw)
    w = raw / raw.sum()

    # Blend with risk parity for robustness.
    rp = np.array(list(risk_parity_weights(returns[cov.columns]).values()))
    blend = 1.0 / max(float(risk_aversion), 1.0)
    w = (1.0 - blend) * w + blend * rp
    w = w / max(float(w.sum()), 1e-12)
    return {c: float(x) for c, x in zip(cov.columns, w)}
