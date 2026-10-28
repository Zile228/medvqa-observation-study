"""Stratified image-cluster bootstrap, multiple testing and equivalence tests."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import ALPHA, SEED, TOST_MARGIN_PP
from .constants import B as B_LOCKED


class ClusterBootstrap:
    """Stratified cluster bootstrap over images grouped by RGB hash.

    One multiplicity matrix W (B x n_clusters) per cohort; the same W is used for every
    cell and contrast of that cohort (paired). Strata: dataset (RAD) or native group
    (NAT).
    """

    def __init__(
        self,
        q_cluster: Sequence[str],
        q_stratum: Sequence[str],
        q_group: Sequence[str],
        B: int = B_LOCKED,
        seed: int = SEED,
        _W: Optional[np.ndarray] = None,
        _clusters: Optional[List[str]] = None,
    ):
        q_cluster = np.asarray(q_cluster, dtype=object)
        q_stratum = np.asarray(q_stratum, dtype=object)
        self.q_group = np.asarray(q_group, dtype=object)
        self.B = B
        self.notes: List[str] = []
        if _clusters is None:
            order = np.argsort(q_cluster.astype(str), kind="stable")
            c_strat: Dict[str, str] = {}
            for i in order:
                c = q_cluster[i]
                if c in c_strat and c_strat[c] != q_stratum[i]:
                    self.notes.append(
                        f"cluster {c} spans strata {c_strat[c]}/{q_stratum[i]}"
                    )
                c_strat.setdefault(c, q_stratum[i])
            strata = sorted(set(c_strat.values()))
            clusters: List[str] = []
            for s in strata:
                clusters.extend(sorted((c for c, v in c_strat.items() if v == s)))
            self.clusters = clusters
            idx = {c: i for i, c in enumerate(clusters)}
            rng = np.random.default_rng(seed)
            W = np.zeros((B, len(clusters)), dtype=np.float64)
            for s in strata:
                cols = np.array([idx[c] for c in clusters if c_strat[c] == s])
                n = len(cols)
                draws = rng.integers(0, n, size=(B, n))
                flat = (draws + np.arange(B)[:, None] * n).ravel()
                W[:, cols] = np.bincount(flat, minlength=B * n).reshape(B, n)
            self.W = W
        else:
            self.clusters = _clusters
            self.W = _W
        cidx = {c: i for i, c in enumerate(self.clusters)}
        self.q_cidx = np.array([cidx[c] for c in q_cluster], dtype=np.int64)
        self.nC = len(self.clusters)

    @staticmethod
    def concat(
        a: "ClusterBootstrap", b: "ClusterBootstrap", qa, qb
    ) -> "ClusterBootstrap":
        """Combine paired radiology and native bootstrap replicates.

        Pooled set (C16, F-INT): replicate b uses RAD replicate b and NAT replicate b.
        """
        W = np.hstack([a.W, b.W])
        clusters = list(a.clusters) + list(b.clusters)
        qc = list(np.asarray(qa["cluster"])) + list(np.asarray(qb["cluster"]))
        qg = list(np.asarray(qa["group"])) + list(np.asarray(qb["group"]))
        return ClusterBootstrap(qc, qc, qg, B=a.B, _W=W, _clusters=clusters)

    def endpoint(
        self, v: np.ndarray, mask: np.ndarray, mode: str
    ) -> Tuple[float, np.ndarray]:
        """Calculate a question mean or a macro-average of group means.

        mode 'mean' = mean over questions (E-P1, pooled); 'macro' = mean of group means
        (E-P2).
        """
        v = np.asarray(v, dtype=float)
        m = np.asarray(mask, dtype=bool) & np.isfinite(v)
        if mode == "mean":
            return self._mean(v, m)
        if mode == "macro":
            pts, reps = ([], [])
            for g in sorted(set(self.q_group[m])):
                mg = m & (self.q_group == g)
                p, r = self._mean(v, mg)
                pts.append(p)
                reps.append(r)
            if not pts:
                return (np.nan, np.full(self.B, np.nan))
            return (float(np.mean(pts)), np.mean(np.vstack(reps), axis=0))
        raise ValueError(mode)

    def _mean(self, v, m):
        S = np.bincount(self.q_cidx[m], weights=v[m], minlength=self.nC)
        N = np.bincount(self.q_cidx[m], minlength=self.nC).astype(float)
        if N.sum() == 0:
            return (np.nan, np.full(self.B, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = self.W @ S / (self.W @ N)
        return (float(S.sum() / N.sum()), reps)


def summarize(point: float, reps: np.ndarray, scale: float = 100.0) -> dict:
    """Summarize paired bootstrap estimates with percentile intervals.

    Args:
        point: Observed estimate before scaling.
        reps: Bootstrap estimates; nonfinite replicates are counted and excluded.
        scale: Multiplier, normally 100 for percentage-point reporting.

    Returns:
        dict: Estimate, 95% interval, two-sided sign-tail p and replicate counts.
    """
    r = reps[np.isfinite(reps)] * scale
    Beff = len(r)
    out = {"estimate": point * scale, "B_eff": Beff, "B_nan": int(len(reps) - Beff)}
    if Beff == 0:
        out.update(ci_low=np.nan, ci_high=np.nan, p=np.nan)
        return out
    lo, hi = np.percentile(r, [2.5, 97.5], method="linear")
    p = min(1.0, 2.0 * min(np.sum(r <= 0), np.sum(r >= 0)) / Beff)
    out.update(ci_low=float(lo), ci_high=float(hi), p=float(p))
    return out


def tost(
    reps: np.ndarray,
    m: int,
    scale: float = 100.0,
    alpha: float = ALPHA,
    margin: float = TOST_MARGIN_PP,
) -> dict:
    """Check equivalence using the prespecified multiplicity-adjusted interval.

    Args:
        reps: Bootstrap estimates before scaling.
        m: Positive number of comparisons in the equivalence family.
        scale: Multiplier applied to estimates before interval calculation.
        alpha: One-sided family error level.
        margin: Symmetric equivalence margin after scaling.

    Returns:
        dict: Interval level, bounds and strict-inside-margin equivalence flag.
    """
    r = reps[np.isfinite(reps)] * scale
    level = 1 - 2 * alpha / m
    lo, hi = np.percentile(r, [100 * alpha / m, 100 * (1 - alpha / m)], method="linear")
    return {
        "tost_level": level,
        "tost_ci_low": float(lo),
        "tost_ci_high": float(hi),
        "equivalent": bool(lo > -margin and hi < margin),
    }


def holm(pvals: Sequence[float], alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Holm's step-down correction within one comparison family.

    Args:
        pvals: Sequence of finite p-values between zero and one.
        alpha: Family-wise rejection threshold.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Adjusted p-values and rejection flags.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return (adj, adj <= alpha)


def state_label(holm_sig: bool, equivalent: Optional[bool]) -> str:
    """Name the combined significance and equivalence outcome.

    Args:
        holm_sig: Whether the multiplicity-adjusted difference is significant.
        equivalent: Equivalence result, or None when no equivalence test applies.

    Returns:
        str: Prespecified interpretation label.
    """
    if holm_sig:
        return "khac_biet"
    if equivalent:
        return "tuong_duong_pm5"
    return "khong_ket_luan"
