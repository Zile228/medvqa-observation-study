"""Mixed-effects and generalized estimating equation sensitivity models."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contrasts import Spec

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    HAVE_SM = True
except Exception:
    HAVE_SM = False


def model_frame(
    rows: pd.DataFrame,
    items: pd.DataFrame,
    spec: Spec,
    metric: str = "correct_lead",
    extra_q: Optional[set] = None,
) -> Tuple[pd.DataFrame, List[Tuple[float, str]]]:
    """Prepare per-question data for an accuracy contrast.

    Long data: one row per (question[, order], cell) for the cells of an accuracy
    contrast.

    Returns the frame and the linear combination over per-cell 'arm' effects (pair index
    -> weight). Pairs are (cell_after, cell_before) with weights from the spec.
    """
    t = spec.terms
    if len(t) % 2 or any(
        (abs(t[i][0] + t[i + 1][0]) > 1e-12 for i in range(0, len(t), 2))
    ):
        raise ValueError(f"{spec.id}: terms are not (after, before) pairs")
    pairs = [(t[i][1], t[i + 1][1], t[i][0]) for i in range(0, len(t), 2)]
    it = (
        items.set_index("qid")
        if "qid" in items.columns and items.index.name != "qid"
        else items
    )
    coh_q = set(
        it.index[
            (it["cohort"] == spec.cohort)
            & it["kind"].isin(["yesno", "closed_short", "mcq"])
        ]
    )
    if extra_q is not None:
        coh_q &= extra_q
    frames = []
    for i, (after, before, w) in enumerate(pairs):
        for cell, arm in ((after, 1), (before, 0)):
            r = rows.loc[
                (rows["cell"] == cell) & rows["qid"].isin(coh_q),
                ["qid", "order", metric, "image_id"],
            ]
            frames.append(r.assign(arm=arm, pair=i, cell=cell))
    df = pd.concat(frames, ignore_index=True).rename(columns={metric: "y"})
    df = df.dropna(subset=["y"])
    cl = it.drop_duplicates("image_id").set_index("image_id")["cluster_id"]
    df["cluster"] = df["image_id"].map(cl)
    return (df, [(w, i) for i, (_, _, w) in enumerate(pairs)])


def fit_gee(
    df: pd.DataFrame,
    comb: List[Tuple[float, int]],
    nat: bool,
    covariate: Optional[str] = None,
    family: str = "binomial",
) -> dict:
    """Fit an exchangeable image-cluster GEE for one contrast.

    GEE, exchangeable within image cluster; arm effect per pair, linear combination
    reported.
    """
    if not HAVE_SM:
        return {"method": "unavailable (statsmodels missing)"}
    d = df.copy()
    npair = int(d["pair"].max()) + 1
    for i in range(npair):
        d[f"arm_p{i}"] = ((d["pair"] == i) & (d["arm"] == 1)).astype(float)
    rhs = [f"arm_p{i}" for i in range(npair)]
    if npair > 1:
        rhs.append("C(pair)")
    if nat:
        rhs.append("C(order)")
    if covariate:
        rhs.append(covariate)
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    formula = "y ~ " + " + ".join(rhs)
    d = d.sort_values("cluster")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.gee(
                formula,
                groups="cluster",
                data=d,
                family=fam,
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        except Exception as e:
            return {"method": f"GEE failed: {e}"}
    L = np.zeros(len(res.params))
    names = list(res.params.index)
    for w, i in comb:
        L[names.index(f"arm_p{i}")] = w
    est = float(L @ res.params.values)
    se = float(np.sqrt(L @ res.cov_params().values @ L))
    lo, hi = (est - 1.959964 * se, est + 1.959964 * se)
    out = {
        "method": f"GEE-{family}-exchangeable (R unavailable)",
        "n_rows": len(d),
        "estimate": est,
        "se": se,
    }
    if family == "binomial":
        out.update(OR=np.exp(est), OR_ci_low=np.exp(lo), OR_ci_high=np.exp(hi))
    else:
        out.update(pp=est * 100, pp_ci_low=lo * 100, pp_ci_high=hi * 100)
    if covariate:
        out["covariate_coef"] = float(res.params[covariate])
    return out


def run_models(rows, items, specs: Sequence[Spec]) -> pd.DataFrame:
    recs = []
    for sp in specs:
        if sp.kind != "acc" or sp.cohort not in ("RAD", "NAT") or (not sp.family):
            continue
        df, comb = model_frame(rows, items, sp)
        recs.append(
            {
                "contrast": sp.id,
                "cohort": sp.cohort,
                **fit_gee(df, comb, sp.cohort == "NAT"),
            }
        )
    return pd.DataFrame(recs)


def s_len(rows, items, specs: Sequence[Spec], obs: pd.DataFrame) -> pd.DataFrame:
    """Estimate observation-length sensitivity for the planned contrasts.

    S-LEN: C7-C9 with z-scored observation output tokens as covariate (logit OR
    and linear pp).
    """
    if obs is None or obs.empty:
        return pd.DataFrame()
    olen = obs.set_index(["observer", "arm", "image_id"])["n_output_tokens"].astype(
        float
    )
    mu, sd = (olen.mean(), olen.std() or 1.0)
    recs = []
    for sp in specs:
        if sp.id not in ("C7", "C8", "C9"):
            continue
        df, comb = model_frame(rows, items, sp)
        keys = []
        for cell, img in zip(df["cell"], df["image_id"]):
            k, o, arm = cell.split("-")
            observer = k if o == "SELF" else o
            keys.append((observer, arm, img))
        df["len_z"] = (np.array([olen.get(x, np.nan) for x in keys]) - mu) / sd
        df = df.dropna(subset=["len_z"])
        for fam in ("binomial", "gaussian"):
            recs.append(
                {
                    "contrast": sp.id,
                    "cohort": sp.cohort,
                    "analysis": "S-LEN",
                    **fit_gee(
                        df, comb, sp.cohort == "NAT", covariate="len_z", family=fam
                    ),
                }
            )
    return pd.DataFrame(recs)


R_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "resources/statistics/mixed_models.R"
).read_text(encoding="utf-8")


def write_r_script(rows, items, specs: Sequence[Spec], out_dir: str) -> str:
    """Export contrast data and an R script for mixed-model analysis.

    Exports per-contrast data + the exact R script, for the user to run where R/lme4
    exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    for sp in specs:
        if sp.kind != "acc" or sp.cohort not in ("RAD", "NAT") or (not sp.family):
            continue
        df, _ = model_frame(rows, items, sp)
        safe = sp.id.replace("'", "p")
        df[["y", "arm", "pair", "cell", "qid", "order", "image_id"]].to_csv(
            os.path.join(out_dir, f"mm_{safe}_{sp.cohort}_.csv"), index=False
        )
    p = os.path.join(out_dir, "mixed_models_lme4.R")
    with open(p, "w", encoding="utf-8") as f:
        f.write(R_TEMPLATE)
    return p
