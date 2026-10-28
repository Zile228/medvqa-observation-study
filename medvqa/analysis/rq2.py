"""Observation support, judge agreement and reference-panel coverage analyses."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from medvqa.judge_scope import validate_scope_metadata

from .bootstrap import ClusterBootstrap, summarize
from .constants import NOT_RUN_MAIN, PROBE_THRESHOLD, SEED
from .constants import B as B_LOCKED
from .metrics import parse_judge_json

AX1_LABELS = ("supported", "contradicted", "not_verifiable")


def split_obs_key(k: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Split an observation key into model, arm and image identifiers.

    Keys have the form `OBS|<observer>|<arm>|<neutral_id>`.
    """
    p = str(k).split("|")
    if len(p) != 4:
        return (None, None, None)
    return (p[1].replace("OBS-", ""), p[2].replace("ARM-", ""), p[3])


SKIP_STATUSES = ("obs_error_skipped", "no_statements", "empty_prediction")


def _status(r) -> str:
    st = getattr(r, "status", "ok")
    return "ok" if st is None or (isinstance(st, float) and np.isnan(st)) else str(st)


def ax1_units(r) -> list:
    """Normalise one J-AX1 record to [(statement_id, sid, obs_key, is_probe, probe_id)].

    Accepts per-statement fields
    (`statements[].obs_key/is_probe/probe_id`) and record-level fields
    (`refs.obs_key`, `refs.is_probe_block`, `refs.probe_ids` aligned with `statements`).
    """
    refs = r.refs if isinstance(r.refs, dict) else {}
    st = refs.get("statements", []) or []
    rec_obs = refs.get("obs_key")
    block = bool(refs.get("is_probe_block"))
    pids = refs.get("probe_ids") or []
    out = []
    for n, s_ in enumerate(st):
        is_probe = bool(s_.get("is_probe", block))
        pid = s_.get("probe_id") or (pids[n] if is_probe and n < len(pids) else None)
        if is_probe and pid is None:
            pid = s_.get("sid")
        out.append(
            (
                int(s_["id"]),
                s_.get("sid"),
                None if is_probe else s_.get("obs_key") or rec_obs,
                is_probe,
                pid,
            )
        )
    return out


def ax1_statements(judge: pd.DataFrame) -> pd.DataFrame:
    """Normalize statement-level support judgments.

    One row per (judge, variant, statement). judge_invalid -> label None (no
    repair). Records with a skip status (no judge call) produce no statements.
    """
    j = judge[judge["task"] == "J-AX1"]
    recs = []
    for r in j.itertuples(index=False):
        if _status(r) in SKIP_STATUSES:
            continue
        units = ax1_units(r)
        ids = {u[0] for u in units}
        obj = parse_judge_json("J-AX1", r.text, ids)
        labels = (
            None if obj is None else {int(e["id"]): e["label"] for e in obj["labels"]}
        )
        for sid_n, sid, ok, is_probe, pid in units:
            obs, arm, img = split_obs_key(ok) if ok else (None, None, None)
            recs.append(
                {
                    "judge": r.judge,
                    "variant": getattr(r, "variant", "main") or "main",
                    "sid": sid,
                    "is_probe": is_probe,
                    "probe_id": pid,
                    "observer": obs,
                    "arm": arm,
                    "image_id": img or _probe_img(pid),
                    "obs_key": ok,
                    "label": None if labels is None else labels.get(sid_n),
                }
            )
    cols = [
        "judge",
        "variant",
        "sid",
        "is_probe",
        "probe_id",
        "observer",
        "arm",
        "image_id",
        "obs_key",
        "label",
    ]
    return pd.DataFrame(recs, columns=cols)


def _probe_img(pid) -> Optional[str]:
    if not pid:
        return None
    p = str(pid).split("-")
    return "-".join(p[1:3]) if len(p) >= 3 else None


def ax2_table(judge: pd.DataFrame, n_items_for_image) -> pd.DataFrame:
    j = judge[judge["task"] == "J-AX2"]
    recs = []
    for r in j.itertuples(index=False):
        if _status(r) in SKIP_STATUSES:
            continue
        refs = r.refs if isinstance(r.refs, dict) else {}
        obs, arm, img = split_obs_key(refs.get("obs_key", ""))
        obj = parse_judge_json("J-AX2", r.text)
        n_items = n_items_for_image(img)
        recs.append(
            {
                "judge": r.judge,
                "observer": obs,
                "arm": arm,
                "image_id": img,
                "valid": obj is not None,
                "items_addressed": obj["items_addressed"] if obj else np.nan,
                "e_r2c": obj["items_addressed"] / n_items
                if obj and n_items
                else np.nan,
                "diagnosis": float(obj["diagnosis"] == "yes") if obj else np.nan,
                "unsupported": float(obj["unsupported_claim"] == "yes")
                if obj
                else np.nan,
            }
        )
    return pd.DataFrame(recs)


def ax3_table(judge: pd.DataFrame) -> pd.DataFrame:
    j = judge[judge["task"] == "J-AX3"]
    recs = []
    for r in j.itertuples(index=False):
        if _status(r) in SKIP_STATUSES:
            continue
        refs = r.refs if isinstance(r.refs, dict) else {}
        obs, arm, img = split_obs_key(refs.get("obs_key", ""))
        obj = parse_judge_json("J-AX3", r.text)
        recs.append(
            {
                "judge": r.judge,
                "observer": obs,
                "arm": arm,
                "image_id": img,
                "qid": refs.get("qid"),
                "relevance": float(obj["relevance"]) if obj else np.nan,
            }
        )
    return pd.DataFrame(recs)


def gwet_ac1(cm: np.ndarray) -> float:
    """Gwet AC1 for two raters from a q x q confusion matrix."""
    n = cm.sum()
    q = cm.shape[0]
    pa = np.trace(cm) / n
    pi = (cm.sum(0) + cm.sum(1)) / (2 * n)
    pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe)


def cohen_kappa(cm: np.ndarray, weights: str = "none") -> float:
    n = cm.sum()
    q = cm.shape[0]
    p = cm / n
    r, c = (p.sum(1), p.sum(0))
    if weights == "quadratic":
        i, j = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
        w = 1 - (i - j) ** 2 / (q - 1) ** 2
    else:
        w = np.eye(q)
    po = (w * p).sum()
    pe = (w * np.outer(r, c)).sum()
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def icc21(x: np.ndarray) -> float:
    """Compute two-way random, absolute-agreement ICC for two raters.

    ICC(2,1) two-way random, absolute agreement, single rater (Shrout & Fleiss), x: n x
    2.
    """
    return _icc_from_stats(*_icc_stats(x, np.ones(len(x))))


def _icc_stats(x, w):
    s = x.sum(1)
    return (
        w.sum(),
        (w * s).sum(),
        (w * s**2).sum(),
        (w * x[:, 0]).sum(),
        (w * x[:, 1]).sum(),
        (w * (x**2).sum(1)).sum(),
    )


def _icc_from_stats(n, S, Srow2, c1, c2, SSq, k=2):
    if n < 2:
        return np.nan
    cf = S**2 / (k * n)
    sst = SSq - cf
    ssr = Srow2 / k - cf
    ssc = (c1**2 + c2**2) / n - cf
    sse = sst - ssr - ssc
    msr, msc, mse = (ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1)))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / den if den else np.nan


def _image_boot(images: List[str], strata: Dict[str, str]) -> ClusterBootstrap:
    imgs = sorted(set(images))
    return ClusterBootstrap(
        imgs,
        [strata.get(i, "ALL") for i in imgs],
        ["ALL"] * len(imgs),
        B=B_LOCKED,
        seed=SEED,
    )


def agreement(
    pairs: pd.DataFrame, cats: List, strata: Dict[str, str], kind: str
) -> dict:
    """pairs: columns image_id, a (J1), b (J2). CI 95% bootstrap by image."""
    pairs = pairs.dropna(subset=["a", "b"])
    if pairs.empty:
        return {"n": 0}
    boot = _image_boot(list(pairs["image_id"]), strata)
    cidx = {c: i for i, c in enumerate(boot.clusters)}
    ci = np.array([cidx[i] for i in pairs["image_id"]])
    q = len(cats)
    pos = {c: i for i, c in enumerate(cats)}
    cell = np.array([pos[a] * q + pos[b] for a, b in zip(pairs["a"], pairs["b"])])
    per = np.zeros((boot.nC, q * q))
    np.add.at(per, (ci, cell), 1)
    rep_cm = (boot.W @ per).reshape(-1, q, q)
    cm = per.sum(0).reshape(q, q)
    out = {"n": int(cm.sum())}
    fns = {"pct_agree": lambda m: np.trace(m) / m.sum()}
    if kind == "weighted":
        fns["kappa_quadratic"] = lambda m: cohen_kappa(m, "quadratic")
    else:
        fns["gwet_ac1"] = gwet_ac1
        fns["cohen_kappa"] = cohen_kappa
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, f in fns.items():
            reps = np.array([f(m) for m in rep_cm])
            s = summarize(f(cm), reps, scale=1.0)
            out[name] = s["estimate"]
            out[f"{name}_ci"] = [s["ci_low"], s["ci_high"]]
    return out


def icc_boot(x: pd.DataFrame, strata: Dict[str, str]) -> dict:
    x = x.dropna(subset=["a", "b"])
    if len(x) < 3:
        return {"n": len(x)}
    boot = _image_boot(list(x["image_id"]), strata)
    cidx = {c: i for i, c in enumerate(boot.clusters)}
    ci = np.array([cidx[i] for i in x["image_id"]])
    v = x[["a", "b"]].to_numpy(float)
    comps = np.column_stack(
        [np.ones(len(v)), v.sum(1), v.sum(1) ** 2, v[:, 0], v[:, 1], (v**2).sum(1)]
    )
    per = np.zeros((boot.nC, 6))
    np.add.at(per, ci, comps)
    rep = boot.W @ per
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.array([_icc_from_stats(*r) for r in rep])
    s = summarize(_icc_from_stats(*per.sum(0)), reps, scale=1.0)
    return {
        "n": len(x),
        "icc21": s["estimate"],
        "icc21_ci": [s["ci_low"], s["ci_high"]],
    }


def _ratio_ci(df: pd.DataFrame, value: str, strata, by: List[str]) -> pd.DataFrame:
    """Bootstrap grouped shares or means by image.

    Share/mean of `value` per group, 95% CI bootstrap by image; adds the two-judge
    average.
    """
    d = df.dropna(subset=[value])
    if d.empty:
        return pd.DataFrame()
    boot = ClusterBootstrap(
        list(d["image_id"]),
        [strata.get(i, "ALL") for i in d["image_id"]],
        ["ALL"] * len(d),
        B=B_LOCKED,
        seed=SEED,
    )
    v = d[value].to_numpy(float)
    recs, reps_by = ([], {})
    for key, g in d.groupby(by + ["judge"]):
        m = d.index.isin(g.index)
        p, r = boot.endpoint(v, m, "mean")
        reps_by[key] = (p, r)
        recs.append(
            {
                **dict(zip(by + ["judge"], key)),
                "n_units": int(m.sum()),
                **summarize(p, r, 1.0),
            }
        )
    for key in sorted({k[:-1] for k in reps_by}):
        js = [reps_by[k] for k in reps_by if k[:-1] == key and k[-1] in ("J1", "J2")]
        if len(js) == 2:
            p = (js[0][0] + js[1][0]) / 2
            r = (js[0][1] + js[1][1]) / 2
            recs.append(
                {**dict(zip(by, key)), "judge": "mean_J1_J2", **summarize(p, r, 1.0)}
            )
    return pd.DataFrame(recs).drop(columns=["p"], errors="ignore")


RQ2_TABLES = (
    "E-R2a",
    "E-R2b",
    "J-PROBE",
    "E-R2c",
    "E-R2d-diagnosis",
    "E-R2d-unsupported",
    "E-R2e",
    "E-R2e-dist",
    "E-R2-evasion",
)
AX1_TABLES = ("E-R2a", "E-R2b", "J-PROBE", "E-R2-evasion")


def not_run_table() -> pd.DataFrame:
    return pd.DataFrame([{"status": NOT_RUN_MAIN}])


def run_rq2(
    judge: pd.DataFrame,
    obs: pd.DataFrame,
    probe_set,
    strata: Dict[str, str],
    n_items_for_image,
    level: str = "FULL",
    ax1_scope=None,
) -> dict:
    """Calculate the support-analysis tables for the selected scope.

    level: "FULL" or the MAIN levels "AX1_SLAKE" (J-AX1 on SLAKE +
    J-PROBE + AC1/kappa) and "NONE" (nothing run; every RQ2 table says 'không chạy
    (MAIN)').
    """
    res: dict = {"tables": {}, "level": level}
    if ax1_scope is not None:
        validate_scope_metadata(ax1_scope)
        res["ax1_scope"] = ax1_scope
        res["tables"]["AX1-coverage"] = ax1_coverage(judge, ax1_scope)
        res["tables"]["AX1-exclusions"] = pd.DataFrame(
            [{**x, "status": "out_of_AX1_coverage"} for x in ax1_scope["excluded"]]
        )
    if level == "NONE":
        res["tables"] = {t: not_run_table() for t in RQ2_TABLES}
        res["note"] = NOT_RUN_MAIN
        res["probe_rule_pass"] = None
        return res
    if level == "AX1_SLAKE":
        res["tables"].update(
            {t: not_run_table() for t in RQ2_TABLES if t not in AX1_TABLES}
        )
        if judge is not None and (not judge.empty):
            dropped = int((~judge["task"].isin(["J-AX1", "J-OPEN"])).sum())
            if dropped:
                res["note_main"] = (
                    f"{dropped} J-AX2/J-AX3 records ignored under MAIN (rq2=AX1_SLAKE)"
                )
            judge = judge[judge["task"] == "J-AX1"]
    if judge is None or judge.empty:
        res["note"] = "no JUDGE-v1 records"
        return res
    s1 = ax1_statements(judge)
    main = s1[s1["variant"] == "main"]
    real = main[~main["is_probe"]].copy()
    tables = res["tables"]
    inv = (
        s1.assign(invalid=s1["label"].isna())
        .groupby(["judge", "variant", "is_probe"])["invalid"]
        .mean()
    )
    res["ax1_judge_invalid_rate"] = {
        f"{k[0]}|{k[1]}|probe={k[2]}": float(v) for k, v in inv.items()
    }
    real["supported"] = np.where(
        real["label"].isna(), np.nan, (real["label"] == "supported").astype(float)
    )
    real["contradicted"] = np.where(
        real["label"].isna(), np.nan, (real["label"] == "contradicted").astype(float)
    )
    tables["E-R2a"] = _ratio_ci(real, "supported", strata, ["observer", "arm"])
    tables["E-R2b"] = _ratio_ci(real, "contradicted", strata, ["observer", "arm"])
    piv = real.pivot_table(
        index=["sid", "image_id"], columns="judge", values="label", aggfunc="first"
    ).reset_index()
    agr = {}
    if {"J1", "J2"} <= set(piv.columns):
        agr["J-AX1"] = agreement(
            piv.rename(columns={"J1": "a", "J2": "b"}),
            list(AX1_LABELS),
            strata,
            "nominal",
        )
    res["probe"] = probe_accuracy(main[main["is_probe"]], probe_set, strata)
    tables["J-PROBE"] = res["probe"]["table"]
    res["probe_rule_pass"] = res["probe"]["pass"]
    a2 = ax2_table(judge, n_items_for_image) if level == "FULL" else pd.DataFrame()
    if not a2.empty:
        res["ax2_judge_invalid_rate"] = (
            a2.groupby("judge")["valid"].apply(lambda v: 1 - v.mean()).to_dict()
        )
        tables["E-R2c"] = _ratio_ci(a2, "e_r2c", strata, ["observer", "arm"])
        tables["E-R2d-diagnosis"] = _ratio_ci(
            a2, "diagnosis", strata, ["observer", "arm"]
        )
        tables["E-R2d-unsupported"] = _ratio_ci(
            a2, "unsupported", strata, ["observer", "arm"]
        )
        p2 = a2.pivot_table(
            index=["observer", "arm", "image_id"],
            columns="judge",
            values=["diagnosis", "unsupported", "e_r2c"],
        ).reset_index()
        if ("diagnosis", "J1") in p2.columns and ("diagnosis", "J2") in p2.columns:
            for q in ("diagnosis", "unsupported"):
                pr = pd.DataFrame(
                    {"image_id": p2["image_id"], "a": p2[q, "J1"], "b": p2[q, "J2"]}
                )
                agr[f"J-AX2-{q}"] = agreement(pr, [0.0, 1.0], strata, "nominal")
            pr = pd.DataFrame(
                {
                    "image_id": p2["image_id"],
                    "a": p2["e_r2c", "J1"],
                    "b": p2["e_r2c", "J2"],
                }
            )
            agr["E-R2c-ICC"] = icc_boot(pr, strata)
    a3 = ax3_table(judge) if level == "FULL" else pd.DataFrame()
    if not a3.empty:
        tables["E-R2e"] = _ratio_ci(a3, "relevance", strata, ["observer", "arm"])
        dist = (
            a3.dropna(subset=["relevance"])
            .groupby(["observer", "judge"])["relevance"]
            .value_counts(normalize=True)
        )
        tables["E-R2e-dist"] = dist.rename("share").reset_index()
        p3 = a3.pivot_table(
            index=["observer", "image_id", "qid"], columns="judge", values="relevance"
        ).reset_index()
        if {"J1", "J2"} <= set(p3.columns):
            agr["J-AX3"] = agreement(
                p3.rename(columns={"J1": "a", "J2": "b"}),
                [0.0, 1.0, 2.0],
                strata,
                "weighted",
            )
    res["agreement"] = agr
    tables["E-R2-evasion"] = evasion_rate(judge)
    res["bias"] = judge_bias(s1, real, a2, obs)
    res["support_by_obs"] = support_per_observation(real)
    return res


def evasion_rate(judge: pd.DataFrame) -> pd.DataFrame:
    """Count visibility statements without descriptive content.

    Lines containing only 'not visible' or 'not assessable' are excluded from judging
    and counted as evasion. Report skipped / (skipped + judged statements) for each
    observer and observation arm.
    """
    j = judge[(judge["task"] == "J-AX1") & (judge.get("variant", "main") == "main")]
    seen, recs = (set(), [])
    for r in j.itertuples(index=False):
        refs = r.refs if isinstance(r.refs, dict) else {}
        units = [u for u in ax1_units(r) if not u[3]]
        ok = units[0][2] if units else refs.get("obs_key")
        if not ok or refs.get("is_probe_block") or ok in seen:
            continue
        seen.add(ok)
        obs, arm, img = split_obs_key(ok)
        ev = refs.get("skipped_evasion", refs.get("evasion", 0))
        recs.append(
            {
                "observer": obs,
                "arm": arm,
                "n_judged": len(units),
                "n_skipped": len(ev) if isinstance(ev, list) else int(ev or 0),
            }
        )
    if not recs:
        return pd.DataFrame()
    d = (
        pd.DataFrame(recs)
        .groupby(["observer", "arm"])[["n_judged", "n_skipped"]]
        .sum()
        .reset_index()
    )
    d["evasion_rate"] = d["n_skipped"] / (d["n_skipped"] + d["n_judged"])
    return d


def probe_accuracy(pr: pd.DataFrame, probe_set, strata) -> dict:
    """Summarize known-answer probe accuracy by judge, class and template.

    Classes are supported and contradicted; templates are A and B.
    """
    exp = _probe_expectations(probe_set)
    if pr.empty or not exp:
        return {"table": pd.DataFrame(), "pass": None, "note": "no probe data"}
    d = pr.copy()
    d["expected"] = d["probe_id"].map(lambda p: exp.get(p, {}).get("expected"))
    d["template"] = d["probe_id"].map(lambda p: exp.get(p, {}).get("template"))
    d["image_id"] = d.apply(
        lambda r: exp.get(r["probe_id"], {}).get("image_id") or r["image_id"], axis=1
    )
    d = d.dropna(subset=["expected"])
    d["correct"] = (d["label"] == d["expected"]).astype(float)
    t1 = _ratio_ci(d.rename(columns={"expected": "cls"}), "correct", strata, ["cls"])
    t2 = _ratio_ci(
        d.rename(columns={"expected": "cls"}), "correct", strata, ["cls", "template"]
    )
    tab = pd.concat([t1.assign(template="ALL"), t2], ignore_index=True)
    tab = tab[tab["judge"] != "mean_J1_J2"]
    acc = d.groupby(["judge", "expected"])["correct"].mean()
    ok = all(
        (
            acc.get((j, c), 0.0) >= PROBE_THRESHOLD
            for j in ("J1", "J2")
            for c in ("supported", "contradicted")
        )
    )
    return {
        "table": tab,
        "pass": bool(ok),
        "point": {f"{k[0]}|{k[1]}": float(v) for k, v in acc.items()},
    }


def _probe_expectations(probe_set) -> Dict[str, dict]:
    if not probe_set:
        return {}
    items = (
        probe_set.get("probes", probe_set) if isinstance(probe_set, dict) else probe_set
    )
    out = {}
    for p in items:
        pid = p.get("probe_id") or p.get("id")
        out[pid] = {
            "expected": p.get("expected_label") or p.get("expected") or p.get("label"),
            "template": p.get("template") or p.get("pattern"),
            "image_id": p.get("neutral_id"),
        }
    return out


def support_per_observation(real: pd.DataFrame) -> pd.DataFrame:
    """Calculate supported-statement shares for each observation.

    E-R2a per observation (share supported), per judge and two-judge mean; used by C17
    and bias (ii).
    """
    d = real.dropna(subset=["supported"])
    g = (
        d.groupby(["observer", "arm", "image_id", "judge"])["supported"]
        .mean()
        .unstack("judge")
    )
    js = [c for c in ("J1", "J2") if c in g.columns]
    g["support_mean"] = g[js].mean(axis=1) if js else np.nan
    return g.reset_index()


def judge_bias(s1, real, a2, obs) -> dict:
    """Measure statement-order agreement and observation-length correlations."""
    out = {}
    rev = s1[s1["variant"] == "reversed"]
    if not rev.empty:
        m = rev.merge(
            s1[s1["variant"] == "main"], on=["judge", "sid"], suffixes=("_r", "_m")
        )
        m = m.dropna(subset=["label_r", "label_m"])
        out["position_keep_rate"] = (
            m.groupby("judge")
            .apply(lambda g: float((g["label_r"] == g["label_m"]).mean()))
            .to_dict()
        )
        kap = {}
        for jn, g in m.groupby("judge"):
            pos = {c: i for i, c in enumerate(AX1_LABELS)}
            cm = np.zeros((3, 3))
            for a, b in zip(g["label_m"], g["label_r"]):
                cm[pos[a], pos[b]] += 1
            kap[jn] = float(cohen_kappa(cm)) if cm.sum() else np.nan
        out["position_kappa"] = kap
    if obs is not None and (not obs.empty):
        olen = obs.set_index(["observer", "arm", "image_id"])["n_output_tokens"]
        sp = {}
        sup = (
            real.dropna(subset=["supported"])
            .groupby(["judge", "observer", "arm", "image_id"])["supported"]
            .mean()
        )
        for jn, g in sup.groupby(level=0):
            x = [olen.get(k[1:], np.nan) for k in g.index]
            ok = np.isfinite(np.array(x, float))
            if ok.sum() > 2:
                sp[f"{jn}|E-R2a"] = float(
                    spearmanr(np.array(x, float)[ok], g.values[ok]).correlation
                )
        if a2 is not None and (not a2.empty):
            for jn, g in a2.dropna(subset=["items_addressed"]).groupby("judge"):
                x = np.array(
                    [
                        olen.get((o, a, i), np.nan)
                        for o, a, i in zip(g["observer"], g["arm"], g["image_id"])
                    ],
                    float,
                )
                ok = np.isfinite(x)
                if ok.sum() > 2:
                    sp[f"{jn}|items_addressed"] = float(
                        spearmanr(x[ok], g["items_addressed"].values[ok]).correlation
                    )
        out["length_spearman"] = sp
    return out


def c17_sample(rows, items, support, answerer, external, eligible_images=None):
    """Restrict every condition to common reference-panel coverage.

    Apply common image coverage before selecting ANY arm, including NONE
    (REFERENCE_PANEL).
    """
    sup = (
        support[support["arm"] == "GRD"].set_index(["observer", "image_id"])[
            "support_mean"
        ]
        if support is not None and (not support.empty)
        else {}
    )
    sl = set(
        items.index[
            (items["dataset"] == "SLAKE")
            & items["kind"].isin(["yesno", "closed_short"])
        ]
    )
    cells = {f"{answerer}-NONE": None, f"{answerer}-SELF-GRD": answerer}
    if external:
        cells[f"{answerer}-{external}-GRD"] = external
    frames, counts = ([], [])
    for c, observer in cells.items():
        all_rows = rows.loc[
            (rows["cell"] == c) & rows["qid"].isin(sl),
            ["qid", "image_id", "correct_lead"],
        ].copy()
        mask = (
            all_rows["image_id"].isin(eligible_images)
            if eligible_images is not None
            else pd.Series(True, index=all_rows.index)
        )
        excluded, frame = (all_rows[~mask], all_rows[mask].copy())
        frame["is_self"] = float(c.endswith("SELF-GRD"))
        frame["is_ext"] = float(observer is not None and (not c.endswith("SELF-GRD")))
        frame["support"] = [
            0.0 if observer is None else sup.get((observer, i), np.nan)
            for i in frame["image_id"]
        ]
        missing = frame[frame["support"].isna()]
        kept = frame.dropna(subset=["support"])
        counts.append(
            {
                "answerer": answerer,
                "cell": c,
                "n_scope_eligible_images": len(eligible_images)
                if eligible_images is not None
                else None,
                "n_images_excluded_scope": excluded["image_id"].nunique(),
                "n_questions_excluded_scope": excluded["qid"].nunique(),
                "n_rows_excluded_scope": len(excluded),
                "n_rows_missing_support": len(missing),
                "n_images_missing_support": missing["image_id"].nunique(),
                "n_questions_missing_support": missing["qid"].nunique(),
                "n_images": kept["image_id"].nunique(),
                "n_questions": kept["qid"].nunique(),
                "n_rows": len(kept),
            }
        )
        frames.append(kept)
    return (pd.concat(frames), pd.DataFrame(counts))


def ax1_coverage(judge, scope):
    """Scope/attrition counts, separate from statement-valid denominators of E-R2a/b."""
    eligible = validate_scope_metadata(scope)
    if judge is None:
        judge = pd.DataFrame()
    records = judge.to_dict("records")
    buckets = {}
    for r in records:
        if r.get("task") != "J-AX1" or (r.get("variant") or "main") != "main":
            continue
        refs = r.get("refs") or {}
        obs_key = refs.get("obs_key")
        if not obs_key:
            continue
        observer, arm, image = split_obs_key(obs_key)
        if image not in eligible:
            raise ValueError(
                "J-AX1 result outside REFERENCE_PANEL common image coverage"
            )
        buckets.setdefault((r["judge"], observer, arm), []).append((image, r))
    recs = []
    for j in ("J1", "J2"):
        for observer in ("L", "Q", "G"):
            for arm in ("GEN", "ANA", "GRD"):
                group = buckets.get((j, observer, arm), [])
                images, with_statements, with_valid = (set(), set(), set())
                counts = {
                    "valid_statements": 0,
                    "invalid_statements": 0,
                    "judge_invalid": 0,
                    "no_statements": 0,
                    "obs_error_skipped": 0,
                }
                for image, r in group:
                    images.add(image)
                    status = r.get("status") or "ok"
                    if status in SKIP_STATUSES:
                        counts[status] = counts.get(status, 0) + 1
                        continue
                    statements = (r.get("refs") or {}).get("statements") or []
                    if statements:
                        with_statements.add(image)
                    valid = (
                        parse_judge_json(
                            "J-AX1", r.get("text"), {int(s["id"]) for s in statements}
                        )
                        is not None
                    )
                    counts["valid_statements" if valid else "invalid_statements"] += (
                        len(statements)
                    )
                    counts["judge_invalid"] += int(not valid)
                    if valid and statements:
                        with_valid.add(image)
                recs.append(
                    {
                        "judge": j,
                        "observer": observer,
                        "arm": arm,
                        "expected_images": 96,
                        "eligible_images": 87,
                        "missing_reference_images": 9,
                        "coverage": 87 / 96,
                        "images_with_records": len(images),
                        "images_missing_records": len(eligible - images),
                        "images_with_statements": len(with_statements),
                        "images_with_valid_statements": len(with_valid),
                        **counts,
                    }
                )
    return pd.DataFrame(recs)


def c17(
    rows: pd.DataFrame,
    items: pd.DataFrame,
    support: pd.DataFrame,
    answerers: List[str],
    ext: Dict[str, str],
    probe_pass: Optional[bool],
    eligible_images=None,
) -> pd.DataFrame:
    """Estimate the planned answer-accuracy and support association.

    C17-k: correct ~ arm x support on SLAKE test closed questions, arm in {NONE,
    SELF-GRD, EXT-GRD}; support = E-R2a (two-judge mean) of the observation used; NONE
    coded by its own indicator. Fitted with GEE (logit, exchangeable, cluster image) —
    final fallback in the model-fitting sequence.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        have_sm = True
    except Exception:
        have_sm = False
    if support is None or support.empty:
        return pd.DataFrame()
    it = items
    cl = it.drop_duplicates("image_id").set_index("image_id")["cluster_id"]
    recs = []
    for k in answerers:
        o = ext.get(k)
        d, _ = c17_sample(rows, items, support, k, o, eligible_images)
        sample_counts = {
            "n_images": d["image_id"].nunique(),
            "n_questions": d["qid"].nunique(),
        }
        obs_sup = d.loc[(d["is_self"] == 1) | (d["is_ext"] == 1), "support"]
        if d.empty or obs_sup.empty or obs_sup.std() == 0:
            recs.append(
                {
                    "contrast": f"C17-{k}",
                    "term": None,
                    "n_rows": len(d),
                    **sample_counts,
                    ("note"): (
                        "not estimable: no SLAKE rows with support, or "
                        "support constant (e.g. every statement judged "
                        "the same label)"
                    ),
                }
            )
            continue
        if not have_sm:
            recs.append(
                {
                    "contrast": f"C17-{k}",
                    "term": None,
                    "n_rows": len(d),
                    **sample_counts,
                    "note": "not fitted: statsmodels unavailable in this Python",
                }
            )
            continue
        d["cluster"] = d["image_id"].map(cl)
        d = d.sort_values("cluster")
        f = "correct_lead ~ is_self + is_ext + is_self:support" + (
            " + is_ext:support" if o else ""
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.gee(
                f,
                groups="cluster",
                data=d,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        ci = res.conf_int()
        for term in [t for t in res.params.index if ":support" in t]:
            recs.append(
                {
                    "contrast": f"C17-{k}",
                    "term": term,
                    "OR_per_unit_support": float(np.exp(res.params[term])),
                    "OR_ci_low": float(np.exp(ci.loc[term, 0])),
                    "OR_ci_high": float(np.exp(ci.loc[term, 1])),
                    "OR_per_0.1": float(np.exp(0.1 * res.params[term])),
                    "p": float(res.pvalues[term]),
                    "n_rows": len(d),
                    **sample_counts,
                    "method": "GEE-binomial-exchangeable",
                    "interpretable_probe_rule": probe_pass,
                }
            )
    return pd.DataFrame(recs)


def rq2_integrity(
    judge: pd.DataFrame,
    res: dict,
    c17_table: Optional[pd.DataFrame],
    has_probe_set: bool,
    level: str = "FULL",
) -> dict:
    """RQ2 outputs that must exist given the judge records present.

    Empty- but-expected -> FAIL (never silent).

    MAIN: level "NONE" -> NOT_RUN_MAIN (never FAIL); "AX1_SLAKE" -> only
    J-AX1/J-PROBE/AC1-kappa are required, C17 is reported if estimable but not required.
    """
    if level == "NONE":
        return {"status": "NOT_RUN_MAIN", "problems": []}
    if judge is None or judge.empty:
        return {
            "status": "NOT_RUN",
            "problems": ["no JUDGE-v1 records: E-R2a..e, J-PROBE, C17 not computed"],
        }
    t = res.get("tables", {})
    ok = judge[
        judge.get("status", pd.Series("ok", index=judge.index)).fillna("ok").eq("ok")
    ]
    tasks = set(ok["task"])
    problems = []

    def need(name, cond_present):
        tab = t.get(name)
        if cond_present and (tab is None or len(tab) == 0):
            problems.append(f"{name} empty although judge records exist")

    s1 = (
        ax1_statements(judge)
        if "J-AX1" in tasks
        else pd.DataFrame(columns=["is_probe", "judge"])
    )
    real_n = int((~s1["is_probe"].astype(bool)).sum()) if len(s1) else 0
    probe_n = int(s1["is_probe"].astype(bool).sum()) if len(s1) else 0
    if "J-AX1" in tasks and real_n == 0:
        problems.append(
            (
                "J-AX1 records present but no non-probe statement"
                " could be read (layout mismatch?)"
            )
        )
    need("E-R2a", "J-AX1" in tasks)
    need("E-R2b", "J-AX1" in tasks)
    need("E-R2-evasion", "J-AX1" in tasks)
    if "J-AX1" in tasks:
        if probe_n == 0:
            problems.append(
                (
                    "no J-PROBE statements found in J-AX1 records "
                    "(judge calibration rule cannot be applied)"
                )
            )
        elif not has_probe_set:
            problems.append("jprobe_set.json missing: probe expected labels unknown")
        need("J-PROBE", probe_n > 0 and has_probe_set)
        if res.get("probe_rule_pass") is None:
            problems.append(
                "probe_rule_pass is None (judge calibration rule not evaluated)"
            )
        judges = set(s1["judge"]) if len(s1) else set()
        if not {"J1", "J2"} <= judges:
            problems.append(
                (f"J-AX1 judges present {sorted(judges)}; both roles J1, J2 required")
            )
        if level == "FULL":
            if (
                c17_table is None
                or len(c17_table) == 0
                or "contrast" not in getattr(c17_table, "columns", [])
            ):
                problems.append(
                    (
                        "C17 empty although J-AX1 exists (statsmodels "
                        "missing or no SLAKE data)"
                    )
                )
            elif "note" in c17_table.columns:
                for r in c17_table[c17_table["note"].notna()].itertuples():
                    problems.append(f"{r.contrast} {r.note}")
        if (
            level == "AX1_SLAKE"
            and "J-AX1" not in res.get("agreement", {})
            and ({"J1", "J2"} <= judges)
        ):
            problems.append("J-AX1 agreement (AC1/kappa) not computed")
    if level == "FULL":
        need("E-R2c", "J-AX2" in tasks)
        need("E-R2e", "J-AX3" in tasks)
    return {"status": "FAIL" if problems else "PASS", "problems": problems}
