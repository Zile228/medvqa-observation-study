"""Planned contrasts, paired condition comparisons and sensitivity analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import parsers as P
from .bootstrap import ClusterBootstrap, holm, state_label, summarize, tost
from .constants import CORR_TYPES, REF_EXT, REF_MAIN, STRICT_EXCLUDED_ROUTES
from .scoring import pair_table


@dataclass
class Spec:
    id: str
    kind: str
    cohort: str
    terms: list
    family: Optional[str] = None
    rq: str = ""
    tost: bool = False
    label: str = ""


def design_cells(design: str) -> Dict[str, list]:
    """List the models and conditions required by a study design.

    Return the answerers, external observers and pooled intervention conditions.
    """
    if design == "FULL":
        return {
            "answerers": ["L", "Q", "M"],
            "ext": {"L": "G", "Q": "G", "M": "G"},
            "rq3": [(k, o) for k in ("L", "Q", "M") for o in ("SELF", "G")],
            "qext": ["L", "M"],
            "c4": ("L", "Q", "M"),
            "int2": True,
        }
    if design == "FB":
        return {
            "answerers": ["L", "Q"],
            "ext": {"L": "G", "Q": "G"},
            "rq3": [(k, o) for k in ("L", "Q") for o in ("SELF", "G")],
            "qext": [],
            "c4": None,
            "int2": True,
        }
    if design == "MAIN":
        return {
            "answerers": ["L", "Q"],
            "ext": {"L": "G", "Q": "G"},
            "rq3": [(k, o) for k in ("L", "Q") for o in ("SELF", "G")],
            "qext": [],
            "c4": None,
            "int2": False,
            "blank0": False,
        }
    if design == "LATE":
        return {
            "answerers": ["L", "Q", "M"],
            "ext": {"L": "Q", "M": "Q"},
            "rq3": [
                ("L", "SELF"),
                ("Q", "SELF"),
                ("M", "SELF"),
                ("L", "Q"),
                ("M", "Q"),
            ],
            "qext": [],
            "c4": ("L", None, "M"),
            "int2": False,
        }
    if design == "FB_LATE":
        return {
            "answerers": ["L", "Q"],
            "ext": {"L": "Q"},
            "rq3": [("L", "SELF"), ("Q", "SELF"), ("L", "Q")],
            "qext": [],
            "c4": None,
            "int2": False,
        }
    raise ValueError(design)


def required_cells(design: str) -> List[str]:
    d = design_cells(design)
    cells = []
    for k in d["answerers"]:
        cells.append(f"{k}-NONE")
        for arm in ("GEN", "ANA", "GRD"):
            cells.append(f"{k}-SELF-{arm}")
    for k, o in d["rq3"]:
        if o != "SELF":
            for arm in ("GEN", "ANA", "GRD"):
                cells.append(f"{k}-{o}-{arm}")
    for k in d["qext"]:
        for arm in ("GEN", "ANA", "GRD"):
            cells.append(f"{k}-Q-{arm}")
    ints = ("SHUF", "CORR", "EMPTY", "PARA", "BLANK") + (
        ("BLANK0",) if d.get("blank0", True) else ()
    )
    cells += [f"Q-INT-{x}" for x in ints]
    if d["int2"]:
        cells += [f"L-INT2-{x}" for x in ("SHUF", "EMPTY", "PARA")]
    return sorted(set(cells))


def _cell(k: str, o: str, arm: str) -> str:
    return f"{k}-{o}-{arm}"


def confirmatory_specs(design: str) -> List[Spec]:
    """Comparison families: F1-RAD, F1-NAT, F3-RAD, F3-NAT, F-INT, F-INT2."""
    d = design_cells(design)
    specs: List[Spec] = []
    for coh in ("RAD", "NAT"):
        f1 = f"F1-{coh}"
        for k in d["answerers"]:
            specs.append(
                Spec(
                    f"C1-{k}",
                    "acc",
                    coh,
                    [(1, f"{k}-SELF-GRD"), (-1, f"{k}-NONE")],
                    f1,
                    "RQ1",
                    True,
                )
            )
        for k, o in d["ext"].items():
            tag = "" if o == "G" else "(Q)"
            specs.append(
                Spec(
                    f"C2-{k}{tag}",
                    "acc",
                    coh,
                    [(1, _cell(k, o, "GRD")), (-1, f"{k}-NONE")],
                    f1,
                    "RQ1",
                    True,
                )
            )
        for k, o in d["ext"].items():
            tag = "" if o == "G" else "(Q)"
            specs.append(
                Spec(
                    f"C3-{k}{tag}",
                    "acc",
                    coh,
                    [(1, _cell(k, o, "GRD")), (-1, f"{k}-SELF-GRD")],
                    f1,
                    "RQ1",
                    True,
                )
            )
        if d["c4"] == ("L", "Q", "M"):
            t = [
                (0.5, "L-G-GRD"),
                (-0.5, "L-NONE"),
                (0.5, "Q-G-GRD"),
                (-0.5, "Q-NONE"),
                (-1, "M-G-GRD"),
                (1, "M-NONE"),
            ]
            specs.append(Spec("C4", "acc", coh, t, f1, "RQ1", True))
        elif d["c4"] == ("L", None, "M"):
            t = [(1, "L-Q-GRD"), (-1, "L-NONE"), (-1, "M-Q-GRD"), (1, "M-NONE")]
            specs.append(Spec("C4'", "acc", coh, t, f1, "RQ1", True))
        f3 = f"F3-{coh}"
        n = len(d["rq3"])
        for cid, (a, b) in {
            "C7": ("GRD", "ANA"),
            "C8": ("GRD", "GEN"),
            "C9": ("ANA", "GEN"),
        }.items():
            t = []
            for k, o in d["rq3"]:
                t += [(1.0 / n, _cell(k, o, a)), (-1.0 / n, _cell(k, o, b))]
            specs.append(Spec(cid, "acc", coh, t, f3, "RQ3", True))
    R = REF_MAIN
    for cid, x in (("C10", "SHUF"), ("C11", "CORR"), ("C12", "EMPTY")):
        specs.append(
            Spec(
                cid,
                "flip",
                "POOL",
                [(1, f"Q-INT-{x}", R), (-1, "Q-INT-PARA", R)],
                "F-INT",
                "CoT",
            )
        )
    for cid, x in (("C13", "SHUF"), ("C14", "CORR"), ("C15", "EMPTY")):
        specs.append(
            Spec(
                cid,
                "tv",
                "POOL",
                [(1, f"Q-INT-{x}", R), (-1, "Q-INT-PARA", R)],
                "F-INT",
                "CoT",
            )
        )
    specs.append(
        Spec("C16", "acc", "POOL", [(1, "Q-INT-BLANK"), (-1, R)], "F-INT", "CoT", True)
    )
    if d["int2"]:
        E = REF_EXT
        specs.append(
            Spec(
                "C18",
                "flip",
                "POOL",
                [(1, "L-INT2-SHUF", E), (-1, "L-INT2-PARA", E)],
                "F-INT2",
                "CoT",
            )
        )
        specs.append(
            Spec(
                "C19",
                "flip",
                "POOL",
                [(1, "L-INT2-EMPTY", E), (-1, "L-INT2-PARA", E)],
                "F-INT2",
                "CoT",
            )
        )
        specs.append(
            Spec(
                "C20",
                "tv",
                "POOL",
                [(1, "L-INT2-SHUF", E), (-1, "L-INT2-PARA", E)],
                "F-INT2",
                "CoT",
            )
        )
        specs.append(
            Spec(
                "C21",
                "tv",
                "POOL",
                [(1, "L-INT2-EMPTY", E), (-1, "L-INT2-PARA", E)],
                "F-INT2",
                "CoT",
            )
        )
    return specs


def descriptive_specs(design: str) -> List[Spec]:
    """Descriptive contrasts with 95% intervals and no Holm adjustment."""
    d = design_cells(design)
    s: List[Spec] = []
    for coh in ("RAD", "NAT"):
        for k in d["qext"]:
            s.append(
                Spec(
                    f"C5-{k}",
                    "acc",
                    coh,
                    [(1, f"{k}-Q-GRD"), (-1, f"{k}-NONE")],
                    rq="RQ1",
                )
            )
            s.append(
                Spec(
                    f"C6-{k}",
                    "acc",
                    coh,
                    [(1, f"{k}-Q-GRD"), (-1, f"{k}-SELF-GRD")],
                    rq="RQ1",
                )
            )
        per_cell = list(d["rq3"]) + [(k, "Q") for k in d["qext"]]
        for k, o in per_cell:
            for cid, (a, b) in {
                "C7": ("GRD", "ANA"),
                "C8": ("GRD", "GEN"),
                "C9": ("ANA", "GEN"),
            }.items():
                s.append(
                    Spec(
                        f"{cid}[{k}-{o}]",
                        "acc",
                        coh,
                        [(1, _cell(k, o, a)), (-1, _cell(k, o, b))],
                        rq="RQ3",
                    )
                )
        s.append(
            Spec(
                f"C16-{coh}", "acc", coh, [(1, "Q-INT-BLANK"), (-1, REF_MAIN)], rq="CoT"
            )
        )
    for coh in ("RAD", "NAT", "POOL"):
        for x in ("SHUF", "CORR", "EMPTY", "PARA", "BLANK"):
            c = f"Q-INT-{x}"
            s.append(
                Spec(
                    f"ACC[{x}-GRD]@{coh}",
                    "acc",
                    coh,
                    [(1, c), (-1, REF_MAIN)],
                    rq="CoT",
                )
            )
            s.append(
                Spec(f"E-S4[{x}]@{coh}", "flip", coh, [(1, c, REF_MAIN)], rq="CoT")
            )
            s.append(
                Spec(f"E-S5-TV[{x}]@{coh}", "tv", coh, [(1, c, REF_MAIN)], rq="CoT")
            )
            s.append(
                Spec(
                    f"E-S5-dpgold[{x}]@{coh}",
                    "dpgold",
                    coh,
                    [(1, c, REF_MAIN)],
                    rq="CoT",
                )
            )
        if d["int2"]:
            for x in ("SHUF", "EMPTY", "PARA"):
                c = f"L-INT2-{x}"
                s.append(
                    Spec(
                        f"ACC[INT2-{x}-GRD]@{coh}",
                        "acc",
                        coh,
                        [(1, c), (-1, REF_EXT)],
                        rq="CoT",
                    )
                )
                s.append(
                    Spec(
                        f"E-S4[INT2-{x}]@{coh}",
                        "flip",
                        coh,
                        [(1, c, REF_EXT)],
                        rq="CoT",
                    )
                )
                s.append(
                    Spec(
                        f"E-S5-TV[INT2-{x}]@{coh}",
                        "tv",
                        coh,
                        [(1, c, REF_EXT)],
                        rq="CoT",
                    )
                )
        if d.get("blank0", True):
            s.append(
                Spec(
                    f"C23@{coh}",
                    "acc",
                    coh,
                    [(1, "Q-INT-BLANK"), (-1, "Q-INT-BLANK0")],
                    rq="CoT",
                )
            )
            s.append(
                Spec(
                    f"C24@{coh}",
                    "acc",
                    coh,
                    [(1, "Q-INT-BLANK0"), (-1, "Q-NONE")],
                    rq="CoT",
                )
            )
    if d["int2"]:
        R, E = (REF_MAIN, REF_EXT)
        for cid, kind, x in (
            ("C22", "flip", "SHUF"),
            ("C22-EMPTY", "flip", "EMPTY"),
            ("C22-TV-SHUF", "tv", "SHUF"),
            ("C22-TV-EMPTY", "tv", "EMPTY"),
        ):
            t = [
                (1, f"Q-INT-{x}", R),
                (-1, "Q-INT-PARA", R),
                (-1, f"L-INT2-{x}", E),
                (1, "L-INT2-PARA", E),
            ]
            s.append(
                Spec(
                    cid,
                    kind,
                    "POOL",
                    t,
                    rq="CoT" if cid == "C22" else "CoT (mo ta bo sung, OQ-15)",
                )
            )
    return s


class Engine:
    """Evaluates specs with the paired cluster bootstrap."""

    def __init__(
        self,
        items: pd.DataFrame,
        rows: pd.DataFrame,
        qtab: Dict[str, pd.DataFrame],
        int_missing: Dict[str, set],
        boots: Dict[str, ClusterBootstrap],
        qids: Dict[str, list],
    ):
        self.items, self.rows, self.qtab = (items, rows, qtab)
        self.int_missing, self.boots, self.qids = (int_missing, boots, qids)
        self._pairs: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self.kind = {c: items.loc[self.qids[c], "kind"].values for c in self.qids}

    def base_mask(self, cohort: str, kind: str, metric: str = "") -> np.ndarray:
        k = self.kind[cohort]
        if kind in ("tv", "dpgold") or metric in ("argmax_correct", "p_gold", "brier"):
            return np.isin(k, ["yesno", "mcq"])
        if metric in ("em", "f1"):
            return k == "open"
        return np.isin(k, ["yesno", "closed_short", "mcq"])

    def qmask(self, cohort: str, fn) -> np.ndarray:
        return np.array([bool(fn(self.items.loc[q])) for q in self.qids[cohort]])

    def vec(self, metric: str, cell: str, cohort: str) -> np.ndarray:
        t = self.qtab[metric]
        if cell not in t.columns:
            raise KeyError(f"cell {cell} not available")
        return t[cell].reindex(self.qids[cohort]).to_numpy(dtype=float)

    def pair_vec(
        self, col: str, int_cell: str, ref_cell: str, cohort: str, parser: str = "lead"
    ) -> np.ndarray:
        key = (int_cell, ref_cell, parser)
        if key not in self._pairs:
            self._pairs[key] = pair_table(self.rows, int_cell, ref_cell, parser)
        return self._pairs[key][col].reindex(self.qids[cohort]).to_numpy(dtype=float)

    def missing_mask(self, cohort: str, cells: Sequence[str]) -> np.ndarray:
        drop = set()
        for c in cells:
            drop |= self.int_missing.get(c, set())
        return ~np.isin(np.array(self.qids[cohort], dtype=object), list(drop))

    def evaluate(
        self,
        spec: Spec,
        metric: str = "correct_lead",
        extra_mask: Optional[np.ndarray] = None,
        parser: str = "lead",
        flip_col: str = "flip",
    ) -> Tuple[float, np.ndarray, int]:
        coh = spec.cohort
        boot = self.boots[coh]
        cells = [t[1] for t in spec.terms] + [t[2] for t in spec.terms if len(t) > 2]
        mask = self.base_mask(coh, spec.kind, metric) & self.missing_mask(coh, cells)
        if extra_mask is not None:
            mask &= extra_mask
        if spec.kind == "acc":
            mode = "macro" if coh == "NAT" else "mean"
            vecs = [(w, self.vec(metric, c, coh)) for w, c in spec.terms]
        else:
            mode = "mean"
            col = {"flip": flip_col, "tv": "tv", "dpgold": "dpgold"}[spec.kind]
            vecs = [
                (w, self.pair_vec(col, ic, rc, coh, parser)) for w, ic, rc in spec.terms
            ]
        for _, v in vecs:
            mask &= np.isfinite(v)
        point, reps = (0.0, np.zeros(boot.B))
        for w, v in vecs:
            p, r = boot.endpoint(v, mask, mode)
            point += w * p
            reps = reps + w * r
        return (point, reps, int(mask.sum()))

    def run_confirmatory(
        self, specs: List[Spec]
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        recs, reps_store = ([], {})
        for sp in specs:
            point, reps, n = self.evaluate(sp)
            reps_store[f"{sp.id}|{sp.family}"] = reps
            recs.append(
                {
                    "contrast": sp.id,
                    "family": sp.family,
                    "rq": sp.rq,
                    "kind": sp.kind,
                    "cohort": sp.cohort,
                    "n_questions": n,
                    **summarize(point, reps),
                }
            )
        df = pd.DataFrame(recs)
        df["p_holm"] = np.nan
        df["holm_sig"] = False
        for fam, g in df.groupby("family"):
            adj, rej = holm(g["p"].values)
            df.loc[g.index, "p_holm"] = adj
            df.loc[g.index, "holm_sig"] = rej
        df["family_size"] = df.groupby("family")["contrast"].transform("count")
        tcols = {
            "tost_level": np.nan,
            "tost_ci_low": np.nan,
            "tost_ci_high": np.nan,
            "equivalent": None,
        }
        for k, v in tcols.items():
            df[k] = v
        for i, sp in enumerate(specs):
            if sp.tost:
                t = tost(
                    reps_store[f"{sp.id}|{sp.family}"], int(df.loc[i, "family_size"])
                )
                for k, v in t.items():
                    df.at[i, k] = v
        df["state"] = [
            state_label(h, e) if t else "khac_biet" if h else "khong_ket_luan"
            for h, e, t in zip(
                df["holm_sig"], df["equivalent"], [s.tost for s in specs]
            )
        ]
        return (df, reps_store)

    def run_descriptive(self, specs: List[Spec], available: set) -> pd.DataFrame:
        recs = []
        for sp in specs:
            cells = {t[1] for t in sp.terms} | {t[2] for t in sp.terms if len(t) > 2}
            if not cells <= available:
                continue
            point, reps, n = self.evaluate(sp)
            recs.append(
                {
                    "contrast": sp.id,
                    "rq": sp.rq,
                    "kind": sp.kind,
                    "cohort": sp.cohort,
                    "n_questions": n,
                    **summarize(point, reps),
                }
            )
        return pd.DataFrame(recs)


def routeq_report(eng: Engine, conf: pd.DataFrame, specs: List[Spec]) -> pd.DataFrame:
    recs = []
    for sp in specs:
        if not (
            sp.id.startswith(("C1-", "C2-", "C3-", "C4")) or sp.id in ("C7", "C8", "C9")
        ):
            continue
        if sp.cohort not in ("RAD", "NAT"):
            continue
        main = conf[(conf["contrast"] == sp.id) & (conf["family"] == sp.family)].iloc[0]
        row = {
            "contrast": sp.id,
            "cohort": sp.cohort,
            "main_estimate": main["estimate"],
            "main_ci_low": main["ci_low"],
            "main_ci_high": main["ci_high"],
            "main_holm_sig": bool(main["holm_sig"]),
        }
        for name, flag in (("routeq", True), ("nonroute", False)):
            m = eng.qmask(
                sp.cohort,
                lambda r, f=flag: r["routeq"] is not None and bool(r["routeq"]) == f,
            )
            point, reps, n = eng.evaluate(sp, extra_mask=m)
            s = summarize(point, reps)
            row.update(
                {
                    f"{name}_n": n,
                    f"{name}_estimate": s["estimate"],
                    f"{name}_ci_low": s["ci_low"],
                    f"{name}_ci_high": s["ci_high"],
                }
            )
        row["conclusion"] = routeq_conclusion(sp.id, row)
        recs.append(row)
    return pd.DataFrame(recs)


def routeq_conclusion(cid: str, r: dict) -> str:
    """Classify contrast results on routing and non-routing questions.

    Apply the interpretation rules to C1-k and C2-k, and check the remaining contrasts
    for consistency.
    """
    if not cid.startswith(("C1-", "C2-")):
        return "kiem_tra_nhat_quan" if cid in ("C7", "C8", "C9") else "mo_ta"
    if not r["main_holm_sig"]:
        return "hieu_chinh_khong_y_nghia: bao_ca_hai_tap_nhu_mo_ta"
    same_dir = np.sign(r["nonroute_estimate"]) == np.sign(r["main_estimate"])
    excl0 = r["nonroute_ci_low"] > 0 or r["nonroute_ci_high"] < 0
    if same_dir and excl0:
        return "observation giúp ích ngoài thông tin modality/vùng"
    return (
        "lợi ích có thể đến từ thông tin route "
        "(modality/vùng) chứ không chỉ từ nội dung quan "
        "sát"
    )


def sensitivity(
    eng: Engine, conf_specs: List[Spec], corr_plan: Optional[dict]
) -> pd.DataFrame:
    recs = []

    def add(sid, sp, point, reps, n, sub=""):
        recs.append(
            {
                "analysis": sid,
                "contrast": sp.id,
                "cohort": sp.cohort,
                "subset": sub,
                "n_questions": n,
                **summarize(point, reps),
            }
        )

    rq13 = [s for s in conf_specs if s.cohort in ("RAD", "NAT") and s.kind == "acc"]
    for sp in [s for s in rq13 if s.cohort == "RAD"]:
        m = eng.qmask("RAD", lambda r: not bool(r["weak_group"]))
        add("S-WEAK", sp, *eng.evaluate(sp, extra_mask=m))
    for sp in rq13 + [s for s in conf_specs if s.id == "C16"]:
        add("S-EXACT", sp, *eng.evaluate(sp, metric="correct_exact"))
    for sp in [s for s in rq13 if s.id == "C7"]:
        m = eng.qmask(
            sp.cohort,
            lambda r: (
                not set(r["routes_l"]) & STRICT_EXCLUDED_ROUTES
                and len(r["routes_l"]) > 0
            ),
        )
        add("S-STRICT", sp, *eng.evaluate(sp, extra_mask=m))
    for sp in rq13:
        col = "dataset" if sp.cohort == "RAD" else "group"
        for v in sorted(set(eng.items.loc[eng.qids[sp.cohort], col])):
            m = eng.qmask(sp.cohort, lambda r, v=v: r[col] == v)
            add("S-DS", sp, *eng.evaluate(sp, extra_mask=m), sub=str(v))
    for sp in rq13:
        add("S-ARGMAX", sp, *eng.evaluate(sp, metric="argmax_correct"))
    for sp in [s for s in conf_specs if s.id in ("C10", "C11", "C12", "C18", "C19")]:
        add("S-FLIPVALID", sp, *eng.evaluate(sp, flip_col="flip_valid"))
    if corr_plan:
        rel = corr_relevance(eng, corr_plan)
        for sp in [s for s in conf_specs if s.id in ("C11", "C14")]:
            for name, flag in (
                ("item_in_question", True),
                ("item_not_in_question", False),
            ):
                m = np.array([rel.get(q) is flag for q in eng.qids[sp.cohort]])
                add("S-CORR-REL", sp, *eng.evaluate(sp, extra_mask=m), sub=name)
    return pd.DataFrame(recs)


def corr_relevance(eng: Engine, plan: dict) -> Dict[str, Optional[bool]]:
    """Match edited item names and synonyms against question text.

    S-CORR-REL: NORM phrase match of edited item name / synonyms in the
    question.
    """
    out = {}
    for q, r in eng.items.iterrows():
        p = plan.get(r["image_id"])
        if not p:
            out[q] = None
            continue
        terms = [p.get("item_name", "")] + list(p.get("synonyms", []))
        nq = f" {P.norm(r['question'])} "
        out[q] = any((P.norm(t) and f" {P.norm(t)} " in nq for t in terms))
    return out


def corr_by_type(
    eng: Engine, corr_plan: Optional[dict], int_rows_type: Dict[str, str]
) -> pd.DataFrame:
    """Answer flips and probability shifts under CORR, grouped by edit type."""
    recs = []
    img_type = dict(int_rows_type)
    if corr_plan:
        img_type.update({k: v.get("edit_type") for k, v in corr_plan.items()})
    for coh in ("RAD", "NAT", "POOL"):
        for t in CORR_TYPES:
            m = eng.qmask(coh, lambda r, t=t: img_type.get(r["image_id"]) == t)
            for kind in ("flip", "tv"):
                sp = Spec(
                    f"{('E-S4' if kind == 'flip' else 'E-S5-TV')}[CORR-{t}]@{coh}",
                    kind,
                    coh,
                    [(1, "Q-INT-CORR", REF_MAIN)],
                )
                point, reps, n = eng.evaluate(sp, extra_mask=m)
                recs.append(
                    {
                        "contrast": sp.id,
                        "edit_type": t,
                        "cohort": coh,
                        "n_questions": n,
                        **summarize(point, reps),
                    }
                )
    return pd.DataFrame(recs)
