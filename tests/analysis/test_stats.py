"""Check the fixed statistical procedures and planned comparison tables."""

import ast
import inspect

import numpy as np
import pytest

from medvqa.analysis import constants as K
from medvqa.analysis.bootstrap import (
    ClusterBootstrap,
    holm,
    state_label,
    summarize,
    tost,
)
from medvqa.analysis.contrasts import (
    confirmatory_specs,
    descriptive_specs,
    required_cells,
)
from medvqa.analysis.rq2 import cohen_kappa, gwet_ac1, icc21


def test_locked_constants():
    assert (
        K.SEED == 20261016
        and K.B == 10000
        and (K.ALPHA == 0.05)
        and (K.TOST_MARGIN_PP == 5.0)
    )
    assert K.PROBE_THRESHOLD == 0.7


def test_holm_known():
    adj, rej = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert list(rej) == [True, False, False, True]


def test_summarize_p_formula():
    reps = np.array([-0.01] * 30 + [0.02] * 970)
    s = summarize(0.015, reps)
    assert s["p"] == pytest.approx(2 * 30 / 1000)
    assert s["estimate"] == pytest.approx(1.5)
    s2 = summarize(0.0, np.zeros(100))
    assert s2["p"] == 1.0


def test_tost_levels():
    rng = np.random.default_rng(0)
    near0 = rng.normal(0, 0.01, 10000)
    t = tost(near0, m=10)
    assert t["tost_level"] == pytest.approx(0.99)
    assert t["equivalent"]
    assert not tost(rng.normal(0.045, 0.01, 10000), m=10)["equivalent"]
    assert tost(near0, m=3)["tost_level"] == pytest.approx(1 - 0.1 / 3)


def test_state_precedence():
    assert state_label(True, True) == "khac_biet"
    assert state_label(False, True) == "tuong_duong_pm5"
    assert state_label(False, False) == "khong_ket_luan"


def test_cluster_bootstrap_stratified_and_paired():
    qc = ["a", "a", "b", "c", "d", "e"]
    st = ["S1", "S1", "S1", "S1", "S2", "S2"]
    bt = ClusterBootstrap(qc, st, st, B=500, seed=K.SEED)
    s1 = [bt.clusters.index(c) for c in "abc"]
    s2 = [bt.clusters.index(c) for c in "de"]
    assert np.all(bt.W[:, s1].sum(1) == 3) and np.all(bt.W[:, s2].sum(1) == 2)
    bt2 = ClusterBootstrap(qc, st, st, B=500, seed=K.SEED)
    assert np.array_equal(bt.W, bt2.W)
    v = np.array([1, 0, 1, 1, 0, 1.0])
    p, r = bt.endpoint(v, np.ones(6, bool), "mean")
    assert p == pytest.approx(4 / 6)
    p2, _ = bt.endpoint(v, np.ones(6, bool), "macro")
    assert p2 == pytest.approx((3 / 4 + 1 / 2) / 2)


def test_agreement_known_values():
    cm = np.array([[20, 5], [10, 15]], float)
    assert cohen_kappa(cm) == pytest.approx(0.4)
    assert gwet_ac1(cm) == pytest.approx((0.7 - 0.495) / 0.505)
    assert cohen_kappa(np.diag([5.0, 3, 2]), "quadratic") == pytest.approx(1.0)


def _icc_reference(x):
    n, k = x.shape
    g = x.mean()
    msr = k * ((x.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - g) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + g) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def test_icc21_matches_anova():
    rng = np.random.default_rng(1)
    t = rng.normal(0, 1, 50)
    x = np.column_stack([t + rng.normal(0, 0.5, 50), t + 0.3 + rng.normal(0, 0.5, 50)])
    assert icc21(x) == pytest.approx(_icc_reference(x))


def test_full_design_matches_protocol():
    assert len(required_cells("FULL")) == 36
    specs = confirmatory_specs("FULL")
    fam = {}
    for s in specs:
        fam[s.family] = fam.get(s.family, 0) + 1
    assert fam == {
        "F1-RAD": 10,
        "F1-NAT": 10,
        "F3-RAD": 3,
        "F3-NAT": 3,
        "F-INT": 7,
        "F-INT2": 4,
    }
    tost_ids = sorted({s.id for s in specs if s.tost})
    assert tost_ids == sorted(
        [
            "C1-L",
            "C1-Q",
            "C1-M",
            "C2-L",
            "C2-Q",
            "C2-M",
            "C3-L",
            "C3-Q",
            "C3-M",
            "C4",
            "C7",
            "C8",
            "C9",
            "C16",
        ]
    )
    c4 = [s for s in specs if s.id == "C4"][0]
    assert dict(((c, w) for w, c in c4.terms)) == {
        "L-G-GRD": 0.5,
        "L-NONE": -0.5,
        "Q-G-GRD": 0.5,
        "Q-NONE": -0.5,
        "M-G-GRD": -1,
        "M-NONE": 1,
    }
    c7 = [s for s in specs if s.id == "C7"][0]
    assert len(c7.terms) == 12 and sum((w for w, _ in c7.terms)) == pytest.approx(0)
    ids = {s.id for s in descriptive_specs("FULL")}
    assert {"C5-L", "C5-M", "C6-L", "C6-M", "C16-RAD", "C16-NAT", "C22"} <= ids


def test_reduced_designs():
    fam = {}
    for s in confirmatory_specs("FB"):
        fam[s.family] = fam.get(s.family, 0) + 1
    assert fam["F1-RAD"] == 6 and "C4" not in {s.id for s in confirmatory_specs("FB")}
    late = confirmatory_specs("LATE")
    assert "F-INT2" not in {s.family for s in late}
    assert "C4'" in {s.id for s in late}


def test_run_all_has_only_path_options():
    from medvqa.analysis import run_all

    src = inspect.getsource(run_all.main)
    opts = [
        node.args[0].value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (node.func.attr == "add_argument")
    ]
    assert opts == ["--runs-dir", "--out-dir"]
