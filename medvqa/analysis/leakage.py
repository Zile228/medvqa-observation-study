"""Check question leakage against observation item names."""

from __future__ import annotations

import pandas as pd

from . import parsers as P


def _in_header(phrase: str, header: str) -> bool:
    n = P.norm(phrase)
    return bool(n) and f" {n} " in f" {P.norm(header)} "


def check13(items: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for coh, g in items.groupby("cohort"):
        cs = g[g["kind"] == "closed_short"]
        if len(cs):
            hit = [_in_header(gd, h) for gd, h in zip(cs["gold"], cs["route_header"])]
            recs.append(
                {
                    "cohort": coh,
                    "kind": "closed_short",
                    "measure": "gold_in_header",
                    "n": len(cs),
                    "rate": sum(hit) / len(cs),
                }
            )
        mc = g[g["kind"] == "mcq"]
        if len(mc):
            gold_hit, one_hit = ([], [])
            for opts, gi, h in zip(mc["options"], mc["gold_index"], mc["route_header"]):
                gold_hit.append(_in_header(opts[gi], h))
                one_hit.append(sum((_in_header(o, h) for o in opts)) == 1)
            recs.append(
                {
                    "cohort": coh,
                    "kind": "mcq",
                    "measure": "gold_option_text_in_header",
                    "n": len(mc),
                    "rate": sum(gold_hit) / len(mc),
                }
            )
            recs.append(
                {
                    "cohort": coh,
                    "kind": "mcq",
                    "measure": "exactly_one_option_in_header",
                    "n": len(mc),
                    "rate": sum(one_hit) / len(mc),
                }
            )
        yn = g[g["kind"] == "yesno"]
        if len(yn):
            known = yn["routeq"].dropna()
            recs.append(
                {
                    "cohort": coh,
                    "kind": "yesno",
                    "measure": "proxy_share_in_routeq_set",
                    "n": len(known),
                    "rate": float(known.astype(bool).mean())
                    if len(known)
                    else float("nan"),
                }
            )
    return pd.DataFrame(recs)
