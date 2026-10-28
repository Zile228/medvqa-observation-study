"""Provenance-based routing of native medical image benchmarks."""

import json
import re
from pathlib import Path

RECORD_FIELDS_COMMON = (
    "neutral_id",
    "modality",
    "provenance_rule",
    "ck_chroma_frac",
    "route_reason",
)
PROVENANCE_RULES = ("dataset_table", "mura_xr", "radimagenet_prefix")
ROUTE_REASONS = (
    "ok",
    "unknown_dataset_level",
    "unknown_explicit_token",
    "other",
    "conflict",
    "ck_color",
    "ck_gray",
)
CLOSED_LABELS = {
    "modality": {"X-ray", "mammography", "MR", "skin", "unknown"},
    "X-ray": {
        "chest",
        "wrist",
        "elbow",
        "humerus",
        "shoulder",
        "finger",
        "hand",
        "other",
        "unknown",
    },
    "mammography": {"breast", "unknown"},
    "MR": {
        "brain",
        "abdomen",
        "knee",
        "shoulder",
        "hip",
        "ankle_hindfoot",
        "midfoot_forefoot",
        "toe_MTP_only",
        "foot_unresolved",
        "spine",
        "other",
        "unknown",
    },
    "skin": {"dermoscopy", "clinical_photo", "unresolved", "unknown"},
    "unknown": {"unknown"},
}
CK_CHROMA_MIN = 24
CK_GRAY_MODALITIES_MAX = 0.05
CK_SKIN_MIN = 0.02
MASS_FAIL_FRACTION = 0.2


class NativeBuildStop(RuntimeError):
    """Raised when image provenance does not satisfy the routing specification."""


class NativeAssigner:
    def __init__(self, spec_path):
        self.spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        if self.spec.get("id") != "NATIVE-PROVENANCE-v1":
            raise ValueError("unexpected native provenance spec id")
        ex = self.spec["extraction"]
        self.re_dataset = re.compile(ex["dataset"]["regex"])
        self.re_mura = re.compile(ex["mura_region"]["regex"])
        self.re_radimg = re.compile(ex["radimagenet_prefix"]["regex"])
        self.mura_tokens = set(ex["mura_region"]["closed_tokens"])
        self.radimg_tokens = set(ex["radimagenet_prefix"]["closed_tokens"])
        self.radimg_unknown = set(ex["radimagenet_prefix"]["explicit_unknown_tokens"])
        self.table = {row["dataset"]: row for row in self.spec["dataset_table"]}
        self.tok = self.spec["token_to_label"]

    @staticmethod
    def normalize(path):
        s = str(path).replace("\\", "/")
        if not s.startswith("Images/"):
            m = re.search("(?:^|/)Images/", s)
            if not m:
                raise NativeBuildStop(
                    "native path has no 'Images' component (T-NO-MATCH)"
                )
            s = s[m.start() :].lstrip("/")
        return s

    def label(self, path, source_dataset=None):
        """Returns (label, dataset).

        `label` holds only spec output fields (minus neutral_id/ck); `dataset` is
        returned separately and may only be used for aggregate counts (never written per
        image).
        """
        p = self.normalize(path)
        m = self.re_dataset.match(p)
        if not m:
            raise NativeBuildStop(
                "native path does not match the dataset regex (T-NO-MATCH)"
            )
        ds = m.group("dataset")
        if ds not in self.table:
            raise NativeBuildStop(
                (
                    "dataset outside dataset_table "
                    "(T-UNKNOWN-DATASET): add a row with a reason "
                    "before locking"
                )
            )
        if source_dataset is not None and str(source_dataset) != ds:
            return (
                {
                    "modality": "unknown",
                    "region": "unknown",
                    "provenance_rule": "dataset_table",
                    "route_reason": "conflict",
                },
                ds,
            )
        row = self.table[ds]
        if ds == "Mura":
            codes = [mm.group("code") for mm in self.re_mura.finditer(p)]
            if len(codes) != 1:
                raise NativeBuildStop(
                    f"MURA path must contain exactly one XR_* code (found {len(codes)})"
                )
            if codes[0] not in self.mura_tokens:
                raise NativeBuildStop("MURA XR_* code outside closed_tokens")
            region = self.tok["mura"][codes[0]]["region"]
            return (
                {
                    "modality": "X-ray",
                    "region": region,
                    "provenance_rule": "mura_xr",
                    "route_reason": "other" if region == "other" else "ok",
                },
                ds,
            )
        if ds == "RadImageNet":
            base = p.rsplit("/", 1)[-1]
            mm = self.re_radimg.match(base)
            if not mm:
                raise NativeBuildStop(
                    "RadImageNet file name does not match the prefix regex"
                )
            key = ("mri-" if mm.group("mri") else "") + mm.group("tok")
            if key in self.radimg_unknown:
                return (
                    {
                        "modality": "unknown",
                        "region": "unknown",
                        "provenance_rule": "radimagenet_prefix",
                        "route_reason": "unknown_explicit_token",
                    },
                    ds,
                )
            if key not in self.radimg_tokens:
                raise NativeBuildStop("RadImageNet prefix key outside closed_tokens")
            t = self.tok["radimagenet"][key]
            return (
                {
                    "modality": t["modality"],
                    "region": t["region"],
                    "provenance_rule": "radimagenet_prefix",
                    "route_reason": "ok",
                },
                ds,
            )
        if row["modality"] == "unknown":
            return (
                {
                    "modality": "unknown",
                    "region": "unknown",
                    "provenance_rule": "dataset_table",
                    "route_reason": row.get("route_reason", "unknown_dataset_level"),
                },
                ds,
            )
        lab = {
            "modality": row["modality"],
            "provenance_rule": "dataset_table",
            "route_reason": "ok",
        }
        if row["modality"] == "skin":
            lab["acquisition"] = row["acquisition"]
        else:
            lab["region"] = row["region"]
        return (lab, ds)

    @staticmethod
    def ck_chroma_frac(rgb_image):
        import numpy as np

        a = np.asarray(rgb_image.convert("RGB"), dtype=np.int16)
        chroma = a.max(axis=2) - a.min(axis=2)
        return float((chroma > CK_CHROMA_MIN).mean())

    @staticmethod
    def apply_ck(label, frac):
        """Apply the color gate to the rounded stored chroma measurement.

        R2-mini delta P2-a: round to 6 decimals FIRST and decide on the stored value, so
        anyone re-checking route_metadata.json (ck_chroma_frac vs thresholds) gets
        exactly the recorded route_reason.
        """
        stored = round(float(frac), 6)
        lab = dict(label, ck_chroma_frac=stored)
        if lab["route_reason"] == "ok":
            if (
                lab["modality"] in ("X-ray", "mammography", "MR")
                and stored > CK_GRAY_MODALITIES_MAX
            ):
                lab["route_reason"] = "ck_color"
            elif lab["modality"] == "skin" and stored < CK_SKIN_MIN:
                lab["route_reason"] = "ck_gray"
        return lab

    def record(self, neutral_id, path, source_dataset, rgb_image):
        lab, ds = self.label(path, source_dataset)
        rec = dict(
            self.apply_ck(lab, self.ck_chroma_frac(rgb_image)), neutral_id=neutral_id
        )
        validate_record(rec)
        return (rec, ds)


def validate_record(rec):
    """Check routing metadata against the permitted fields and labels.

    Accept only the declared fields and the labels in CLOSED_LABELS.
    """
    allowed = set(RECORD_FIELDS_COMMON) | {"region", "acquisition"}
    extra = set(rec) - allowed
    if (
        extra
        or not set(RECORD_FIELDS_COMMON) <= set(rec)
        or ("region" in rec) == ("acquisition" in rec)
    ):
        raise NativeBuildStop(f"native route record has wrong fields: {sorted(rec)}")
    if (
        rec["provenance_rule"] not in PROVENANCE_RULES
        or rec["route_reason"] not in ROUTE_REASONS
    ):
        raise NativeBuildStop(
            "native route record has an undeclared provenance_rule/route_reason"
        )
    m = rec["modality"]
    if m not in CLOSED_LABELS["modality"]:
        raise NativeBuildStop("modality outside closed list")
    sub = rec.get("region", rec.get("acquisition"))
    if sub not in CLOSED_LABELS[m]:
        raise NativeBuildStop("region/acquisition outside closed list")
    if not (
        isinstance(rec["ck_chroma_frac"], float) and 0.0 <= rec["ck_chroma_frac"] <= 1.0
    ):
        raise NativeBuildStop("ck_chroma_frac invalid")
    return True


def mass_fail_check(dev_rows):
    """dev_rows: [(dataset, route_reason)] for COH-NAT-DEV.

    Returns (per- dataset table, list of datasets with > 20% CK fails). Does NOT raise:
    prep_data.ipynb (A6b) writes the aggregate report and then raises NativeBuildStop if
    the list is non-empty.
    """
    from collections import defaultdict

    acc = defaultdict(lambda: {"n": 0, "ck_fail": 0})
    for ds, reason in dev_rows:
        acc[ds]["n"] += 1
        acc[ds]["ck_fail"] += reason in ("ck_color", "ck_gray")
    table = {
        ds: dict(v, frac=round(v["ck_fail"] / v["n"], 4))
        for ds, v in sorted(acc.items())
    }
    bad = [ds for ds, v in table.items() if v["frac"] > MASS_FAIL_FRACTION]
    return (table, bad)


def decoder_versions():
    """Record Pillow and image-codec versions used in preparation.

    Pillow + codec library versions (spec consistency_checks.decoder_pin), recorded in
    the prep run_config.
    """
    import PIL
    from PIL import features

    out = {"pillow": PIL.__version__}
    for f in ("jpg", "zlib", "libtiff", "webp"):
        try:
            out[f] = features.version(f)
        except Exception:
            out[f] = None
    try:
        out["libjpeg_turbo"] = features.check_feature("libjpeg_turbo")
    except Exception:
        out["libjpeg_turbo"] = None
    import numpy

    out["numpy"] = numpy.__version__
    return out
