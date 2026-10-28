"""Predefined endpoint, parser, resampling and multiplicity constants."""

from __future__ import annotations

SEED = 20261016
B = 10000
CI_LEVEL = 0.95
ALPHA = 0.05
TOST_MARGIN_PP = 5.0
PROBE_THRESHOLD = 0.7
ANSWERERS = ("L", "Q", "M")
OBSERVERS_ALL = ("L", "Q", "M", "G")
OBS_ARMS = ("GEN", "ANA", "GRD")
NAT_GROUPS = ("MR", "X-Ray", "Dermoscopy")
RAD_DATASETS = ("SLAKE", "VQA-RAD")
STRICT_EXCLUDED_ROUTES = frozenset(
    {"RT-MR-ABD", "RT-MR-MIDFOOT", "RT-CT-FACE", "RT-SKIN-CLIN", "RT-FALLBACK"}
)
ROUTEQ_SLAKE_CONTENT = frozenset({"modality", "organ", "plane"})
ROUTEQ_VQARAD_TOKENS = ("MODALITY", "ORGAN", "PLANE")
ROUTEQ_OMNI_TYPES = frozenset({"modality recognition", "anatomy identification"})
YESNO_LEAD = (
    "^(is|are|does|do|did|was|were|can|could|has|have"
    "|had|will|would|should|may|might)\\b"
)
INT_MAIN = ("SHUF", "CORR", "EMPTY", "PARA", "BLANK", "BLANK0")
INT_EXT = ("SHUF", "EMPTY", "PARA")
CORR_TYPES = ("PRES", "LAT", "SEV")
REF_MAIN = "Q-SELF-GRD"
REF_EXT = "L-G-GRD"
DESIGNS = ("FULL", "FB", "LATE", "FB_LATE", "MAIN")
MAIN_RQ2_LEVELS = ("AX1_SLAKE", "NONE")
NOT_RUN_MAIN = "không chạy (MAIN)"
JUDGE_STRIP_ONE_JSON_FENCE = True
JUDGE_JSON_FENCE_REGEX = "^```(?:json)?[ \\t]*\\r?\\n(.*?)\\r?\\n```$"
