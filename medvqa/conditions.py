"""Observation conditions and the answer-cell matrix used by the study."""

OBSERVERS = {"L": "OBS-L", "Q": "OBS-Q", "M": "OBS-M", "G": "OBS-G"}
ARMS3 = ("GRD", "ANA", "GEN")
ANSWER_CAP = {"yesno": 8, "mcq": 8, "closed_short": 24, "open": 24}
DESIGNS = ("FULL", "FB", "MAIN")
DEFAULT_DESIGN = "MAIN"
RQ1_OPEN_CELLS = [
    "CELL-L-NONE",
    "CELL-Q-NONE",
    "CELL-M-NONE",
    "CELL-L-SELF-GRD",
    "CELL-Q-SELF-GRD",
    "CELL-M-SELF-GRD",
    "CELL-L-G-GRD",
    "CELL-Q-G-GRD",
    "CELL-M-G-GRD",
    "CELL-L-Q-GRD",
    "CELL-M-Q-GRD",
]


def _cell(
    a,
    name,
    observer=None,
    arm=None,
    intervention=None,
    image="original",
    obs_kind="obs",
):
    return {
        "cell": f"CELL-{a}-{name}",
        "answerer": f"ANS-{a}",
        "obs_observer": observer,
        "obs_arm": f"ARM-{arm}" if arm else None,
        "intervention": intervention,
        "image_condition": image,
        "obs_kind": obs_kind,
    }


def _design(fb, design):
    """Validate an experiment design and resolve the reduced-design flag."""
    if design is None:
        design = "FB" if fb else "FULL"
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    return design


def answer_cells(answerer, fb=False, design=None):
    """Return the answer conditions for one model under the selected design."""
    d = _design(fb, design)
    a = answerer
    reduced = d in ("FB", "MAIN")
    if reduced and a == "M":
        return []
    cells = [_cell(a, "NONE", obs_kind="none")]
    for arm in ARMS3:
        cells.append(_cell(a, f"SELF-{arm}", OBSERVERS[a], arm))
    for arm in ARMS3:
        cells.append(_cell(a, f"G-{arm}", "OBS-G", arm))
    if a in ("L", "M") and (not reduced):
        for arm in ARMS3:
            cells.append(_cell(a, f"Q-{arm}", "OBS-Q", arm))
    if a == "Q":
        cells += [
            _cell(a, "INT-SHUF", "OBS-Q", "GRD", "INT-SHUF", obs_kind="shuf"),
            _cell(a, "INT-CORR", "OBS-Q", "GRD", "INT-CORR", obs_kind="edit"),
            _cell(a, "INT-EMPTY", None, None, "INT-EMPTY", obs_kind="empty"),
            _cell(a, "INT-PARA", "OBS-Q", "GRD", "INT-PARA", obs_kind="edit"),
            _cell(a, "INT-BLANK", "OBS-Q", "GRD", "INT-BLANK", image="gray128"),
        ]
        if d != "MAIN":
            cells.append(
                _cell(
                    a,
                    "INT-BLANK0",
                    None,
                    None,
                    "INT-BLANK0",
                    image="gray128",
                    obs_kind="none",
                )
            )
    if a == "L" and d != "MAIN":
        cells += [
            _cell(a, "INT2-SHUF", "OBS-G", "GRD", "INT2-SHUF", obs_kind="shuf"),
            _cell(a, "INT2-EMPTY", None, None, "INT2-EMPTY", obs_kind="empty"),
            _cell(a, "INT2-PARA", "OBS-G", "GRD", "INT2-PARA", obs_kind="edit"),
        ]
    return cells


def all_cells(fb=False, design=None):
    return [c for a in ("L", "Q", "M") for c in answer_cells(a, fb, design)]


def observers_for(design):
    """Observers whose observations are produced under a design."""
    return (
        ["OBS-L", "OBS-Q", "OBS-G"]
        if design in ("FB", "MAIN")
        else ["OBS-L", "OBS-Q", "OBS-M", "OBS-G"]
    )


def edit_types_for(design):
    """Return edited interventions enabled by the selected study design."""
    return (
        ["INT-PARA", "INT-CORR"]
        if design == "MAIN"
        else ["INT-PARA", "INT-CORR", "INT2-PARA"]
    )


MAIN_REQUIRED_CELLS = (
    [
        f"CELL-L-{n}"
        for n in ("NONE", "SELF-GEN", "SELF-ANA", "SELF-GRD", "G-GEN", "G-ANA", "G-GRD")
    ]
    + [
        f"CELL-Q-{n}"
        for n in ("NONE", "SELF-GEN", "SELF-ANA", "SELF-GRD", "G-GEN", "G-ANA", "G-GRD")
    ]
    + [
        f"CELL-Q-{n}"
        for n in ("INT-SHUF", "INT-CORR", "INT-EMPTY", "INT-PARA", "INT-BLANK")
    ]
)


def required_cells(design):
    """Return every condition required before outcome scoring can begin."""
    return [c["cell"] for c in all_cells(design=design)]


def design_state(design, rq2="AX1_SLAKE"):
    """Describe the main experiment matrix and reference-evaluation scope."""
    if design != "MAIN":
        raise ValueError("only the MAIN design_state format is defined here")
    if rq2 not in ("AX1_SLAKE", "NONE"):
        raise ValueError("rq2 must be AX1_SLAKE or NONE")
    return {
        "branch": "MAIN",
        "fb": True,
        "u3": "A",
        "u4": "A",
        "rq2": rq2,
        "configuration_ref": "MAIN",
    }


def observation_jobs(observer, neutral_ids):
    """Observation cache units: (observer, arm, neutral_id), arms GEN/ANA/GRD."""
    return [
        (OBSERVERS[observer], f"ARM-{arm}", n)
        for arm in ("GEN", "ANA", "GRD")
        for n in neutral_ids
    ]
