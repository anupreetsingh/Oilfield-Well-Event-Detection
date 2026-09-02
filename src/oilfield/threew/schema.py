"""3W event types and the label space the classifiers predict.

Every observation in a 3W run carries a ``class`` label: 0 is normal operation, ``k`` is the
steady state of event ``k``, and ``100 + k`` is the transient leading into it. Windows labelled
with a transient are trained as the event itself, which is what lets the models alert before
the fault reaches steady state.
"""

from __future__ import annotations

import math

EVENT_TYPES = {
    0: "NORMAL",
    1: "ABRUPT_INCREASE_OF_BSW",
    2: "SPURIOUS_CLOSURE_OF_DHSV",
    3: "SEVERE_SLUGGING",
    4: "FLOW_INSTABILITY",
    5: "RAPID_PRODUCTIVITY_LOSS",
    6: "QUICK_RESTRICTION_IN_PCK",
    7: "SCALING_IN_PCK",
    8: "HYDRATE_IN_PRODUCTION_LINE",
    9: "HYDRATE_IN_SERVICE_LINE",
}
TRANSIENT_OFFSET = 100

# Both hydrate events restrict flow through the same lines, so they share one class: the
# classifiers predict normal operation plus 8 fault types.
LABEL_MERGE = {9: 8}
CLASS_NAMES = [
    "NORMAL",
    "ABRUPT_INCREASE_OF_BSW",
    "SPURIOUS_CLOSURE_OF_DHSV",
    "SEVERE_SLUGGING",
    "FLOW_INSTABILITY",
    "RAPID_PRODUCTIVITY_LOSS",
    "QUICK_RESTRICTION_IN_PCK",
    "SCALING_IN_PCK",
    "HYDRATE",
]
N_CLASSES = len(CLASS_NAMES)

# Event types whose runs include a labelled transient period (dataset.ini, TRANSIENT = True).
EVENTS_WITH_TRANSIENT = {1, 2, 5, 6, 7, 8, 9}

SOURCES = ("WELL", "SIMULATED", "DRAWN")


def model_label(raw_class: float | int | None) -> int | None:
    """Map a raw 3W observation label (0..9, 101..109) to the merged 0..8 class index."""
    if raw_class is None or (isinstance(raw_class, float) and math.isnan(raw_class)):
        return None
    event = int(raw_class) % TRANSIENT_OFFSET
    return LABEL_MERGE.get(event, event)


def is_transient(raw_class: float | int | None) -> bool:
    if raw_class is None or (isinstance(raw_class, float) and math.isnan(raw_class)):
        return False
    return int(raw_class) >= TRANSIENT_OFFSET
