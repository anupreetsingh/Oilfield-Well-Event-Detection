"""Find 3W runs on disk and read one run into a tidy pandas frame.

The dataset is laid out as ``dataset/<event type>/<SOURCE>[_<start>].parquet``: one file per run
(an "instance"), with real runs named after their well (``WELL-00014_20170207190000``) and
simulated or hand-drawn runs numbered (``SIMULATED_00012``, ``DRAWN_00003``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PA_PER_BAR = 1e5


def parse_instance_name(path: str | Path) -> dict:
    path = Path(path)
    event_type = int(path.parent.name)
    stem = path.stem
    head = stem.split("_")[0]
    if head.startswith("WELL"):
        source, well_id = "WELL", head
    else:
        source, well_id = head, None
    return {
        "instance_id": f"{event_type}-{stem}",
        "path": str(path),
        "event_type": event_type,
        "source": source,
        "well_id": well_id,
    }


def discover_instances(raw_dir: str | Path, event_types: list[int] | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("[0-9]/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no 3W parquet files under {raw_dir}; run scripts/download_3w.sh")
    df = pd.DataFrame([parse_instance_name(f) for f in files])
    df["file_bytes"] = [f.stat().st_size for f in files]
    if event_types is not None:
        df = df[df["event_type"].isin(event_types)]
    return df.reset_index(drop=True)


def read_instance(path: str | Path, sensors: dict[str, dict]) -> pd.DataFrame:
    """Read one run with only the modelled sensors, renamed, in bar, with UTC timestamps.

    Missing sensor columns become all-NaN so every run has the same shape downstream.
    """
    available = set(pq.read_schema(path).names)
    wanted = [name for name in sensors if name in available] + ["class", "timestamp"]
    table = pq.read_table(path, columns=[c for c in wanted if c in available])
    df = table.to_pandas()
    if "timestamp" not in df.columns:
        df = df.reset_index()

    out = pd.DataFrame({"ts": pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")})
    for name, spec in sensors.items():
        values = df[name].to_numpy(dtype="float64") if name in df else np.full(len(df), np.nan)
        if spec["kind"] == "pressure":
            values = values / PA_PER_BAR
        out[spec["column"]] = values
    out["raw_class"] = df["class"].astype("float64").to_numpy() if "class" in df else np.nan
    return out.sort_values("ts", kind="stable").reset_index(drop=True)

