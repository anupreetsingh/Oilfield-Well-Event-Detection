"""Per-run cleaning of 3W sensor readings.

A run flows through three masks, in order, and every rejected value becomes NaN:

1. out-of-range: values outside each sensor's physical limits (historian sentinels such as
   -9 999 000 Pa, overflowed floats, dead gauges reporting 0 bar);
2. frozen: a pressure or temperature that repeats the exact same value for
   ``frozen_min_seconds`` or longer, which a live gauge sampled at 1 Hz never does (only for
   real wells: simulated runs have no gauges, and hand-drawn runs are flat by construction);
3. low coverage: a sensor left with less than ``min_sensor_coverage`` valid readings is dropped
   for the whole run, so models never see a sensor that is mostly holes.

The surviving 1 Hz readings are then averaged into fixed buckets (10 s by default), which is
the resolution the window features and the CNN work at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    frozen_min_seconds: float = 600
    frozen_kinds: tuple[str, ...] = ("pressure", "temperature")
    frozen_sources: tuple[str, ...] = ("WELL",)
    min_sensor_coverage: float = 0.5
    min_valid_sensors: int = 3
    resample_seconds: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> CleaningConfig:
        d = dict(d)
        for key in ("frozen_kinds", "frozen_sources"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def out_of_range_mask(values: pd.Series, lo: float, hi: float) -> pd.Series:
    return values.notna() & ((values < lo) | (values > hi))


def frozen_mask(values: pd.Series, ts: pd.Series, min_seconds: float) -> pd.Series:
    """True for readings inside a run of identical consecutive values lasting >= min_seconds."""
    v = values.to_numpy()
    if len(v) == 0:
        return pd.Series(False, index=values.index)
    # NaN != NaN, so gaps always start a new run and never count as frozen.
    starts = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(starts)
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    first = pd.Series(t).groupby(run_id).transform("min").to_numpy()
    last = pd.Series(t).groupby(run_id).transform("max").to_numpy()
    return pd.Series(((last - first) >= min_seconds) & ~np.isnan(v), index=values.index)


def clean_instance(raw: pd.DataFrame, sensors: dict[str, dict], cfg: CleaningConfig,
                   source: str = "WELL") -> tuple[pd.DataFrame, list[dict]]:
    """Apply the three masks to one run. Returns the cleaned frame and per-sensor quality rows."""
    df = raw.copy()
    n = len(df)
    quality = []
    for spec in sensors.values():
        col = spec["column"]
        s = df[col]
        missing = int(s.isna().sum())

        oor = out_of_range_mask(s, spec["min"], spec["max"])
        s = s.mask(oor)

        frozen = pd.Series(False, index=s.index)
        if spec["kind"] in cfg.frozen_kinds and source in cfg.frozen_sources:
            frozen = frozen_mask(s, df["ts"], cfg.frozen_min_seconds)
            s = s.mask(frozen)

        valid_fraction = float(s.notna().mean()) if n else 0.0
        kept = valid_fraction >= cfg.min_sensor_coverage
        df[col] = s if kept else np.nan
        quality.append({
            "sensor": col,
            "n_rows": n,
            "n_missing": missing,
            "n_out_of_range": int(oor.sum()),
            "n_frozen": int(frozen.sum()),
            "valid_fraction": valid_fraction,
            "kept": bool(kept),
        })
    return df, quality


def resample_instance(clean: pd.DataFrame, sensor_columns: list[str], seconds: int) -> pd.DataFrame:
    """Average cleaned 1 Hz readings into ``seconds``-wide buckets.

    Each bucket keeps the last labelled ``raw_class`` it saw, so a bucket's label is the state
    of the well at the end of the bucket - the moment a detector would act on it.
    """
    bucket = clean["ts"].dt.floor(f"{seconds}s")
    grouped = clean.groupby(bucket, sort=True)
    out = grouped[sensor_columns].mean()
    out["raw_class"] = grouped["raw_class"].last()  # groupby.last skips NaN
    out["n_obs"] = grouped.size()
    out.index.name = "ts"
    return out.reset_index()


def usable_sensor_count(quality: list[dict]) -> int:
    return sum(q["kept"] for q in quality)
