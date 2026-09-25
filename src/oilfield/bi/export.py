"""Write the NPT star schema Power BI reads (see powerbi/README.md).

    fact_activity  one row per reported activity: hours, NPT flag     -> denominators
    fact_npt       one row per NPT activity: hours, cause, confidence -> NPT hours by cause
    dim_wellbore, dim_cause, dim_date

    python -m oilfield.bi.export            # CSVs into data/powerbi/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from oilfield.config import resolve_path
from oilfield.npt.labeling import CAUSE_LABELS, CAUSES


def _date_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y%m%d").astype(int)


def star_schema(activities: pd.DataFrame, npt_events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    wellbores = (
        activities.groupby("wellbore")
        .agg(well=("well", "first"), wellbore_name=("wellbore_name", "first"),
             first_report=("report_date", "min"), last_report=("report_date", "max"))
        .reset_index()
        .reset_index(names="wellbore_key")
    )
    wb_key = wellbores.set_index("wellbore")["wellbore_key"]
    causes = pd.DataFrame({"cause_key": range(len(CAUSES)), "cause": CAUSES,
                           "cause_label": [CAUSE_LABELS[c] for c in CAUSES], "sort_order": range(len(CAUSES))})
    cause_key = causes.set_index("cause")["cause_key"]

    days = pd.date_range(pd.to_datetime(activities["report_date"]).min(), pd.to_datetime(activities["report_date"]).max())
    dates = pd.DataFrame({
        "date_key": days.strftime("%Y%m%d").astype(int), "date": days.date, "year": days.year,
        "month": days.month, "month_name": days.strftime("%b"), "iso_week": days.isocalendar().week.to_numpy(),
    })

    fact_activity = pd.DataFrame({
        "activity_id": activities["activity_id"],
        "wellbore_key": activities["wellbore"].map(wb_key),
        "date_key": _date_key(activities["report_date"]),
        "hours": activities["duration_h"].round(3),
        "is_npt": activities["is_npt"].astype(int),
        "phase": activities["phase"],
        "code": activities["code"],
    })
    fact_npt = pd.DataFrame({
        "activity_id": npt_events["activity_id"],
        "wellbore_key": npt_events["wellbore"].map(wb_key),
        "date_key": _date_key(npt_events["report_date"]),
        "cause_key": npt_events["cause"].map(cause_key),
        "hours": npt_events["duration_h"].round(3),
        "confidence": npt_events["confidence"],
        "md_m": npt_events["md_m"],
        "remark": npt_events["comments"],
    })
    return {"fact_activity": fact_activity, "fact_npt": fact_npt, "dim_wellbore": wellbores,
            "dim_cause": causes, "dim_date": dates}


def write(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npt-dir", default="data/npt", help="where oilfield.npt.tag wrote its Parquet files")
    parser.add_argument("--out", default="data/powerbi")
    args = parser.parse_args(argv)
    npt_dir = resolve_path(args.npt_dir)
    tables = star_schema(pd.read_parquet(npt_dir / "activities.parquet"), pd.read_parquet(npt_dir / "npt_events.parquet"))
    out = write(tables, resolve_path(args.out))
    print(f"wrote {', '.join(tables)} to {out}")


if __name__ == "__main__":
    main()
