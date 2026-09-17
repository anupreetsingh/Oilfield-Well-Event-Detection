"""Synthetic WITSML daily drilling reports for offline demos and tests.

The real Volve reports need an Equinor data-village account, so CI and first-time users run
the NPT pipeline and the copilot against these instead. Wellbores are named ``NO 15/9-S-<n>``
("S" for synthetic) so they can never be mistaken for real Volve wells. A ``npt_truth.csv``
written next to the XML records the cause each NPT remark was generated from.

    python -m oilfield.npt.synthetic --out data/raw/volve_synthetic --wellbores 8 --days 40
"""

from __future__ import annotations

import argparse
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from oilfield.npt.witsml import wellbore_slug

NS = "http://www.witsml.org/schemas/1series"

PRODUCTIVE = [
    ("drilling", "drilling -- drill", [
        'Drilled {hole} hole from {md0} m to {md1} m MD. Flow {flow} lpm, {rpm} rpm, WOB {wob} t.',
        'Drilled ahead {hole} section from {md0} m to {md1} m. Took surveys every stand.',
        'Rotary drilled from {md0} m to {md1} m MD with {rpm} rpm. Pumped hi-vis sweep at {md1} m.',
    ]),
    ("drilling", "drilling -- trip", [
        'POOH with {hole} BHA from {md1} m to surface. Hole in good condition.',
        'RIH with {hole} drilling BHA from surface to {md0} m. Filled pipe every 20 stands.',
        'Tripped in hole to {md1} m, washed down last stand.',
    ]),
    ("drilling", "drilling -- circulating conditioning", [
        'Circulated bottoms up at {md1} m, {flow} lpm. Shakers clean.',
        'Circulated and conditioned mud prior to POOH. Mud weight in/out 1.35 sg.',
    ]),
    ("casing", "drilling -- casing", [
        'Ran 9 5/8" casing from surface to {md0} m. Filled every 5 joints.',
        'Made up casing hanger and landed casing at {md1} m.',
    ]),
    ("drilling", "drilling -- bop activities", [
        'Pressure tested BOP to 345 bar, all tests OK.',
        'Function tested BOP from both panels.',
    ]),
    ("evaluation", "formation evaluation -- log", [
        'Logged with wireline from {md1} m to {md0} m. Good data quality.',
        'Recorded LWD memory data while POOH from {md1} m.',
    ]),
]

NPT = {
    "stuck_pipe": (["other", "other", "fish"], [
        'Pipe stuck at {md1} m while picking up. Worked pipe with max overpull {op} t, no go.',
        'Got stuck at {md1} m. Jarred up and down, pipe came free after {n} attempts.',
        'Packed off at {md1} m with loss of returns. Worked string free with {op} t overpull.',
        'Unable to pull string above {md1} m, suspected differential sticking. Spotted pill and jarred.',
    ]),
    "fishing": (["fish", "fish", "other"], [
        'Made up overshot and RIH to {md1} m to fish lost BHA.',
        'Fishing for junk at {md1} m with junk basket. Recovered pieces of bit cone.',
        'Engaged fish at {md1} m, jarred and pulled free. POOH with fish.',
    ]),
    "well_control": (["well control", "other"], [
        'Observed {n} m3 gain in pit, shut in well. Recorded SIDPP {sidpp} bar.',
        'Kick taken at {md1} m. Circulated out kick through choke using drillers method.',
        'Total losses to formation at {md1} m. Pumped LCM pill and monitored well.',
        'Influx detected on flow check. Shut in well and weighted up mud to 1.45 sg.',
    ]),
    "downhole_tool_failure": (["repair", "other"], [
        'MWD lost signal at {md1} m. Troubleshot pulser, no pulses. POOH to change tool.',
        'Mud motor failure observed at {md1} m, no differential pressure. Decided to POOH.',
        'RSS stopped working, unable to steer. Pulled out to change out RSS.',
        'LWD tool failure, no real time data from {md0} m.',
    ]),
    "rig_repair": (["repair", "repair", "maintain"], [
        'Repaired top drive, hydraulic leak on IBOP actuator.',
        'Drawworks electrical fault. Electrician troubleshooting, repaired faulty card.',
        'Repaired mud pump no. {n} liner washout.',
        'Iron roughneck failure, changed out hydraulic hose and tested.',
        'Changed out saver sub on top drive due to damaged threads.',
    ]),
    "wait_on_weather": (["wait"], [
        'WOW. Wind {wind} knots, heave {heave} m, crane operations suspended.',
        'Waiting on weather, wave height {heave} m. Unable to land casing joints from boat.',
        'Rig on standby due to storm, wind gusting {wind} knots.',
    ]),
    "wait_on_equipment": (["wait", "wait", "other"], [
        'Waiting on cement unit spares from shore.',
        'Waited for supply boat with casing equipment.',
        'Awaiting replacement MWD tool from town.',
        'Waiting on orders from town regarding sidetrack plan.',
    ]),
    "other": (["other"], [
        'Held safety stand down meeting with all crews after near miss.',
        'Slipped and cut drill line, {n} m.',
        'Rig move delayed by permit, idle time.',
    ]),
}
CAUSE_WEIGHTS = {"stuck_pipe": 3, "fishing": 1, "well_control": 2, "downhole_tool_failure": 2, "rig_repair": 4,
                 "wait_on_weather": 3, "wait_on_equipment": 2, "other": 2}


def _fill(template: str, rng: random.Random, md: int) -> str:
    return template.format(
        hole=rng.choice(['12 1/4"', '8 1/2"', '17 1/2"']), md0=md, md1=md + rng.randint(20, 300),
        flow=rng.randint(2000, 4200), rpm=rng.randint(80, 180), wob=rng.randint(5, 25), op=rng.randint(20, 90),
        n=rng.randint(2, 9), sidpp=rng.randint(10, 60), wind=rng.randint(35, 60), heave=round(rng.uniform(3, 7), 1),
    )


def _activity_xml(t0: datetime, t1: datetime, md: int, phase: str, code: str, comment: str) -> str:
    state = "fail" if code.startswith("interruption") else "ok"
    return (
        "    <activity>\n"
        f"      <dTimStart>{t0:%Y-%m-%dT%H:%M:%S}+01:00</dTimStart>\n"
        f"      <dTimEnd>{t1:%Y-%m-%dT%H:%M:%S}+01:00</dTimEnd>\n"
        f'      <md uom="m" datum="RKB">{md}</md>\n'
        f"      <phase>{phase}</phase>\n"
        f"      <proprietaryCode>{escape(code)}</proprietaryCode>\n"
        f"      <state>{state}</state>\n"
        f"      <comments>{escape(comment)}</comments>\n"
        "    </activity>\n"
    )


def generate(out_dir: str | Path, n_wellbores: int = 8, n_days: int = 40, seed: int = 0,
             start: date = date(2014, 1, 6)) -> pd.DataFrame:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for w in range(1, n_wellbores + 1):
        wellbore = f"NO 15/9-S-{w}" + (" A" if w % 3 == 0 else "")
        slug = wellbore_slug(wellbore)
        md = rng.randint(400, 1500)
        day0 = start + timedelta(days=rng.randint(0, 200))
        for d in range(n_days):
            day = day0 + timedelta(days=d)
            t = datetime.combine(day, datetime.min.time())
            end_of_day = t + timedelta(days=1)
            acts, seq, npt_hours = [], 0, 0.0
            while t < end_of_day:
                hours = rng.choice([0.5, 1, 1.5, 2, 3, 4, 6])
                t1 = min(t + timedelta(hours=hours), end_of_day)
                seq += 1
                if rng.random() < 0.12:
                    cause = rng.choices(list(CAUSE_WEIGHTS), weights=list(CAUSE_WEIGHTS.values()))[0]
                    minors, templates = NPT[cause]
                    code = f"interruption -- {rng.choice(minors)}"
                    phase, comment = "drilling", _fill(rng.choice(templates), rng, md)
                    truth.append({"activity_id": f"{slug}:{day:%Y%m%d}:{seq:02d}", "cause": cause})
                    npt_hours += (t1 - t).total_seconds() / 3600
                else:
                    phase, code, templates = rng.choice(PRODUCTIVE)
                    comment = _fill(rng.choice(templates), rng, md)
                    if code == "drilling -- drill":
                        md += rng.randint(20, 300)
                acts.append(_activity_xml(t, t1, md, phase, code, comment))
                t = t1
            summary = f"Drilled to {md} m MD." + (f" {npt_hours:.1f} h NPT." if npt_hours else "")
            xml = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                f'<drillReports xmlns="{NS}" version="1.4.0.0">\n'
                f'  <drillReport uidWell="S{w}" uidWellbore="S{w}" uid="S{w}-{day:%Y%m%d}">\n'
                f"    <nameWell>NO 15/9-S-{w}</nameWell>\n"
                f"    <nameWellbore>{wellbore}</nameWellbore>\n"
                f"    <dTimStart>{day:%Y-%m-%d}T00:00:00+01:00</dTimStart>\n"
                f"    <dTimEnd>{day + timedelta(days=1):%Y-%m-%d}T00:00:00+01:00</dTimEnd>\n"
                "    <statusInfo>\n"
                f'      <md uom="m">{md}</md>\n'
                f"      <sum24Hr>{escape(summary)}</sum24Hr>\n"
                f"      <forecast24Hr>{escape(rng.choice(['Continue drilling ahead.', 'POOH and run casing.', 'Circulate and condition, then log.']))}</forecast24Hr>\n"
                "    </statusInfo>\n"
                + "".join(acts)
                + "  </drillReport>\n</drillReports>\n"
            )
            (out / f"{slug}_{day:%Y_%m_%d}.xml").write_text(xml)
    truth_df = pd.DataFrame(truth)
    truth_df.to_csv(out / "npt_truth.csv", index=False)
    return truth_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/volve_synthetic")
    parser.add_argument("--wellbores", type=int, default=8)
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    truth = generate(args.out, args.wellbores, args.days, args.seed)
    print(f"wrote {args.wellbores * args.days} reports with {len(truth)} NPT activities to {args.out}")


if __name__ == "__main__":
    main()
