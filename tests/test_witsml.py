import pandas as pd

from oilfield.npt.synthetic import generate
from oilfield.npt.witsml import parse_ddr_dir, parse_ddr_file, wellbore_slug

DDR = """<?xml version="1.0" encoding="utf-8"?>
<drillReports xmlns="http://www.witsml.org/schemas/1series" version="1.4.0.0">
  <drillReport uidWell="W1" uidWellbore="W1B" uid="r1">
    <nameWell>NO 15/9-F-12</nameWell>
    <nameWellbore>NO 15/9-F-12</nameWellbore>
    <dTimStart>2008-02-14T00:00:00+01:00</dTimStart>
    <dTimEnd>2008-02-15T00:00:00+01:00</dTimEnd>
    <statusInfo>
      <md uom="m">2550</md>
      <sum24Hr>Drilled 12 1/4" hole to 2550 m. Repaired top drive.</sum24Hr>
      <forecast24Hr>Drill ahead to section TD.</forecast24Hr>
    </statusInfo>
    <activity>
      <dTimStart>2008-02-14T00:00:00+01:00</dTimStart>
      <dTimEnd>2008-02-14T06:30:00+01:00</dTimEnd>
      <md uom="m" datum="RKB">2480</md>
      <phase>drilling</phase>
      <proprietaryCode>drilling -- drill</proprietaryCode>
      <state>ok</state>
      <comments>Drilled 12 1/4" hole from 2410 m to 2480 m.</comments>
    </activity>
    <activity>
      <dTimStart>2008-02-14T06:30:00+01:00</dTimStart>
      <dTimEnd>2008-02-14T09:00:00+01:00</dTimEnd>
      <md uom="m" datum="RKB">2480</md>
      <phase>drilling</phase>
      <proprietaryCode>interruption -- repair</proprietaryCode>
      <state>fail</state>
      <comments>Repaired top drive,
        hydraulic leak on IBOP actuator.</comments>
    </activity>
  </drillReport>
</drillReports>
"""


def test_parse_activities_and_status(tmp_path):
    path = tmp_path / "F-12_2008_02_14.xml"
    path.write_text(DDR)
    activities, reports = parse_ddr_file(path)
    assert [a["activity_id"] for a in activities] == ["F-12:20080214:01", "F-12:20080214:02"]
    npt = activities[1]
    assert npt["is_npt"] and npt["code_minor"] == "repair"
    assert npt["duration_h"] == 2.5
    assert npt["start_time"] == pd.Timestamp("2008-02-14 06:30")  # rig local time
    assert npt["comments"] == "Repaired top drive, hydraulic leak on IBOP actuator."
    assert reports[0]["forecast_24h"] == "Drill ahead to section TD." and reports[0]["md_m"] == 2550


def test_revised_report_replaces_earlier_one(tmp_path):
    (tmp_path / "a_v1.xml").write_text(DDR)
    (tmp_path / "a_v2.xml").write_text(DDR.replace("Repaired top drive,", "Repaired top drive (revised),"))
    activities, reports = parse_ddr_dir(tmp_path)
    assert len(reports) == 1 and len(activities) == 2
    assert "revised" in activities.iloc[1]["comments"]


def test_wellbore_slugs():
    assert wellbore_slug("NO 15/9-F-12") == "F-12"
    assert wellbore_slug("NO 15/9-F-1 C") == "F-1C"
    assert wellbore_slug("15/9-19 BT2") == "19BT2"


def test_synthetic_reports_round_trip(tmp_path):
    truth = generate(tmp_path, n_wellbores=3, n_days=5, seed=1)
    activities, reports = parse_ddr_dir(tmp_path)
    assert len(reports) == 15
    assert set(truth["activity_id"]) == set(activities.loc[activities["is_npt"], "activity_id"])
    # every report covers the full day
    assert activities.groupby(["wellbore", "report_date"])["duration_h"].sum().round(6).eq(24).all()
