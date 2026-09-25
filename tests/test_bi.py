from oilfield.bi.export import star_schema
from oilfield.npt.synthetic import generate
from oilfield.npt.witsml import parse_ddr_dir


def test_star_schema_keys_and_totals(tmp_path):
    generate(tmp_path, n_wellbores=2, n_days=6, seed=4)
    activities, _ = parse_ddr_dir(tmp_path)
    npt = activities[activities.is_npt].assign(cause="rig_repair", confidence=0.9)
    t = star_schema(activities, npt)

    assert t["fact_activity"]["hours"].sum() == 2 * 6 * 24
    assert t["fact_npt"]["hours"].sum() == round(npt["duration_h"].sum(), 3)
    for fact in ("fact_activity", "fact_npt"):
        assert t[fact]["wellbore_key"].isin(t["dim_wellbore"]["wellbore_key"]).all()
        assert t[fact]["date_key"].isin(t["dim_date"]["date_key"]).all()
    assert t["fact_npt"]["cause_key"].isin(t["dim_cause"]["cause_key"]).all()
