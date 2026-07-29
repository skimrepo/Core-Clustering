from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES


def test_redlamp_anomaly_types_has_twelve_entries_normal_first():
    assert len(REDLAMP_ANOMALY_TYPES) == 12
    assert REDLAMP_ANOMALY_TYPES[0] == "normal"


def test_redlamp_anomaly_types_matches_redlamp_main_py_order():
    # Verbatim copy of RedLamp's own default anomaly_types list
    # (RedLamp/main.py:480). main.anomaly_scoreing() hardcodes "class index 0
    # is normal", so this exact order (not just the set) must match for a
    # cross-loaded checkpoint's classifier output to mean the same thing.
    assert REDLAMP_ANOMALY_TYPES == [
        "normal", "spike", "flip", "speedup", "noise", "cutoff",
        "average", "scale", "wander", "contextual", "upsidedown", "mixture",
    ]
