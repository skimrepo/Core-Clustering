import pytest

from core_clustering.target_transforms import ScalarMetricTargetTransform


def test_identity_mode_is_a_pure_passthrough():
    t = ScalarMetricTargetTransform(mode="identity")
    assert t.forward(0.0) == 0.0
    assert t.forward(3.7) == 3.7
    assert t.forward(-1.5) == -1.5
    assert t.inverse(3.7) == 3.7


def test_positive_unbounded_to_unit_known_values():
    t = ScalarMetricTargetTransform(mode="positive_unbounded_to_unit")
    assert t.forward(0.0) == pytest.approx(0.0)
    assert t.forward(1.0) == pytest.approx(0.5)
    assert t.forward(3.0) == pytest.approx(0.75)
    assert t.forward(6.0) == pytest.approx(6 / 7)


def test_positive_unbounded_to_unit_is_monotonic_and_bounded():
    t = ScalarMetricTargetTransform(mode="positive_unbounded_to_unit")
    values = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0]
    outputs = [t.forward(v) for v in values]
    assert outputs == sorted(outputs)
    assert all(0.0 <= o < 1.0 for o in outputs)


def test_positive_unbounded_to_unit_inverse_round_trips():
    t = ScalarMetricTargetTransform(mode="positive_unbounded_to_unit")
    for raw in (0.0, 0.5, 1.0, 3.0, 6.0, 42.0):
        metric = t.forward(raw)
        assert t.inverse(metric) == pytest.approx(raw, rel=1e-6, abs=1e-9)


def test_positive_unbounded_to_unit_inverse_clamps_near_one_safely():
    t = ScalarMetricTargetTransform(mode="positive_unbounded_to_unit")
    # metric values at/above 1.0 are out-of-domain (unreachable by forward())
    # but must not raise or return inf/NaN -- callers may see values this
    # close to 1 from a model's raw prediction.
    result = t.inverse(1.0)
    assert result == result  # not NaN
    assert result != float("inf")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        ScalarMetricTargetTransform(mode="not_a_real_mode")
