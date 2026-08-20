MODE_IDENTITY = "identity"
MODE_POSITIVE_UNBOUNDED_TO_UNIT = "positive_unbounded_to_unit"
_VALID_MODES = (MODE_IDENTITY, MODE_POSITIVE_UNBOUNDED_TO_UNIT)


class ScalarMetricTargetTransform:
    """Generic domain-aware transform from a raw scalar attribute value to
    the value actually used as a metric-learning (embedding-distance
    regression) target, and back. Not an intensity-specific hack -- which
    attribute uses which mode is a plain lookup by domain semantics
    (location/extent: already-bounded [0,1] fractions, "identity"; a
    positive-unbounded quantity like universal realized intensity:
    "positive_unbounded_to_unit"), so no `if attribute == "intensity"`
    branching is needed anywhere in the architecture/trainer/loss code --
    only the dataset picks a mode per attribute name via this class.

    "positive_unbounded_to_unit" maps [0, infinity) -> [0, 1) via
    d = raw / (1 + raw): monotonic, hyperparameter-free, no artificial
    maximum needs to be chosen, and stays compatible with an L2-normalized
    embedding space's bounded distance geometry (MTL_V22_REPORT.md Section
    11) -- unlike the raw value itself, which is unbounded and cannot be a
    sane regression target once embeddings live on a unit sphere."""

    def __init__(self, mode: str = MODE_IDENTITY):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode

    def forward(self, raw_value: float) -> float:
        if self.mode == MODE_IDENTITY:
            return raw_value
        return raw_value / (1.0 + raw_value)

    def inverse(self, metric_value: float, eps: float = 1e-6) -> float:
        if self.mode == MODE_IDENTITY:
            return metric_value
        d = min(metric_value, 1.0 - eps)
        return d / (1.0 - d)
