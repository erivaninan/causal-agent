import numpy as np
import pandas as pd
import pytest

from cais.methods import METHOD_MAPPING
from cais.methods.donut_regression_discontinuity.estimator import estimate_effect


def _donut_data(seed=21, n=5000, effect=2.5):
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1.0, 1.0, n)
    treatment = (running >= 0).astype(int)
    outcome = effect * treatment + 0.7 * running + rng.normal(0, 0.5, n)
    contaminated = np.abs(running) <= 0.08
    outcome[contaminated] += np.where(running[contaminated] >= 0, 8.0, -8.0)
    return pd.DataFrame({"running": running, "treatment": treatment, "outcome": outcome})


def test_donut_rdd_recovers_effect_and_reports_exclusion():
    result = estimate_effect(
        _donut_data(),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff_value=0.0,
        bandwidth=0.5,
        donut_radius=0.1,
    )

    assert result["effect_estimate"] == pytest.approx(2.5, abs=0.25)
    assert result["standard_error"] > 0
    assert result["diagnostics"]["n_excluded_by_donut"] > 0
    assert result["diagnostics"]["n_retained"] < result["diagnostics"]["n_in_outer_window"]
    assert result["warnings"]
    assert METHOD_MAPPING["donut_regression_discontinuity_design"] is estimate_effect


def test_donut_rdd_supports_treatment_below_cutoff():
    df = _donut_data(seed=22)
    df["running"] *= -1

    result = estimate_effect(
        df,
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff_value=0.0,
        treat_above_cutoff=False,
        bandwidth=0.5,
        donut_radius=0.1,
    )

    assert result["effect_estimate"] == pytest.approx(2.5, abs=0.25)
    assert result["diagnostics"]["treat_above_cutoff"] is False


@pytest.mark.parametrize(
    ("donut_radius", "bandwidth", "message"),
    [
        (None, 0.5, "donut_radius"),
        (0.0, 0.5, "donut_radius"),
        (0.5, 0.5, "strictly smaller"),
        (0.6, 0.5, "strictly smaller"),
    ],
)
def test_donut_rdd_rejects_invalid_geometry(donut_radius, bandwidth, message):
    with pytest.raises(ValueError, match=message):
        estimate_effect(
            _donut_data(seed=23),
            treatment="treatment",
            outcome="outcome",
            running_variable="running",
            cutoff_value=0.0,
            bandwidth=bandwidth,
            donut_radius=donut_radius,
        )


def test_donut_rdd_rejects_nonsharp_assignment_outside_donut():
    df = _donut_data(seed=24)
    outside = df.index[(df["running"] > 0.2) & (df["running"] < 0.5)][0]
    df.loc[outside, "treatment"] = 0

    with pytest.raises(ValueError, match="not deterministic"):
        estimate_effect(
            df,
            treatment="treatment",
            outcome="outcome",
            running_variable="running",
            cutoff_value=0.0,
            bandwidth=0.5,
            donut_radius=0.1,
        )
