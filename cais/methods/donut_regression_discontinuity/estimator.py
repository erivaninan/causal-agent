"""Local-linear sharp RDD with a prespecified exclusion zone at the cutoff."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cais.methods.causal_method import CausalMethod


def _default_bandwidth(running: pd.Series) -> float:
    running_range = float(running.max() - running.min())
    if not np.isfinite(running_range) or running_range <= 0:
        raise ValueError("The running variable must have a positive finite range.")
    return 0.1 * running_range


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]] = None,
    running_variable: Optional[str] = None,
    cutoff_value: Optional[float] = None,
    treat_above_cutoff: bool = True,
    bandwidth: Optional[float] = None,
    donut_radius: Optional[float] = None,
    verify_sharp_assignment: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate an extrapolated sharp-RD jump after excluding the cutoff vicinity.

    Observations are retained when ``donut_radius < abs(X-c) <= bandwidth``.
    The donut radius must be prespecified; this function never chooses it from
    the observed outcome or treatment effect.
    """
    covariates = covariates or []
    if not running_variable:
        raise ValueError("Donut RDD requires a running_variable.")
    if cutoff_value is None or not np.isfinite(cutoff_value):
        raise ValueError("Donut RDD requires a finite cutoff_value.")
    if donut_radius is None or not np.isfinite(donut_radius) or donut_radius <= 0:
        raise ValueError("donut_radius must be a prespecified positive finite number.")

    required = [outcome, running_variable, *covariates]
    if verify_sharp_assignment:
        required.append(treatment)
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Donut RDD is missing required columns: {missing}")

    analysis = df[required].dropna().copy()
    if analysis.empty:
        raise ValueError("No complete observations are available for Donut RDD.")
    final_bandwidth = bandwidth if bandwidth is not None else _default_bandwidth(analysis[running_variable])
    if not np.isfinite(final_bandwidth) or final_bandwidth <= 0:
        raise ValueError("bandwidth must be a positive finite number.")
    if donut_radius >= final_bandwidth:
        raise ValueError("donut_radius must be strictly smaller than bandwidth.")

    analysis["running_centered"] = analysis[running_variable].astype(float) - float(cutoff_value)
    distance = analysis["running_centered"].abs()
    in_outer_window = distance <= final_bandwidth
    in_donut = distance <= donut_radius
    n_in_outer_window = int(in_outer_window.sum())
    n_excluded = int((in_outer_window & in_donut).sum())
    analysis = analysis[in_outer_window & ~in_donut].copy()
    if analysis.empty:
        raise ValueError("No observations remain after applying the Donut RDD exclusion zone.")

    if treat_above_cutoff:
        analysis["assignment"] = (analysis["running_centered"] > 0).astype(float)
    else:
        analysis["assignment"] = (analysis["running_centered"] < 0).astype(float)
    side_counts = analysis["assignment"].value_counts()
    if side_counts.get(0.0, 0) < 3 or side_counts.get(1.0, 0) < 3:
        raise ValueError("Donut RDD requires at least three retained observations on each side.")

    assignment_compliance = None
    if verify_sharp_assignment:
        if not np.isin(analysis[treatment].unique(), [0, 1]).all():
            raise ValueError("Sharp Donut RDD requires treatment coded 0/1.")
        assignment_compliance = float(
            np.mean(analysis[treatment].astype(float) == analysis["assignment"])
        )
        if assignment_compliance < 1.0:
            raise ValueError(
                "Treatment receipt is not deterministic outside the donut; consider Fuzzy RDD "
                "or set verify_sharp_assignment=False only when treatment is assignment itself."
            )

    analysis["assignment_x_running"] = analysis["assignment"] * analysis["running_centered"]
    design_columns = ["assignment", "running_centered", "assignment_x_running", *covariates]
    design = sm.add_constant(analysis[design_columns], has_constant="add")
    if len(analysis) <= design.shape[1]:
        raise ValueError("Insufficient observations for the requested Donut RDD specification.")
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
        raise ValueError("The Donut RDD local-linear design matrix is rank deficient.")

    kernel_weights = 1.0 - analysis["running_centered"].abs().to_numpy() / final_bandwidth
    kernel_weights = np.maximum(kernel_weights, np.finfo(float).eps)
    fit = sm.WLS(analysis[outcome].astype(float), design, weights=kernel_weights).fit(cov_type="HC1")
    effect = float(fit.params["assignment"])
    standard_error = float(fit.bse["assignment"])
    confidence_interval = [float(value) for value in fit.conf_int().loc["assignment"]]
    p_value = float(fit.pvalues["assignment"])

    return {
        "effect_estimate": effect,
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "method_used": "Donut RDD local linear",
        "diagnostics": {
            "cutoff_value": float(cutoff_value),
            "bandwidth_used": float(final_bandwidth),
            "donut_radius": float(donut_radius),
            "kernel": "triangular",
            "n_in_outer_window": n_in_outer_window,
            "n_excluded_by_donut": n_excluded,
            "n_retained": int(len(analysis)),
            "n_treated_side": int(side_counts.get(1.0, 0)),
            "n_control_side": int(side_counts.get(0.0, 0)),
            "sharp_assignment_compliance": assignment_compliance,
            "treat_above_cutoff": bool(treat_above_cutoff),
        },
        "warnings": [
            "The discontinuity at the cutoff is extrapolated across an excluded region; "
            "report sensitivity to multiple prespecified donut radii and bandwidths.",
            "A donut design does not by itself establish that manipulation is confined to the excluded region.",
        ],
        "raw_results": {
            "estimand": "extrapolated local sharp-RD jump at the cutoff after donut exclusion",
            "retained_region": "donut_radius < abs(X-cutoff) <= bandwidth",
            "inference": "HC1 robust local-linear WLS",
        },
    }


class DonutRDD(CausalMethod):
    """Pipeline adapter for sharp Donut RDD."""

    name = "Donut Regression Discontinuity Design"
    description = "Excludes a prespecified neighborhood around a cutoff and extrapolates local-linear fits to the cutoff."
    assumptions = [
        "Contamination or manipulation is confined to the prespecified donut region.",
        "Treatment assignment is sharp outside the excluded region.",
        "Potential-outcome regression functions remain smooth across the excluded region.",
        "No other intervention or measurement process changes at the cutoff.",
        "The donut radius was not selected from the estimated treatment effect.",
    ]

    def validate_assumptions(self, df: pd.DataFrame, variables: Any) -> Dict[str, Any]:
        required = [
            variables.treatment_variable,
            variables.outcome_variable,
            variables.running_variable,
        ]
        missing = [name for name in required if not name or name not in df.columns]
        return {
            "assumptions_valid": not missing,
            "failed_assumptions": missing,
            "warnings": ["Localized contamination and extrapolation require substantive review."],
            "suggestions": [],
        }

    def estimate_effect(self, df: pd.DataFrame, variables: Any, query: Optional[str] = None) -> Dict[str, Any]:
        donut_radius = getattr(variables, "donut_radius", None)
        return estimate_effect(
            df=df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            covariates=variables.confounders or [],
            running_variable=variables.running_variable,
            cutoff_value=variables.cutoff_value,
            treat_above_cutoff=(
                True if variables.treat_above_cutoff is None else variables.treat_above_cutoff
            ),
            donut_radius=donut_radius,
        )
