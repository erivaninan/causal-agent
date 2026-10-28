"""Non-parametric front-door adjustment for discrete treatments and mediators.

The implementation follows Pearl's front-door functional::

    E[Y | do(X=x)] = sum_m P(m | x) sum_x' E[Y | m, x'] P(x')

It supports binary or categorical treatments and mediators, numeric outcomes,
and uses a non-parametric bootstrap for uncertainty estimates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from cais.methods.causal_method import CausalMethod
from cais.models import Variables


def _sorted_levels(series: pd.Series) -> list[Any]:
    """Return stable levels for numeric or string-like discrete variables."""
    values = series.dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _frontdoor_means(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    outcome: str,
    *,
    min_cell_count: int = 1,
) -> tuple[Dict[Any, float], Dict[str, Any]]:
    """Evaluate the discrete front-door functional for every treatment level."""
    x_levels = _sorted_levels(df[treatment])
    m_levels = _sorted_levels(df[mediator])
    if len(x_levels) < 2:
        raise ValueError("Front-door adjustment requires at least two treatment levels.")
    if len(m_levels) < 2:
        raise ValueError("Front-door adjustment requires at least two mediator levels.")

    p_x = df[treatment].value_counts(normalize=True, sort=False)
    cell_counts = df.groupby([treatment, mediator], observed=False).size()
    missing_cells = []
    sparse_cells = []
    conditional_y: Dict[tuple[Any, Any], float] = {}

    for x_level in x_levels:
        for m_level in m_levels:
            count = int(cell_counts.get((x_level, m_level), 0))
            if count == 0:
                missing_cells.append({"treatment": x_level, "mediator": m_level})
                continue
            if count < min_cell_count:
                sparse_cells.append(
                    {"treatment": x_level, "mediator": m_level, "count": count}
                )
            mask = (df[treatment] == x_level) & (df[mediator] == m_level)
            conditional_y[(x_level, m_level)] = float(df.loc[mask, outcome].mean())

    if missing_cells:
        raise ValueError(
            "Front-door positivity is violated: some treatment-mediator cells "
            f"are empty: {missing_cells}."
        )

    # Back-door adjustment of M -> Y by X.
    y_do_m = {
        m_level: float(
            sum(
                conditional_y[(x_level, m_level)] * float(p_x.loc[x_level])
                for x_level in x_levels
            )
        )
        for m_level in m_levels
    }

    interventional_means: Dict[Any, float] = {}
    mediator_probabilities: Dict[str, float] = {}
    for x_level in x_levels:
        subset = df.loc[df[treatment] == x_level, mediator]
        p_m_given_x = subset.value_counts(normalize=True, sort=False)
        interventional_means[x_level] = float(
            sum(float(p_m_given_x.get(m_level, 0.0)) * y_do_m[m_level] for m_level in m_levels)
        )
        for m_level in m_levels:
            mediator_probabilities[f"P({mediator}={m_level}|{treatment}={x_level})"] = float(
                p_m_given_x.get(m_level, 0.0)
            )

    diagnostics = {
        "treatment_levels": x_levels,
        "mediator_levels": m_levels,
        "treatment_probabilities": {str(k): float(v) for k, v in p_x.items()},
        "mediator_probabilities": mediator_probabilities,
        "mediator_interventional_outcome_means": {
            str(k): float(v) for k, v in y_do_m.items()
        },
        "cell_counts": {
            f"{treatment}={x_level},{mediator}={m_level}": int(
                cell_counts.get((x_level, m_level), 0)
            )
            for x_level in x_levels
            for m_level in m_levels
        },
        "sparse_cells": sparse_cells,
    }
    return interventional_means, diagnostics


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Optional[list[str]] = None,
    mediator_variable: Optional[str] = None,
    mediator: Optional[str] = None,
    treatment_reference_level: Optional[Any] = None,
    treatment_comparison_level: Optional[Any] = None,
    n_bootstrap: int = 500,
    random_state: Optional[int] = 731460,
    min_cell_count: int = 5,
    **_: Any,
) -> Dict[str, Any]:
    """Estimate a discrete front-door average treatment contrast.

    ``covariates`` are intentionally unused: the canonical front-door formula
    adjusts the mediator-outcome relationship for the treatment itself.
    """
    mediator_name = mediator_variable or mediator
    if not mediator_name:
        raise ValueError("Front-door adjustment requires a mediator variable.")

    required = [treatment, mediator_name, outcome]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required front-door columns: {missing}")

    analysis = df[required].dropna().copy()
    if analysis.empty:
        raise ValueError("No complete observations remain for front-door adjustment.")
    if not pd.api.types.is_numeric_dtype(analysis[outcome]):
        raise ValueError("The front-door outcome must be numeric.")

    treatment_levels = _sorted_levels(analysis[treatment])
    if len(treatment_levels) != 2:
        raise ValueError(
            "This front-door implementation currently requires a binary treatment; "
            f"found {len(treatment_levels)} levels."
        )
    if analysis[mediator_name].nunique() > 20:
        raise ValueError(
            "This front-door implementation currently requires a discrete mediator "
            "with at most 20 levels."
        )

    reference = (
        treatment_reference_level
        if treatment_reference_level is not None
        else treatment_levels[0]
    )
    if reference not in treatment_levels:
        raise ValueError(f"Unknown treatment reference level: {reference!r}.")
    comparison = treatment_comparison_level
    if comparison is None:
        comparison = next(level for level in treatment_levels if level != reference)
    if comparison not in treatment_levels or comparison == reference:
        raise ValueError(f"Invalid treatment comparison level: {comparison!r}.")

    means, diagnostics = _frontdoor_means(
        analysis,
        treatment,
        mediator_name,
        outcome,
        min_cell_count=min_cell_count,
    )
    effect = float(means[comparison] - means[reference])

    bootstrap_effects: list[float] = []
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        for _iteration in range(n_bootstrap):
            indices = rng.integers(0, len(analysis), size=len(analysis))
            sample = analysis.iloc[indices]
            try:
                sample_means, _ = _frontdoor_means(
                    sample,
                    treatment,
                    mediator_name,
                    outcome,
                    min_cell_count=1,
                )
            except ValueError:
                continue
            bootstrap_effects.append(
                float(sample_means[comparison] - sample_means[reference])
            )

    if n_bootstrap > 0 and len(bootstrap_effects) < max(30, int(0.8 * n_bootstrap)):
        raise ValueError(
            "Too many bootstrap samples violated front-door positivity; "
            "the data do not provide stable support for inference."
        )

    if bootstrap_effects:
        bootstrap_array = np.asarray(bootstrap_effects)
        standard_error = float(np.std(bootstrap_array, ddof=1))
        confidence_interval = tuple(
            float(value) for value in np.quantile(bootstrap_array, [0.025, 0.975])
        )
        p_value = (
            float(2 * stats.norm.sf(abs(effect / standard_error)))
            if standard_error > 0
            else (0.0 if effect != 0 else 1.0)
        )
    else:
        standard_error = None
        confidence_interval = (None, None)
        p_value = None

    diagnostics.update(
        {
            "n_observations": len(analysis),
            "reference_level": reference,
            "comparison_level": comparison,
            "interventional_outcome_means": {
                str(level): float(value) for level, value in means.items()
            },
            "bootstrap_requested": n_bootstrap,
            "bootstrap_successful": len(bootstrap_effects),
            "identification_warning": (
                "The estimate is causal only if the front-door criterion holds: full "
                "mediation, no unblocked treatment-mediator confounding, and all "
                "mediator-outcome back-door paths blocked by treatment."
            ),
        }
    )

    return {
        "effect_estimate": effect,
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "method_used": "Front-door Adjustment",
        "diagnostics": diagnostics,
        "raw_results": {
            "interventional_outcome_means": diagnostics["interventional_outcome_means"],
            "contrast": f"{comparison} - {reference}",
        },
    }


class FrontdoorAdjustment(CausalMethod):
    """Pipeline adapter for the discrete front-door estimator."""

    name = "Frontdoor Adjustment"
    description = (
        "Identifies the effect of a binary treatment on a numeric outcome through "
        "an observed discrete mediator using Pearl's front-door adjustment formula."
    )
    assumptions = [
        "All directed treatment-outcome paths pass through the mediator (full mediation).",
        "There is no unblocked back-door path from treatment to mediator.",
        "Treatment blocks every back-door path from mediator to outcome.",
        "All required treatment-mediator combinations have positive probability.",
    ]

    def validate_assumptions(self, df: pd.DataFrame, variables: Variables) -> Dict[str, Any]:
        mediator = variables.mediator_variable
        required = [variables.treatment_variable, mediator, variables.outcome_variable]
        missing_names = [name for name in required if not name]
        missing_columns = self.check_missing(df, [name for name in required if name])
        return {
            "assumptions_valid": not missing_names and not missing_columns,
            "failed_assumptions": (
                ["Treatment, mediator, and outcome must be identified."]
                if missing_names
                else []
            ),
            "warnings": [
                "Graphical front-door assumptions require substantive justification; "
                "they cannot be established from the observed distribution alone."
            ],
            "suggestions": [],
            "missing_columns": sorted(missing_columns),
        }

    def estimate_effect(
        self,
        df: pd.DataFrame,
        variables: Variables,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        return estimate_effect(
            df=df,
            treatment=variables.treatment_variable,
            outcome=variables.outcome_variable,
            mediator_variable=variables.mediator_variable,
            treatment_reference_level=variables.treatment_reference_level,
        )
