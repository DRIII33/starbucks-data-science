"""
================================================================================
PHASE 3A: CAUSAL INFERENCE MODELING - COMPLETE PRODUCTION CODE
================================================================================

GOOGLE COLAB
Copy/paste this entire file into a single Colab cell.

PURPOSE
-------
Estimate the causal effect of PROMO_20 versus CONTROL on daily_net_revenue
using Double Machine Learning (DML) with EconML LinearDML and LightGBM
nuisance models.

METHODS
-------
✓ BigQuery data loading
✓ Explicit source-schema validation
✓ Binary treatment definition
✓ Explicit causal adjustment set
✓ One-hot encoding of categorical covariates
✓ DoWhy causal graph construction
✓ Backdoor identification
✓ EconML LinearDML
✓ LightGBM outcome nuisance model
✓ LightGBM treatment/propensity nuisance model
✓ Model-based inference
✓ ATE + 95% confidence interval
✓ Statistical significance
✓ CATE / heterogeneous treatment effects
✓ Market-segment HTE
✓ Product-category HTE
✓ Covariate balance diagnostics
✓ Propensity-score overlap diagnostics
✓ Causal graph visualization
✓ ATE comparison visualization
✓ HTE visualization
✓ CATE distribution
✓ Outcome distribution
✓ Propensity overlap visualization
✓ CSV + JSON output artifacts

IMPORTANT
---------
This version intentionally does NOT silently replace BigQuery data with
synthetic data if BigQuery access fails. A causal-analysis notebook should
never silently generate portfolio results from a different dataset.

Author: Starbucks Data Science Portfolio
Date: August 2026
================================================================================
"""

# ============================================================================
# IMPORTS & GLOBAL CONFIGURATION
# ============================================================================

import os
import sys
import json
import subprocess
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reduce unnecessary TensorFlow logging if another dependency imports TF.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# BigQuery configuration
# ---------------------------------------------------------------------------

PROJECT_ID = "driiiportfolio-506303"
DATASET_NAME = "starbucks_transactions"
TABLE_NAME = "analytics_ready_promo_data"

MAX_ROWS = 100_000

# ---------------------------------------------------------------------------
# Colab output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("/content/phase_3a_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STARTUP
# ============================================================================

print("\n" + "=" * 80)
print("INITIALIZING PHASE 3A: CAUSAL INFERENCE MODELING")
print("=" * 80 + "\n")


# ============================================================================
# PACKAGE INSTALLATION
# ============================================================================

print("[1/7] Installing / verifying required libraries...")
print("-" * 80)


def ensure_package(import_name, pip_name):
    """
    Import a package if available; otherwise install it with pip.

    Parameters
    ----------
    import_name : str
        Python import path.
    pip_name : str
        pip package name.
    """
    try:
        __import__(import_name)
        print(f"  ✓ {pip_name:28s} available")
    except ImportError:
        print(f"  → Installing {pip_name}...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                pip_name,
            ]
        )
        print(f"  ✓ {pip_name:28s} installed")


required_packages = [
    ("dowhy", "dowhy"),
    ("econml", "econml"),
    ("networkx", "networkx"),
    ("pydot", "pydot"),
    ("lightgbm", "lightgbm"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("google.cloud.bigquery", "google-cloud-bigquery"),
]

for import_name, pip_name in required_packages:
    ensure_package(import_name, pip_name)


# ============================================================================
# IMPORTS
# ============================================================================

print("\n[2/7] Importing libraries...")
print("-" * 80)

from google.cloud import bigquery

from dowhy import CausalModel

from econml.dml import LinearDML

from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
)

from scipy.stats import norm

import matplotlib.pyplot as plt
import networkx as nx
import pydot


plt.rcParams["figure.figsize"] = (12, 6)

print("  ✓ All libraries imported successfully")


# ============================================================================
# SECTION 1 — LOAD DATA FROM BIGQUERY
# ============================================================================

print("\n[3/7] Loading data from BigQuery...")
print("-" * 80)

try:

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`
    ORDER BY transaction_date, store_id, category
    LIMIT {MAX_ROWS}
    """

    print(
        f"  Query source: "
        f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"
    )

    df_analytics = client.query(query).to_dataframe()

    data_source = "BigQuery"

    print(f"  ✓ Loaded {len(df_analytics):,} rows")
    print(f"  ✓ Columns: {len(df_analytics.columns)}")

except Exception as exc:

    raise RuntimeError(
        "\n"
        "BIGQUERY DATA LOAD FAILED\n"
        "--------------------------\n"
        "The production version intentionally does not silently substitute "
        "synthetic data because doing so could create misleading causal "
        "results for a portfolio project.\n\n"
        "Verify:\n"
        "  1. Colab authentication is active.\n"
        "  2. The project is accessible.\n"
        "  3. The dataset exists.\n"
        "  4. The table exists.\n"
        "  5. Your account has permission to query it.\n\n"
        f"Requested table:\n"
        f"  {PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}\n\n"
        f"Original BigQuery error:\n{exc}"
    ) from exc


if df_analytics.empty:
    raise ValueError(
        "BigQuery returned zero rows. "
        "Causal inference cannot proceed."
    )


# ============================================================================
# SECTION 2 — SOURCE SCHEMA VALIDATION
# ============================================================================

print("\n[4/7] Validating source schema and preparing causal sample...")
print("-" * 80)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

required_columns = [
    "transaction_date",
    "store_id",
    "market_segment",
    "category",
    "treatment_group",
    "base_price",
    "unit_cost",
    "elasticity",
    "rolling_7day_net_revenue",
    "rolling_7day_units_sold",
    "daily_net_revenue",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df_analytics.columns
]

if missing_columns:

    raise ValueError(
        "The BigQuery table is missing required columns:\n"
        + "\n".join(f"  • {column}" for column in missing_columns)
    )


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

df_analytics["transaction_date"] = pd.to_datetime(
    df_analytics["transaction_date"],
    errors="coerce",
)

if df_analytics["transaction_date"].isna().any():

    invalid_dates = int(
        df_analytics["transaction_date"].isna().sum()
    )

    raise ValueError(
        f"transaction_date contains {invalid_dates:,} "
        "unparseable values."
    )


# ============================================================================
# CAUSAL SAMPLE DEFINITION
# ============================================================================

# Original analytical design:
#
# Treatment = PROMO_20
# Control   = CONTROL
#
# PROMO_33 is intentionally excluded because this is a binary
# PROMO_20-versus-CONTROL causal comparison.

df_causal = df_analytics[
    df_analytics["treatment_group"].isin(
        ["CONTROL", "PROMO_20"]
    )
].copy()


if df_causal.empty:

    raise ValueError(
        "No observations with treatment_group in "
        "{'CONTROL', 'PROMO_20'} were found."
    )


# ---------------------------------------------------------------------------
# Binary treatment
# ---------------------------------------------------------------------------

df_causal["treatment"] = (
    df_causal["treatment_group"]
    .eq("PROMO_20")
    .astype(np.int8)
)


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

df_causal["outcome"] = pd.to_numeric(
    df_causal["daily_net_revenue"],
    errors="coerce",
)


# ============================================================================
# TREATMENT / OUTCOME VALIDATION
# ============================================================================

if df_causal["treatment"].nunique() != 2:

    raise ValueError(
        "The causal sample must contain both treatment groups:\n"
        "  CONTROL\n"
        "  PROMO_20"
    )


if df_causal["outcome"].isna().any():

    invalid_outcomes = int(
        df_causal["outcome"].isna().sum()
    )

    raise ValueError(
        f"Outcome contains {invalid_outcomes:,} "
        "missing/non-numeric observations."
    )


if not np.isfinite(
    df_causal["outcome"].to_numpy(dtype=np.float64)
).all():

    raise ValueError(
        "Outcome contains infinite values."
    )


# ============================================================================
# EXPLICIT CAUSAL ADJUSTMENT SET
# ============================================================================

"""
The original implementation declared a specific confounder/adjustment set
but later constructed X using every dataframe column except treatment/outcome.

That was unsafe because it allowed variables such as:

    promo_id
    discount_pct
    daily_profit
    transaction_date
    store_id
    treatment_group

to potentially enter the DML feature matrix.

This implementation explicitly defines the variables that may enter X.
"""

numeric_adjustment_cols = [
    "base_price",
    "unit_cost",
    "elasticity",
    "rolling_7day_net_revenue",
    "rolling_7day_units_sold",
]

categorical_adjustment_cols = [
    "market_segment",
    "category",
]


# ---------------------------------------------------------------------------
# Variables explicitly excluded from X
# ---------------------------------------------------------------------------

excluded_columns = {
    "treatment_group",
    "treatment",
    "outcome",
    "daily_net_revenue",
    "discount_pct",
    "promo_id",
    "daily_profit",
    "transaction_date",
    "store_id",
}


# ============================================================================
# NUMERIC FEATURE VALIDATION
# ============================================================================

for column in numeric_adjustment_cols:

    df_causal[column] = pd.to_numeric(
        df_causal[column],
        errors="coerce",
    )


# ============================================================================
# REQUIRED-VARIABLE MISSINGNESS HANDLING
# ============================================================================

analysis_columns = (
    numeric_adjustment_cols
    + categorical_adjustment_cols
    + ["treatment", "outcome"]
)

before_rows = len(df_causal)

df_causal = df_causal.dropna(
    subset=analysis_columns
).copy()

rows_dropped = before_rows - len(df_causal)


if df_causal.empty:

    raise ValueError(
        "No complete observations remain after "
        "required-variable validation."
    )


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================

X_parts = []


# ---------------------------------------------------------------------------
# Numeric adjustment variables
# ---------------------------------------------------------------------------

numeric_part = (
    df_causal[numeric_adjustment_cols]
    .astype(np.float64)
)

X_parts.append(numeric_part)


# ---------------------------------------------------------------------------
# Categorical adjustment variables
# ---------------------------------------------------------------------------

for column in categorical_adjustment_cols:

    categorical_values = (
        df_causal[column]
        .astype("string")
    )

    dummies = pd.get_dummies(
        categorical_values,
        prefix=column,
        drop_first=True,
        dtype=np.float64,
    )

    if dummies.shape[1] == 0:

        raise ValueError(
            f"Categorical variable `{column}` "
            "produced no usable dummy variables."
        )

    X_parts.append(dummies)


# ---------------------------------------------------------------------------
# Final DML feature matrix
# ---------------------------------------------------------------------------

X_numeric = pd.concat(
    X_parts,
    axis=1,
)

X_numeric = X_numeric.astype(
    np.float64
)

X_numeric = X_numeric.replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================================
# FINAL FEATURE MATRIX VALIDATION
# ============================================================================

if X_numeric.isna().any().any():

    bad_columns = (
        X_numeric.columns[
            X_numeric.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Adjustment matrix contains NaN/infinite values "
        f"after encoding. Affected columns: {bad_columns}"
    )


# ---------------------------------------------------------------------------
# Treatment and outcome arrays
# ---------------------------------------------------------------------------

Y = df_causal["outcome"].to_numpy(
    dtype=np.float64
)

T = df_causal["treatment"].to_numpy(
    dtype=np.int8
)


# ============================================================================
# DIMENSION VALIDATION
# ============================================================================

if not (
    len(X_numeric)
    == len(Y)
    == len(T)
):

    raise RuntimeError(
        "DIMENSION VALIDATION FAILED\n"
        f"X rows: {len(X_numeric):,}\n"
        f"Y rows: {len(Y):,}\n"
        f"T rows: {len(T):,}"
    )


# ---------------------------------------------------------------------------
# Treatment variation
# ---------------------------------------------------------------------------

if T.sum() == 0:

    raise ValueError(
        "No treated observations remain."
    )

if T.sum() == len(T):

    raise ValueError(
        "No control observations remain."
    )


# ============================================================================
# SAMPLE SUMMARY
# ============================================================================

treated_count = int(
    (T == 1).sum()
)

control_count = int(
    (T == 0).sum()
)

print(
    f"  ✓ Treatment: "
    f"{treated_count:,} treated / "
    f"{control_count:,} control"
)

print(
    "  ✓ Outcome: "
    "daily_net_revenue"
)

print(
    f"  ✓ Explicit adjustment variables: "
    f"{len(numeric_adjustment_cols) + len(categorical_adjustment_cols)}"
)

print(
    f"  ✓ DML features after encoding: "
    f"{X_numeric.shape[1]}"
)

print(
    f"  ✓ Rows retained: "
    f"{len(df_causal):,}"
)

print(
    f"  ✓ Rows dropped for required fields: "
    f"{rows_dropped:,}"
)


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

control_outcome = (
    df_causal.loc[
        T == 0,
        "outcome"
    ]
)

treated_outcome = (
    df_causal.loc[
        T == 1,
        "outcome"
    ]
)

control_mean = float(
    control_outcome.mean()
)

treated_mean = float(
    treated_outcome.mean()
)

naive_ate = float(
    treated_mean - control_mean
)


print("\n  Summary Statistics:")
print("-" * 80)

print(
    f"  Control mean outcome:     "
    f"${control_mean:.2f}"
)

print(
    f"  Treatment mean outcome:   "
    f"${treated_mean:.2f}"
)

print(
    f"  Naive ATE:                "
    f"${naive_ate:.2f}"
)

print(
    "  Note: the naive difference is "
    "an unadjusted association, not a causal estimate."
)


# ============================================================================
# SECTION 3 — CAUSAL GRAPH CONSTRUCTION
# ============================================================================

print("\n[5/7] Constructing causal graph and identifying effect...")
print("-" * 80)


causal_graph_dot = """
digraph {

    rankdir=LR;

    market_segment -> treatment;
    market_segment -> outcome;

    category -> treatment;
    category -> outcome;

    base_price -> outcome;
    unit_cost -> outcome;
    elasticity -> outcome;

    rolling_7day_net_revenue -> outcome;
    rolling_7day_units_sold -> outcome;

    treatment -> outcome
        [color=red, penwidth=2];

}
"""


# ============================================================================
# PARSE GRAPH
# ============================================================================

pydot_graphs = (
    pydot.graph_from_dot_data(
        causal_graph_dot
    )
)

if not pydot_graphs:

    raise RuntimeError(
        "The causal graph DOT specification "
        "could not be parsed."
    )


causal_graph_nx = (
    nx.nx_pydot.from_pydot(
        pydot_graphs[0]
    )
)


print(
    "  ✓ Causal graph successfully parsed"
)


# ============================================================================
# DOWHY IDENTIFICATION
# ============================================================================

dowhy_columns = [
    "treatment",
    "outcome",
    "market_segment",
    "category",
] + numeric_adjustment_cols


dowhy_data = (
    df_causal[dowhy_columns]
    .copy()
)


causal_model = CausalModel(
    data=dowhy_data,
    treatment="treatment",
    outcome="outcome",
    graph=causal_graph_dot,
)


identified_estimand = (
    causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )
)


print(
    "  ✓ DoWhy causal effect identification completed"
)

print(
    "  ✓ DML adjustment matrix constructed explicitly"
)


# ============================================================================
# SECTION 4 — DOUBLE MACHINE LEARNING
# ============================================================================

print("\n[6/7] Estimating causal effect with EconML LinearDML...")
print("-" * 80)


# ============================================================================
# OUTCOME NUISANCE MODEL
# ============================================================================

model_y = LGBMRegressor(

    n_estimators=100,

    learning_rate=0.05,

    max_depth=4,

    num_leaves=15,

    min_child_samples=20,

    subsample=0.90,

    colsample_bytree=0.90,

    random_state=RANDOM_STATE,

    verbosity=-1,
)


# ============================================================================
# TREATMENT NUISANCE MODEL
# ============================================================================

model_t = LGBMClassifier(

    n_estimators=100,

    learning_rate=0.05,

    max_depth=4,

    num_leaves=15,

    min_child_samples=20,

    subsample=0.90,

    colsample_bytree=0.90,

    random_state=RANDOM_STATE,

    verbosity=-1,
)


# ============================================================================
# LINEARDML
# ============================================================================

dml_model = LinearDML(

    model_y=model_y,

    model_t=model_t,

    discrete_treatment=True,

    cv=3,

    random_state=RANDOM_STATE,
)


print(
    "  Fitting full validated sample..."
)

print(
    f"  X shape: {X_numeric.shape}"
)

print(
    f"  T shape: {T.shape}"
)

print(
    f"  Y shape: {Y.shape}"
)


# ============================================================================
# CRITICAL FIX
# ============================================================================
#
# The original code used:
#
#     inference='debiased'
#
# which is not supported by the EconML version installed in the user's
# Colab environment.
#
# The user's traceback explicitly reported:
#
#     valid values are
#     ['bootstrap', 'auto', 'statsmodels']
#
# Therefore this implementation uses:
#
#     inference='statsmodels'
#
# The invalid X_numeric[:100] fallback has also been completely removed.
#
# ============================================================================

dml_model.fit(

    Y,

    T,

    X=X_numeric,

    inference="statsmodels",

)


print(
    "  ✓ DML fit completed successfully"
)


# ============================================================================
# POST-FIT DIMENSION VALIDATION
# ============================================================================

if not (
    len(X_numeric)
    == len(Y)
    == len(T)
):

    raise RuntimeError(
        "Post-fit dimension validation failed."
    )


# ============================================================================
# ATE ESTIMATION
# ============================================================================

ate = float(
    np.asarray(
        dml_model.ate(
            X_numeric
        )
    ).squeeze()
)


# ============================================================================
# 95% CONFIDENCE INTERVAL
# ============================================================================

ate_lower, ate_upper = (
    dml_model.ate_interval(
        X=X_numeric,
        alpha=0.05,
    )
)


ate_lower = float(
    np.asarray(
        ate_lower
    ).squeeze()
)

ate_upper = float(
    np.asarray(
        ate_upper
    ).squeeze()
)


# ============================================================================
# ECONML INFERENCE OBJECT
# ============================================================================

ate_inference = (
    dml_model.ate_inference(
        X=X_numeric
    )
)


# ============================================================================
# STANDARD ERROR
# ============================================================================

try:

    se = float(
        np.asarray(
            ate_inference.stderr_mean
        ).squeeze()
    )

except Exception:

    # Fallback based on the confidence interval.
    #
    # 95% normal CI:
    #
    # estimate ± 1.959964 * SE
    #
    # Therefore:
    #
    # SE = (upper - lower) / (2 * 1.959964)

    se = float(
        (
            ate_upper
            - ate_lower
        )
        /
        (
            2
            * norm.ppf(0.975)
        )
    )


# ============================================================================
# P-VALUE
# ============================================================================

try:

    p_value = float(
        np.asarray(
            ate_inference.pvalue()
        ).squeeze()
    )

except Exception:

    if se > 0:

        p_value = float(
            2
            * norm.sf(
                abs(
                    ate / se
                )
            )
        )

    else:

        p_value = np.nan


# ============================================================================
# TEST STATISTIC
# ============================================================================

if se > 0:

    t_stat = float(
        ate / se
    )

else:

    t_stat = np.nan


# ============================================================================
# RELATIVE EFFECT
# ============================================================================

baseline_mean = float(
    df_causal["outcome"].mean()
)


if baseline_mean != 0:

    pct_effect = float(
        (
            ate
            /
            baseline_mean
        )
        * 100
    )

else:

    pct_effect = np.nan


# ============================================================================
# PRINT ATE RESULTS
# ============================================================================

print("\n  CAUSAL EFFECT ESTIMATES:")
print("-" * 80)

print(
    f"  Average Treatment Effect: "
    f"${ate:.2f}"
)

print(
    f"  95% Confidence Interval: "
    f"[${ate_lower:.2f}, ${ate_upper:.2f}]"
)

print(
    f"  Standard Error: "
    f"${se:.4f}"
)

print(
    f"  Test Statistic: "
    f"{t_stat:.4f}"
)

print(
    f"  p-value: "
    f"{p_value:.6g}"
)


# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n  INTERPRETATION:")
print("-" * 80)


if ate > 0:

    print(
        f"  ✓ PROMO_20 is estimated to increase "
        f"daily net revenue by ${abs(ate):.2f}."
    )

    print(
        "    Direction of estimated causal effect: POSITIVE"
    )

else:

    print(
        f"  ✗ PROMO_20 is estimated to decrease "
        f"daily net revenue by ${abs(ate):.2f}."
    )

    print(
        "    Direction of estimated causal effect: NEGATIVE"
    )


print(
    f"  Relative effect versus sample baseline: "
    f"{pct_effect:.2f}%"
)


# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================

print("\n  STATISTICAL SIGNIFICANCE:")
print("-" * 80)


if (
    np.isfinite(p_value)
    and p_value < 0.05
):

    print(
        "  ✓ Effect is statistically significant "
        "at α = 0.05"
    )

else:

    print(
        "  ⚠ Effect is not statistically significant "
        "at α = 0.05"
    )


# ============================================================================
# SECTION 5 — HETEROGENEOUS TREATMENT EFFECTS
# ============================================================================

print("\n[7/7] Estimating heterogeneous treatment effects...")
print("-" * 80)


# ============================================================================
# CATE
# ============================================================================

cate = np.asarray(
    dml_model.effect(
        X_numeric
    )
).reshape(-1)


# ============================================================================
# CATE DIMENSION CHECK
# ============================================================================

if len(cate) != len(df_causal):

    raise RuntimeError(
        "CATE dimension mismatch:\n"
        f"  CATE effects: {len(cate):,}\n"
        f"  Causal rows:  {len(df_causal):,}"
    )


# ============================================================================
# CATE DATAFRAME
# ============================================================================

df_cate = df_causal.copy()

df_cate["cate"] = cate


# ============================================================================
# HTE BY MARKET SEGMENT
# ============================================================================

hte_by_segment = (
    df_cate
    .groupby("market_segment")["cate"]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
            "count",
        ]
    )
    .round(2)
)


print(
    "\n  Conditional Average Treatment Effect "
    "by Market Segment:"
)

print("-" * 80)

print(
    hte_by_segment.to_string()
)


# ============================================================================
# HTE BY CATEGORY
# ============================================================================

hte_by_category = (
    df_cate
    .groupby("category")["cate"]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
            "count",
        ]
    )
    .round(2)
)


print(
    "\n  Conditional Average Treatment Effect "
    "by Product Category:"
)

print("-" * 80)

print(
    hte_by_category.to_string()
)


# ============================================================================
# ROBUSTNESS / OVERLAP CHECK
# ============================================================================

print("\n  Robustness and overlap diagnostics:")
print("-" * 80)


# ============================================================================
# TREATMENT PREVALENCE
# ============================================================================

treatment_rate = float(
    T.mean()
)


print(
    f"  Treatment prevalence: "
    f"{treatment_rate:.4f}"
)


# ============================================================================
# PROPENSITY MODEL
# ============================================================================

propensity_model = LGBMClassifier(

    n_estimators=100,

    learning_rate=0.05,

    max_depth=4,

    num_leaves=15,

    min_child_samples=20,

    random_state=RANDOM_STATE,

    verbosity=-1,

)


propensity_model.fit(
    X_numeric,
    T,
)


propensity = (
    propensity_model
    .predict_proba(X_numeric)[:, 1]
)


# ============================================================================
# PROPENSITY SUMMARY
# ============================================================================

propensity_min = float(
    np.min(propensity)
)

propensity_max = float(
    np.max(propensity)
)

propensity_q01 = float(
    np.quantile(
        propensity,
        0.01
    )
)

propensity_q99 = float(
    np.quantile(
        propensity,
        0.99
    )
)


# ---------------------------------------------------------------------------
# Practical overlap diagnostic.
#
# This is deliberately described as an empirical overlap check rather than
# a proof that the formal positivity assumption holds.
# ---------------------------------------------------------------------------

overlap_ok = bool(
    (
        propensity_min > 0.01
    )
    and
    (
        propensity_max < 0.99
    )
)


if overlap_ok:

    positivity_label = (
        "No severe empirical overlap violation detected"
    )

else:

    positivity_label = (
        "Potential limited overlap; inspect propensity distribution"
    )


print(
    f"  Propensity min/max: "
    f"{propensity_min:.4f} / {propensity_max:.4f}"
)

print(
    f"  Propensity 1%/99%: "
    f"{propensity_q01:.4f} / {propensity_q99:.4f}"
)

print(
    f"  Overlap assessment: "
    f"{positivity_label}"
)


# ============================================================================
# SENSITIVITY DIAGNOSTIC
# ============================================================================

"""
The original code calculated:

    sqrt(V_y / (n * V_t))

and labeled that a "Rotnitzky-Robbins bound."

That is not sufficient to claim a formal omitted-variable sensitivity
bound.

Rather than preserve a misleading statistical label, this implementation
reports a transparent heuristic diagnostic only.

It should NOT be interpreted as proof that the estimate is robust to
arbitrary unmeasured confounding.
"""

outcome_std = float(
    np.std(
        Y,
        ddof=1
    )
)


effective_n = max(
    len(Y),
    1
)


heuristic_bias_scale = float(
    outcome_std
    /
    np.sqrt(effective_n)
)


robust_to_heuristic_bias = bool(
    abs(ate)
    >
    heuristic_bias_scale
)


print(
    f"  Heuristic bias scale: "
    f"${heuristic_bias_scale:.4f}"
)

print(
    "  NOTE: this is a diagnostic, "
    "not a formal omitted-variable sensitivity bound."
)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n  Creating visualizations...")
print("-" * 80)


# ============================================================================
# FIGURE 1 — CAUSAL GRAPH
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)


pos = nx.spring_layout(
    causal_graph_nx,
    k=2,
    iterations=50,
    seed=RANDOM_STATE,
)


nx.draw_networkx_nodes(
    causal_graph_nx,
    pos,
    node_size=2500,
    ax=ax,
    alpha=0.9,
)


nx.draw_networkx_labels(
    causal_graph_nx,
    pos,
    font_size=9,
    font_weight="bold",
    ax=ax,
)


nx.draw_networkx_edges(
    causal_graph_nx,
    pos,
    arrows=True,
    arrowsize=20,
    ax=ax,
)


if causal_graph_nx.has_edge(
    "treatment",
    "outcome",
):

    nx.draw_networkx_edges(
        causal_graph_nx,
        pos,
        edgelist=[
            (
                "treatment",
                "outcome",
            )
        ],
        arrows=True,
        arrowsize=25,
        width=3,
        ax=ax,
    )


ax.set_title(
    "Causal Graph: PROMO_20 → Daily Net Revenue",
    fontsize=14,
    fontweight="bold",
)

ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "causal_graph_structure.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ causal_graph_structure.png"
)


# ============================================================================
# FIGURE 2 — NAIVE VS CAUSAL ATE
# ============================================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


y_positions = np.arange(2)


ax.scatter(
    [
        naive_ate,
        ate,
    ],
    y_positions,
    s=180,
    zorder=3,
)


ax.errorbar(
    ate,
    1,
    xerr=[
        [
            ate - ate_lower
        ],
        [
            ate_upper - ate
        ],
    ],
    fmt="none",
    capsize=6,
    linewidth=2,
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
)


ax.set_yticks(
    y_positions
)


ax.set_yticklabels(
    [
        "Naive ATE (Unadjusted)",
        "Causal ATE (DML)",
    ]
)


ax.set_xlabel(
    "Average Treatment Effect ($)"
)

ax.set_title(
    "Naive vs. Causal Treatment Effect",
    fontsize=14,
    fontweight="bold",
)


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "ate_comparison_causal_vs_naive.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ ate_comparison_causal_vs_naive.png"
)


# ============================================================================
# FIGURE 3 — HTE BY MARKET SEGMENT
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


segment_means = (
    hte_by_segment["mean"]
    .sort_values()
)


segment_stds = (
    hte_by_segment
    .loc[
        segment_means.index,
        "std"
    ]
    .fillna(0)
)


x_positions = np.arange(
    len(segment_means)
)


ax.bar(
    x_positions,
    segment_means.values,
    yerr=segment_stds.values,
    capsize=5,
)


ax.axhline(
    ate,
    linestyle="--",
    linewidth=2,
    label=f"Overall ATE: ${ate:.2f}",
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(
    segment_means.index
)


ax.set_xlabel(
    "Market Segment"
)

ax.set_ylabel(
    "Conditional Treatment Effect ($)"
)

ax.set_title(
    "Heterogeneous Treatment Effects by Market Segment",
    fontsize=14,
    fontweight="bold",
)


ax.legend()


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "hte_by_market_segment.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ hte_by_market_segment.png"
)


# ============================================================================
# FIGURE 4 — CATE DISTRIBUTION
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


ax.hist(
    cate,
    bins=50,
    edgecolor="black",
    alpha=0.75,
    density=True,
)


ax.axvline(
    ate,
    linestyle="--",
    linewidth=2,
    label=f"Mean ATE: ${ate:.2f}",
)


ax.axvline(
    np.median(cate),
    linestyle="--",
    linewidth=2,
    label=(
        f"Median CATE: "
        f"${np.median(cate):.2f}"
    ),
)


ax.axvline(
    0,
    linewidth=1,
)


ax.set_xlabel(
    "Conditional Treatment Effect ($)"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Distribution of Conditional Treatment Effects",
    fontsize=14,
    fontweight="bold",
)


ax.legend()


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "cate_distribution.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ cate_distribution.png"
)


# ============================================================================
# FIGURE 5 — OUTCOME DISTRIBUTION
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

axes[0].boxplot(
    [
        control_outcome.to_numpy(),
        treated_outcome.to_numpy(),
    ],
    labels=[
        "Control",
        "PROMO_20",
    ],
)


axes[0].set_ylabel(
    "Daily Net Revenue ($)"
)

axes[0].set_title(
    "Treatment vs Control Outcome Distribution"
)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

axes[1].hist(
    control_outcome,
    bins=40,
    alpha=0.6,
    label="Control",
    density=True,
    edgecolor="black",
)


axes[1].hist(
    treated_outcome,
    bins=40,
    alpha=0.6,
    label="PROMO_20",
    density=True,
    edgecolor="black",
)


axes[1].set_xlabel(
    "Daily Net Revenue ($)"
)

axes[1].set_ylabel(
    "Density"
)

axes[1].set_title(
    "Outcome Distribution Comparison"
)

axes[1].legend()


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "outcome_distribution_comparison.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ outcome_distribution_comparison.png"
)


# ============================================================================
# FIGURE 6 — STANDARDIZED MEAN DIFFERENCE
# ============================================================================

smd_rows = []


for column in numeric_adjustment_cols:

    treated_values = (
        df_causal
        .loc[
            T == 1,
            column
        ]
        .to_numpy(
            dtype=float
        )
    )

    control_values = (
        df_causal
        .loc[
            T == 0,
            column
        ]
        .to_numpy(
            dtype=float
        )
    )


    treated_variance = np.var(
        treated_values,
        ddof=1,
    )

    control_variance = np.var(
        control_values,
        ddof=1,
    )


    pooled_std = np.sqrt(
        (
            treated_variance
            +
            control_variance
        )
        /
        2
    )


    if pooled_std > 0:

        smd = (
            np.mean(treated_values)
            -
            np.mean(control_values)
        ) / pooled_std

    else:

        smd = 0.0


    smd_rows.append(
        (
            column,
            abs(float(smd)),
        )
    )


smd_df = pd.DataFrame(
    smd_rows,
    columns=[
        "variable",
        "absolute_smd",
    ],
).sort_values(
    "absolute_smd"
)


fig, ax = plt.subplots(
    figsize=(12, 7)
)


ax.barh(
    smd_df["variable"],
    smd_df["absolute_smd"],
)


ax.axvline(
    0.1,
    linestyle="--",
    linewidth=2,
    label="SMD = 0.10",
)


ax.axvline(
    0.2,
    linestyle="--",
    linewidth=2,
    label="SMD = 0.20",
)


ax.set_xlabel(
    "Absolute Standardized Mean Difference"
)

ax.set_title(
    "Pre-Adjustment Covariate Balance",
    fontsize=14,
    fontweight="bold",
)


ax.legend()


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "confounder_balance_smd.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ confounder_balance_smd.png"
)


# ============================================================================
# FIGURE 7 — PROPENSITY OVERLAP
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


ax.hist(
    propensity[T == 0],
    bins=40,
    alpha=0.6,
    label="Control",
    density=True,
)


ax.hist(
    propensity[T == 1],
    bins=40,
    alpha=0.6,
    label="PROMO_20",
    density=True,
)


ax.set_xlabel(
    "Estimated Propensity P(T=1 | X)"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Treatment Propensity / Overlap Check",
    fontsize=14,
    fontweight="bold",
)


ax.legend()


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "propensity_overlap.png",
    dpi=120,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "  ✓ propensity_overlap.png"
)


# ============================================================================
# EXPORT DATA
# ============================================================================

print("\n  Saving output artifacts...")
print("-" * 80)


# ============================================================================
# CATE CSV
# ============================================================================

df_cate_export = df_cate[
    [
        "transaction_date",
        "store_id",
        "treatment",
        "outcome",
        "cate",
        "market_segment",
        "category",
    ]
].copy()


df_cate_export.to_csv(
    OUTPUT_DIR
    /
    "heterogeneous_treatment_effects.csv",
    index=False,
)


print(
    "  ✓ heterogeneous_treatment_effects.csv"
)


# ============================================================================
# SMD CSV
# ============================================================================

smd_df.to_csv(
    OUTPUT_DIR
    /
    "confounder_balance_smd.csv",
    index=False,
)


print(
    "  ✓ confounder_balance_smd.csv"
)


# ============================================================================
# JSON RESULTS
# ============================================================================

causal_results = {

    "model_type":
        "Double Machine Learning with "
        "LightGBM nuisance models",

    "estimation_method":
        "EconML LinearDML",

    "inference_method":
        "statsmodels",

    "data_source":
        data_source,

    "source_table":
        (
            f"{PROJECT_ID}."
            f"{DATASET_NAME}."
            f"{TABLE_NAME}"
        ),

    "rows_loaded":
        int(len(df_analytics)),

    "rows_analyzed":
        int(len(df_causal)),

    "rows_dropped_for_required_fields":
        int(rows_dropped),

    "sample_sizes": {

        "control":
            control_count,

        "treatment":
            treated_count,

        "total":
            int(len(T)),
    },

    "treatment_prevalence":
        treatment_rate,

    "adjustment_variables":
        (
            numeric_adjustment_cols
            +
            categorical_adjustment_cols
        ),

    "model_feature_count":
        int(X_numeric.shape[1]),

    "average_treatment_effect": {

        "point_estimate":
            ate,

        "ci_lower":
            ate_lower,

        "ci_upper":
            ate_upper,

        "standard_error":
            se,

        "test_statistic":
            t_stat,

        "p_value":
            p_value,

        "is_significant_05":
            bool(
                np.isfinite(p_value)
                and
                p_value < 0.05
            ),
    },

    "naive_ate": {

        "point_estimate":
            naive_ate,

        "description":
            (
                "Unadjusted difference in means; "
                "not a causal estimate."
            ),
    },

    "relative_effect_percent":
        pct_effect,

    "heterogeneous_effects": {

        "by_market_segment":
            {
                str(key): float(value)
                for key, value
                in (
                    df_cate
                    .groupby(
                        "market_segment"
                    )["cate"]
                    .mean()
                    .items()
                )
            },

        "by_category":
            {
                str(key): float(value)
                for key, value
                in (
                    df_cate
                    .groupby(
                        "category"
                    )["cate"]
                    .mean()
                    .items()
                )
            },
    },

    "overlap_check": {

        "propensity_min":
            propensity_min,

        "propensity_max":
            propensity_max,

        "propensity_q01":
            propensity_q01,

        "propensity_q99":
            propensity_q99,

        "assessment":
            positivity_label,
    },

    "sensitivity_analysis": {

        "heuristic_bias_scale":
            heuristic_bias_scale,

        "robust_to_heuristic_bias_scale":
            robust_to_heuristic_bias,

        "interpretation":
            (
                "Diagnostic only. This is NOT a formal "
                "omitted-variable sensitivity bound."
            ),
    },

    "timestamp":
        datetime.now().isoformat(),
}


with open(
    OUTPUT_DIR
    /
    "causal_inference_results.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        causal_results,
        f,
        indent=2,
        allow_nan=False,
    )


print(
    "  ✓ causal_inference_results.json"
)


# ============================================================================
# FINAL EXECUTIVE SUMMARY
# ============================================================================

print("\n" + "=" * 80)

print(
    "✓ PHASE 3A COMPLETE: "
    "CAUSAL INFERENCE MODELING"
)

print("=" * 80)

print(
    f"""
CAUSAL INFERENCE EXECUTIVE SUMMARY
----------------------------------

Treatment:
    PROMO_20 vs CONTROL

Outcome:
    Daily Net Revenue

Sample:
    {len(df_causal):,} observations

Treatment:
    {treated_count:,}

Control:
    {control_count:,}

DML Causal ATE:
    ${ate:.2f}

95% Confidence Interval:
    [${ate_lower:.2f}, ${ate_upper:.2f}]

Standard Error:
    ${se:.4f}

Test Statistic:
    {t_stat:.4f}

p-value:
    {p_value:.6g}

Statistically Significant:
    {"YES" if np.isfinite(p_value) and p_value < 0.05 else "NO"}

Relative Effect:
    {pct_effect:.2f}%

Naive ATE:
    ${naive_ate:.2f}

Difference:
    ${ate - naive_ate:.2f}

Empirical Overlap:
    {positivity_label}

Output Directory:
    {OUTPUT_DIR}
"""
)


# ============================================================================
# LIST OUTPUT ARTIFACTS
# ============================================================================

print(
    "Generated artifacts:"
)

for output_file in sorted(
    OUTPUT_DIR.iterdir()
):

    if output_file.is_file():

        size_kb = (
            output_file.stat().st_size
            /
            1024
        )

        print(
            f"  ✓ {output_file.name:<45s}"
            f"{size_kb:>10.1f} KB"
        )


print("\n" + "=" * 80)

print(
    "✓ PHASE 3A DELIVERABLES READY FOR PHASE 3B & 3C"
)

print("=" * 80)
