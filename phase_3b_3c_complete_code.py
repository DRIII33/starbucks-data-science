"""
================================================================================
PHASE 3B & 3C: COMPLETE BAYESIAN ELASTICITY & PROFIT OPTIMIZATION PIPELINE
================================================================================

PROJECT:
    Starbucks Data Science Portfolio Project

PHASES:
    3B - Bayesian Price Elasticity Modeling
    3C - Nonlinear Constrained Profit Optimization

ENVIRONMENT:
    Google Colab

BIGQUERY PROJECT:
    driiiportfolio-506303

BIGQUERY DATASET:
    starbucks_transactions

BIGQUERY TABLE:
    analytics_ready_promo_data

================================================================================
IMPORTANT METHODOLOGICAL NOTES
================================================================================

PHASE 3B
--------
The elasticity model estimates:

    ln(Q) = alpha + elasticity * ln(P) + error

where:

    Q = daily units sold
    P = actual selling price

Because the model uses the ORIGINAL log-price and log-quantity scales,
the posterior coefficient "elasticity" is directly interpretable as:

    % change in quantity / % change in price

The model does NOT use standardized log-price/log-quantity variables
for the elasticity coefficient.

PHASE 3C
--------
The optimization model is NONLINEAR, not linear programming.

Demand is modeled as:

    Q(P) = Q0 * (P / P0)^elasticity

Profit is:

    Profit(P) = Q(P) * (P - UnitCost)

Decision variable:

    discount percentage

Constraints:

    0% <= discount <= 30%
    expected units >= minimum threshold
    discounted price > unit cost

The optimization is solved with SciPy SLSQP.

================================================================================
DATA POLICY
================================================================================

This notebook intentionally DOES NOT silently fall back to fabricated data
when BigQuery fails.

If production data cannot be loaded or validated, execution stops with an
actionable error.

This prevents a successful-looking portfolio analysis from accidentally
being generated from synthetic fallback data.

================================================================================
OUTPUT ARTIFACTS
================================================================================

PHASE 3B:

    elasticity_trace.nc
    elasticity_model_summary.csv
    elasticity_results.json
    category_elasticity_results.csv
    bayesian_elasticity_trace.png
    elasticity_posterior_distribution.png
    bayesian_elasticity_ppc.png
    demand_curves_by_category.png

PHASE 3C:

    optimization_results.json
    optimization_strategy_detailed.csv
    optimal_discount_strategy.png
    profit_improvement_comparison.png
    sensitivity_analysis_profit_discount.png
    optimization_diagnostics.json

PIPELINE:

    execution_manifest.json

================================================================================
DATE:
    August 2026
================================================================================
"""


# ============================================================================
# 0. INITIALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("INITIALIZING PHASE 3B & 3C")
print("BAYESIAN ELASTICITY + NONLINEAR PROFIT OPTIMIZATION")
print("=" * 80)

print("\nProject:")
print("  BigQuery Project : driiiportfolio-506303")
print("  Dataset          : starbucks_transactions")
print("  Table            : analytics_ready_promo_data")
print("  Environment      : Google Colab")
print("=" * 80)


# ============================================================================
# 1. INSTALL / VERIFY DEPENDENCIES
# ============================================================================

print("\n[1/8] Installing / verifying required libraries...")
print("-" * 80)

import os
import sys
import subprocess
import importlib
import warnings

warnings.filterwarnings("ignore")

# Reduce TensorFlow informational logging if TensorFlow is indirectly loaded.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensure_package(import_name, pip_name=None):
    """
    Verify that a Python package is available.

    Parameters
    ----------
    import_name : str
        Python import name.
    pip_name : str, optional
        Package name used by pip.
    """
    pip_name = pip_name or import_name

    try:
        importlib.import_module(import_name)
        print(f"  ✓ {pip_name:<25} available")
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

        # Verify after installation.
        importlib.import_module(import_name)
        print(f"  ✓ {pip_name:<25} installed successfully")


required_packages = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pymc", "pymc"),
    ("arviz", "arviz"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("google.cloud.bigquery", "google-cloud-bigquery"),
    ("db_dtypes", "db-dtypes"),
]

for import_name, pip_name in required_packages:
    ensure_package(import_name, pip_name)


# ============================================================================
# 2. IMPORTS
# ============================================================================

print("\n[2/8] Importing libraries...")
print("-" * 80)

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az

from scipy.optimize import minimize, Bounds

import matplotlib.pyplot as plt

from google.cloud import bigquery


print(f"  ✓ NumPy       : {np.__version__}")
print(f"  ✓ Pandas      : {pd.__version__}")
print(f"  ✓ PyMC        : {pm.__version__}")
print(f"  ✓ ArviZ       : {az.__version__}")
print(f"  ✓ SciPy       : {__import__('scipy').__version__}")
print(f"  ✓ All imports completed successfully")


# ============================================================================
# 3. CONFIGURATION
# ============================================================================

print("\n[3/8] Configuring pipeline...")
print("-" * 80)

# --------------------------------------------------------------------------
# BigQuery
# --------------------------------------------------------------------------

PROJECT_ID = "driiiportfolio-506303"
DATASET_NAME = "starbucks_transactions"
TABLE_NAME = "analytics_ready_promo_data"

FULL_TABLE_NAME = (
    f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"
)


# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------

RANDOM_SEED = 42

# Global Bayesian model.
GLOBAL_DRAWS = 2000
GLOBAL_TUNE = 1000

# Category models.
CATEGORY_DRAWS = 1000
CATEGORY_TUNE = 500

TARGET_ACCEPT = 0.90

# Minimum category-level observations.
MIN_CATEGORY_OBSERVATIONS = 5

# Minimum number of unique price points required for an elasticity estimate.
MIN_CATEGORY_PRICE_POINTS = 2

# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------

MIN_DISCOUNT = 0.00
MAX_DISCOUNT = 0.30

# Minimum expected daily units per category.
MIN_EXPECTED_UNITS = 50.0

# --------------------------------------------------------------------------
# BigQuery loading
# --------------------------------------------------------------------------

MAX_ROWS = 100_000

# --------------------------------------------------------------------------
# Output directory
# --------------------------------------------------------------------------

OUTPUT_DIR = os.getcwd()

print(f"  BigQuery table:")
print(f"    {FULL_TABLE_NAME}")

print(f"\n  Modeling:")
print(f"    Global draws       : {GLOBAL_DRAWS:,}")
print(f"    Global tune        : {GLOBAL_TUNE:,}")
print(f"    Category draws     : {CATEGORY_DRAWS:,}")
print(f"    Category tune      : {CATEGORY_TUNE:,}")
print(f"    Target acceptance  : {TARGET_ACCEPT}")

print(f"\n  Optimization:")
print(f"    Discount range     : {MIN_DISCOUNT:.0%} - {MAX_DISCOUNT:.0%}")
print(f"    Minimum units      : {MIN_EXPECTED_UNITS:.0f}")

print(f"\n  Output directory:")
print(f"    {OUTPUT_DIR}")


# ============================================================================
# 4. BIGQUERY DATA LOADING
# ============================================================================

print("\n[4/8] Loading production data from BigQuery...")
print("-" * 80)

# Explicitly define the fields required by this phase.
REQUIRED_COLUMNS = [
    "transaction_date",
    "store_id",
    "category",
    "treatment_group",
    "promo_id",
    "discount_pct",
    "base_price",
    "unit_cost",
    "daily_units_sold",
    "daily_net_revenue",
    "daily_profit",
]


try:

    client = bigquery.Client(project=PROJECT_ID)

    # First inspect the table schema.
    table_ref = f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"

    print(f"  → Inspecting table schema...")
    table_metadata = client.get_table(table_ref)

    available_columns = [field.name for field in table_metadata.schema]

    print(f"  ✓ Table found")
    print(f"  ✓ Available columns: {len(available_columns)}")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from BigQuery table: "
            + ", ".join(missing_columns)
        )

    print("  ✓ Required schema validation passed")


    # ----------------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------------

    query = f"""
    SELECT
        transaction_date,
        store_id,
        category,
        treatment_group,
        promo_id,
        discount_pct,
        base_price,
        unit_cost,
        daily_units_sold,
        daily_net_revenue,
        daily_profit
    FROM `{FULL_TABLE_NAME}`
    ORDER BY transaction_date, store_id, category
    LIMIT {MAX_ROWS}
    """

    print(f"\n  → Executing query...")
    print(f"    LIMIT {MAX_ROWS:,}")

    df_analytics = client.query(query).to_dataframe()

    print(f"\n  ✓ Loaded {len(df_analytics):,} rows")
    print(f"  ✓ Loaded {len(df_analytics.columns)} columns")


except Exception as e:

    raise RuntimeError(
        "\n\n"
        "BIGQUERY DATA LOAD FAILED.\n"
        "The pipeline intentionally stops instead of generating synthetic "
        "fallback data.\n\n"
        f"Project : {PROJECT_ID}\n"
        f"Dataset : {DATASET_NAME}\n"
        f"Table   : {TABLE_NAME}\n\n"
        f"Original error:\n{repr(e)}\n\n"
        "Verify that:\n"
        "1. The Colab runtime is authenticated.\n"
        "2. Project ID is correct.\n"
        "3. The dataset exists.\n"
        "4. The table exists.\n"
        "5. The required columns exist.\n"
        "6. The account has permission to query the table."
    )


# ============================================================================
# 5. DATA VALIDATION
# ============================================================================

print("\n[5/8] Validating production dataset...")
print("-" * 80)


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )


def report_missing_values(df, columns):
    results = {}

    for column in columns:
        results[column] = int(df[column].isna().sum())

    return results


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

require_columns(df_analytics, REQUIRED_COLUMNS)

print("  ✓ Required columns present")


# --------------------------------------------------------------------------
# Datatypes
# --------------------------------------------------------------------------

df_analytics["transaction_date"] = pd.to_datetime(
    df_analytics["transaction_date"],
    errors="coerce"
)

numeric_columns = [
    "discount_pct",
    "base_price",
    "unit_cost",
    "daily_units_sold",
    "daily_net_revenue",
    "daily_profit",
]

for column in numeric_columns:
    df_analytics[column] = pd.to_numeric(
        df_analytics[column],
        errors="coerce"
    )

print("  ✓ Numeric fields converted")


# --------------------------------------------------------------------------
# Missingness
# --------------------------------------------------------------------------

missing_report = report_missing_values(
    df_analytics,
    REQUIRED_COLUMNS
)

print("\n  Missing-value summary:")

for column, count in missing_report.items():

    if count > 0:
        print(f"    ⚠ {column:<25} {count:,}")
    else:
        print(f"    ✓ {column:<25} 0")


# --------------------------------------------------------------------------
# Remove invalid rows required for elasticity/optimization.
# --------------------------------------------------------------------------

model_required = [
    "category",
    "discount_pct",
    "base_price",
    "unit_cost",
    "daily_units_sold",
]

before_rows = len(df_analytics)

df_analytics = df_analytics.dropna(
    subset=model_required
).copy()

after_rows = len(df_analytics)

print(
    f"\n  Removed {before_rows - after_rows:,} rows "
    f"with missing modeling fields"
)


# --------------------------------------------------------------------------
# Numeric validity
# --------------------------------------------------------------------------

invalid_price = (
    ~np.isfinite(df_analytics["base_price"])
    | (df_analytics["base_price"] <= 0)
)

invalid_cost = (
    ~np.isfinite(df_analytics["unit_cost"])
    | (df_analytics["unit_cost"] < 0)
)

invalid_units = (
    ~np.isfinite(df_analytics["daily_units_sold"])
    | (df_analytics["daily_units_sold"] <= 0)
)

invalid_discount = (
    ~np.isfinite(df_analytics["discount_pct"])
    | (df_analytics["discount_pct"] < 0)
    | (df_analytics["discount_pct"] >= 1)
)

invalid_rows = (
    invalid_price
    | invalid_cost
    | invalid_units
    | invalid_discount
)

invalid_count = int(invalid_rows.sum())

if invalid_count > 0:

    print(
        f"  ⚠ Removing {invalid_count:,} rows with invalid "
        "economic values"
    )

    df_analytics = df_analytics.loc[~invalid_rows].copy()


# --------------------------------------------------------------------------
# Category validation
# --------------------------------------------------------------------------

df_analytics["category"] = (
    df_analytics["category"]
    .astype(str)
    .str.strip()
)

df_analytics = df_analytics[
    df_analytics["category"].ne("")
    & df_analytics["category"].ne("nan")
].copy()


if df_analytics.empty:

    raise ValueError(
        "No valid rows remain after data validation."
    )


# --------------------------------------------------------------------------
# Actual price
# --------------------------------------------------------------------------

df_analytics["price_point"] = (
    df_analytics["base_price"]
    * (1.0 - df_analytics["discount_pct"])
)

invalid_actual_price = (
    ~np.isfinite(df_analytics["price_point"])
    | (df_analytics["price_point"] <= 0)
)

if invalid_actual_price.any():

    count = int(invalid_actual_price.sum())

    print(
        f"  ⚠ Removing {count:,} rows with invalid "
        "calculated selling price"
    )

    df_analytics = df_analytics.loc[
        ~invalid_actual_price
    ].copy()


# --------------------------------------------------------------------------
# Final dataset checks
# --------------------------------------------------------------------------

print("\n  Final validated dataset:")
print(f"    Rows       : {len(df_analytics):,}")
print(f"    Categories : {df_analytics['category'].nunique():,}")
print(
    f"    Date range : "
    f"{df_analytics['transaction_date'].min()} → "
    f"{df_analytics['transaction_date'].max()}"
)

print("\n  Price statistics:")
print(
    df_analytics[
        [
            "base_price",
            "price_point",
            "daily_units_sold",
            "discount_pct"
        ]
    ].describe().round(3)
)


# ============================================================================
# PHASE 3B
# BAYESIAN PRICE ELASTICITY MODEL
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3B: BAYESIAN PRICE ELASTICITY MODELING")
print("=" * 80)


# ============================================================================
# 3B-1. BUILD ELASTICITY DATASET
# ============================================================================

print("\n[3B-1] Preparing elasticity dataset...")
print("-" * 80)

"""
IMPORTANT:

The original implementation aggregated by:

    category + price_point

and then standardized both log-price and log-units.

That makes the coefficient difficult to interpret as conventional
price elasticity.

Here we preserve the observation-level relationship and model:

    log(units) ~ log(price)

directly.

This maintains the economic interpretation of the coefficient.
"""


df_elasticity = df_analytics[
    [
        "transaction_date",
        "store_id",
        "category",
        "price_point",
        "daily_units_sold",
        "base_price",
        "discount_pct",
        "unit_cost",
        "daily_net_revenue",
        "daily_profit",
    ]
].copy()


# Log transformations.
df_elasticity["log_price"] = np.log(
    df_elasticity["price_point"]
)

df_elasticity["log_units_sold"] = np.log(
    df_elasticity["daily_units_sold"]
)


# Remove any unexpected numerical failures.
finite_mask = (
    np.isfinite(df_elasticity["log_price"])
    & np.isfinite(df_elasticity["log_units_sold"])
)

df_elasticity = df_elasticity.loc[
    finite_mask
].copy()


if len(df_elasticity) < 20:

    raise ValueError(
        "Insufficient observations for Bayesian elasticity modeling. "
        f"Only {len(df_elasticity)} valid observations remain."
    )


print(
    f"  ✓ Valid elasticity observations: "
    f"{len(df_elasticity):,}"
)

print(
    f"  ✓ Categories: "
    f"{df_elasticity['category'].nunique():,}"
)

print(
    f"  ✓ Price range: "
    f"${df_elasticity['price_point'].min():.2f} - "
    f"${df_elasticity['price_point'].max():.2f}"
)

print(
    f"  ✓ Unit range: "
    f"{df_elasticity['daily_units_sold'].min():.0f} - "
    f"{df_elasticity['daily_units_sold'].max():.0f}"
)


# ============================================================================
# 3B-2. PRICE VARIATION CHECK
# ============================================================================

print("\n[3B-2] Checking price variation...")
print("-" * 80)

global_unique_prices = (
    df_elasticity["price_point"]
    .nunique()
)

if global_unique_prices < 2:

    raise ValueError(
        "The dataset does not contain sufficient price variation "
        "to estimate price elasticity."
    )

print(
    f"  ✓ Global unique price points: "
    f"{global_unique_prices}"
)


category_variation = (
    df_elasticity
    .groupby("category")
    .agg(
        n_obs=("category", "size"),
        unique_prices=("price_point", "nunique"),
        min_price=("price_point", "min"),
        max_price=("price_point", "max"),
    )
    .reset_index()
)

print("\n  Category price variation:")
print(
    category_variation.to_string(index=False)
)


# ============================================================================
# 3B-3. GLOBAL BAYESIAN MODEL
# ============================================================================

print("\n[3B-3] Building pooled Bayesian elasticity model...")
print("-" * 80)

X_global = (
    df_elasticity["log_price"]
    .to_numpy(dtype=float)
)

Y_global = (
    df_elasticity["log_units_sold"]
    .to_numpy(dtype=float)
)


print(
    f"  X shape: {X_global.shape}"
)

print(
    f"  Y shape: {Y_global.shape}"
)

print(
    "\n  Model:"
)

print(
    "    log(units) = intercept + elasticity × log(price) + error"
)

print(
    "\n  Interpretation:"
)

print(
    "    elasticity = % change in quantity / % change in price"
)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

with pm.Model() as elasticity_model:

    # Intercept on the log-units scale.
    intercept = pm.Normal(
        "intercept",
        mu=float(np.mean(Y_global)),
        sigma=5.0
    )

    # Economically informed prior.
    #
    # A Normal prior centered at -1.5 permits both moderately elastic
    # and relatively inelastic demand while reflecting the expectation
    # that price elasticity is generally negative.
    elasticity = pm.Normal(
        "elasticity",
        mu=-1.5,
        sigma=1.0
    )

    # Positive residual standard deviation.
    sigma = pm.HalfNormal(
        "sigma",
        sigma=1.0
    )

    mu = (
        intercept
        + elasticity * X_global
    )

    y_obs = pm.Normal(
        "y_obs",
        mu=mu,
        sigma=sigma,
        observed=Y_global
    )

    # Explicit NUTS configuration.
    nuts = pm.NUTS(
        target_accept=TARGET_ACCEPT,
        max_treedepth=10
    )

    print(
        "\n  → Sampling global posterior..."
    )

    trace = pm.sample(
        draws=GLOBAL_DRAWS,
        tune=GLOBAL_TUNE,
        chains=2,
        cores=2,
        step=nuts,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        progressbar=True
    )


print("\n  ✓ Global Bayesian sampling completed")


# ============================================================================
# 3B-4. MODEL DIAGNOSTICS
# ============================================================================

print("\n[3B-4] Evaluating Bayesian convergence...")
print("-" * 80)

summary = az.summary(
    trace,
    var_names=[
        "intercept",
        "elasticity",
        "sigma"
    ],
    round_to=6
)

print("\n" + summary.to_string())


# --------------------------------------------------------------------------
# Divergences
# --------------------------------------------------------------------------

if "diverging" in trace.sample_stats:

    n_divergences = int(
        trace.sample_stats["diverging"]
        .sum()
        .item()
    )

else:

    n_divergences = 0


# --------------------------------------------------------------------------
# R-hat
# --------------------------------------------------------------------------

if "r_hat" in summary.columns:

    rhat_max = float(
        summary["r_hat"].max()
    )

else:

    rhat_max = np.nan


# --------------------------------------------------------------------------
# ESS
# --------------------------------------------------------------------------

if "ess_bulk" in summary.columns:

    ess_bulk_min = float(
        summary["ess_bulk"].min()
    )

    ess_bulk_mean = float(
        summary["ess_bulk"].mean()
    )

else:

    ess_bulk_min = np.nan
    ess_bulk_mean = np.nan


print("\n  Diagnostics:")

print(
    f"    Divergences : {n_divergences}"
)

print(
    f"    Max R-hat   : {rhat_max:.4f}"
)

print(
    f"    Min ESS     : {ess_bulk_min:.0f}"
)

print(
    f"    Mean ESS    : {ess_bulk_mean:.0f}"
)


# Explicit convergence assessment.
convergence_ok = (
    n_divergences == 0
    and np.isfinite(rhat_max)
    and rhat_max < 1.05
    and np.isfinite(ess_bulk_min)
    and ess_bulk_min >= 400
)


if convergence_ok:

    print(
        "\n  ✓ Bayesian convergence checks passed"
    )

else:

    print(
        "\n  ⚠ Bayesian convergence checks require review"
    )


# ============================================================================
# 3B-5. GLOBAL ELASTICITY EXTRACTION
# ============================================================================

print("\n[3B-5] Extracting posterior elasticity...")
print("-" * 80)

elasticity_samples = (
    trace.posterior["elasticity"]
    .values
    .flatten()
)

intercept_samples = (
    trace.posterior["intercept"]
    .values
    .flatten()
)

sigma_samples = (
    trace.posterior["sigma"]
    .values
    .flatten()
)


elasticity_mean = float(
    np.mean(elasticity_samples)
)

elasticity_std = float(
    np.std(elasticity_samples, ddof=1)
)


elasticity_hdi = az.hdi(
    trace,
    var_names=["elasticity"],
    hdi_prob=0.95
)

elasticity_hdi_lower = float(
    elasticity_hdi["elasticity"].values[0]
)

elasticity_hdi_upper = float(
    elasticity_hdi["elasticity"].values[1]
)


print("\n  GLOBAL PRICE ELASTICITY")

print(
    f"    Posterior mean : {elasticity_mean:.6f}"
)

print(
    f"    Posterior SD   : {elasticity_std:.6f}"
)

print(
    f"    95% HDI        : "
    f"[{elasticity_hdi_lower:.6f}, "
    f"{elasticity_hdi_upper:.6f}]"
)


if elasticity_mean < 0:

    print(
        "\n    ✓ Estimated elasticity is negative, "
        "consistent with conventional demand behavior."
    )

else:

    print(
        "\n    ⚠ Estimated elasticity is non-negative. "
        "This requires economic investigation."
    )


if abs(elasticity_mean) > 1:

    elasticity_classification = "ELASTIC"

else:

    elasticity_classification = "INELASTIC"


print(
    f"    Demand classification: "
    f"{elasticity_classification}"
)

print(
    f"\n    Interpretation:"
)

print(
    f"    A 1% increase in price is associated with "
    f"approximately a {abs(elasticity_mean):.2f}% "
    f"change in quantity in the opposite direction."
)


# ============================================================================
# 3B-6. CATEGORY-SPECIFIC BAYESIAN ELASTICITY
# ============================================================================

print("\n[3B-6] Estimating category-specific elasticity...")
print("-" * 80)

category_results = []

category_traces = {}


categories = sorted(
    df_elasticity["category"]
    .dropna()
    .unique()
)


for category in categories:

    df_cat = df_elasticity[
        df_elasticity["category"] == category
    ].copy()

    n_obs = len(df_cat)

    n_prices = (
        df_cat["price_point"]
        .nunique()
    )

    print(
        f"\n  Category: {category}"
    )

    print(
        f"    Observations : {n_obs:,}"
    )

    print(
        f"    Price points : {n_prices}"
    )


    # ----------------------------------------------------------------------
    # Sufficiency check
    # ----------------------------------------------------------------------

    if (
        n_obs < MIN_CATEGORY_OBSERVATIONS
        or n_prices < MIN_CATEGORY_PRICE_POINTS
    ):

        print(
            "    ⚠ Insufficient variation/observations "
            "for category-specific Bayesian estimate."
        )

        category_results.append(
            {
                "category": category,
                "n_observations": n_obs,
                "unique_price_points": n_prices,
                "elasticity_mean": np.nan,
                "elasticity_sd": np.nan,
                "hdi_95_lower": np.nan,
                "hdi_95_upper": np.nan,
                "divergences": np.nan,
                "rhat_max": np.nan,
                "ess_bulk_min": np.nan,
                "status": "INSUFFICIENT_DATA",
            }
        )

        continue


    X_cat = (
        df_cat["log_price"]
        .to_numpy(dtype=float)
    )

    Y_cat = (
        df_cat["log_units_sold"]
        .to_numpy(dtype=float)
    )


    # ----------------------------------------------------------------------
    # Category model
    # ----------------------------------------------------------------------

    with pm.Model() as cat_model:

        intercept_cat = pm.Normal(
            "intercept",
            mu=float(np.mean(Y_cat)),
            sigma=5.0
        )

        elasticity_cat = pm.Normal(
            "elasticity",
            mu=-1.5,
            sigma=1.0
        )

        sigma_cat = pm.HalfNormal(
            "sigma",
            sigma=1.0
        )

        mu_cat = (
            intercept_cat
            + elasticity_cat * X_cat
        )

        y_obs_cat = pm.Normal(
            "y_obs",
            mu=mu_cat,
            sigma=sigma_cat,
            observed=Y_cat
        )

        nuts_cat = pm.NUTS(
            target_accept=TARGET_ACCEPT,
            max_treedepth=10
        )

        try:

            trace_cat = pm.sample(
                draws=CATEGORY_DRAWS,
                tune=CATEGORY_TUNE,
                chains=2,
                cores=1,
                step=nuts_cat,
                random_seed=RANDOM_SEED,
                return_inferencedata=True,
                progressbar=False
            )

        except Exception as category_error:

            print(
                f"    ⚠ Category model failed: "
                f"{repr(category_error)}"
            )

            category_results.append(
                {
                    "category": category,
                    "n_observations": n_obs,
                    "unique_price_points": n_prices,
                    "elasticity_mean": np.nan,
                    "elasticity_sd": np.nan,
                    "hdi_95_lower": np.nan,
                    "hdi_95_upper": np.nan,
                    "divergences": np.nan,
                    "rhat_max": np.nan,
                    "ess_bulk_min": np.nan,
                    "status": "MODEL_FAILED",
                }
            )

            continue


    category_traces[category] = trace_cat


    # ----------------------------------------------------------------------
    # Category diagnostics
    # ----------------------------------------------------------------------

    cat_summary = az.summary(
        trace_cat,
        var_names=[
            "intercept",
            "elasticity",
            "sigma"
        ],
        round_to=6
    )


    cat_elasticity_samples = (
        trace_cat.posterior["elasticity"]
        .values
        .flatten()
    )

    cat_elasticity_mean = float(
        np.mean(cat_elasticity_samples)
    )

    cat_elasticity_sd = float(
        np.std(
            cat_elasticity_samples,
            ddof=1
        )
    )


    cat_hdi = az.hdi(
        trace_cat,
        var_names=["elasticity"],
        hdi_prob=0.95
    )

    cat_hdi_lower = float(
        cat_hdi["elasticity"].values[0]
    )

    cat_hdi_upper = float(
        cat_hdi["elasticity"].values[1]
    )


    if "diverging" in trace_cat.sample_stats:

        cat_divergences = int(
            trace_cat.sample_stats["diverging"]
            .sum()
            .item()
        )

    else:

        cat_divergences = 0


    cat_rhat_max = float(
        cat_summary["r_hat"].max()
    )

    cat_ess_bulk_min = float(
        cat_summary["ess_bulk"].min()
    )


    cat_converged = (
        cat_divergences == 0
        and cat_rhat_max < 1.05
        and cat_ess_bulk_min >= 400
    )


    status = (
        "CONVERGED"
        if cat_converged
        else "DIAGNOSTIC_REVIEW_REQUIRED"
    )


    category_results.append(
        {
            "category": category,
            "n_observations": n_obs,
            "unique_price_points": n_prices,
            "elasticity_mean": cat_elasticity_mean,
            "elasticity_sd": cat_elasticity_sd,
            "hdi_95_lower": cat_hdi_lower,
            "hdi_95_upper": cat_hdi_upper,
            "divergences": cat_divergences,
            "rhat_max": cat_rhat_max,
            "ess_bulk_min": cat_ess_bulk_min,
            "status": status,
        }
    )


    print(
        f"    Elasticity: "
        f"{cat_elasticity_mean:.4f}"
    )

    print(
        f"    95% HDI: "
        f"[{cat_hdi_lower:.4f}, "
        f"{cat_hdi_upper:.4f}]"
    )

    print(
        f"    Divergences: "
        f"{cat_divergences}"
    )

    print(
        f"    R-hat max: "
        f"{cat_rhat_max:.4f}"
    )

    print(
        f"    ESS min: "
        f"{cat_ess_bulk_min:.0f}"
    )

    print(
        f"    Status: {status}"
    )


category_results_df = pd.DataFrame(
    category_results
)


print("\n  Category elasticity results:")
print(
    category_results_df.to_string(
        index=False
    )
)


# ============================================================================
# 3B-7. SELECT ELASTICITIES FOR OPTIMIZATION
# ============================================================================

print("\n[3B-7] Preparing validated elasticities for optimization...")
print("-" * 80)


"""
Optimization requires a finite elasticity estimate.

Preference order:

    1. Category-specific Bayesian elasticity
    2. Global pooled Bayesian elasticity

However, a category-specific estimate is used only if its diagnostics
are acceptable.

If a category-specific model is unavailable or fails diagnostics,
the global pooled estimate is used explicitly and labeled as such.
"""


category_elasticity_map = {}
elasticity_source_map = {}


for _, row in category_results_df.iterrows():

    category = row["category"]

    category_estimate = row["elasticity_mean"]

    category_status = row["status"]


    if (
        np.isfinite(category_estimate)
        and category_status == "CONVERGED"
    ):

        category_elasticity_map[category] = float(
            category_estimate
        )

        elasticity_source_map[category] = (
            "category_specific_bayesian"
        )

    else:

        if not np.isfinite(elasticity_mean):

            raise ValueError(
                f"No valid elasticity estimate is available "
                f"for category '{category}'."
            )

        category_elasticity_map[category] = float(
            elasticity_mean
        )

        elasticity_source_map[category] = (
            "global_pooled_bayesian_fallback"
        )


print("\n  Elasticities entering optimization:")

for category in sorted(category_elasticity_map):

    print(
        f"    {category:<30} "
        f"{category_elasticity_map[category]:>9.4f} "
        f"({elasticity_source_map[category]})"
    )


# ============================================================================
# 3B-8. VISUALIZATION: TRACE PLOTS
# ============================================================================

print("\n[3B-8] Creating Phase 3B visualizations...")
print("-" * 80)


# --------------------------------------------------------------------------
# Trace plot
# --------------------------------------------------------------------------

az.plot_trace(
    trace,
    var_names=[
        "intercept",
        "elasticity",
        "sigma"
    ],
    figsize=(14, 8)
)

plt.suptitle(
    "Bayesian Price Elasticity Model — Trace & Posterior Diagnostics",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

trace_plot_path = os.path.join(
    OUTPUT_DIR,
    "bayesian_elasticity_trace.png"
)

plt.savefig(
    trace_plot_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print(
    f"  ✓ Saved: {trace_plot_path}"
)


# ============================================================================
# 3B-9. POSTERIOR DISTRIBUTION
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.hist(
    elasticity_samples,
    bins=60,
    density=True,
    alpha=0.70,
    edgecolor="black",
    label="Posterior samples"
)

ax.axvline(
    elasticity_mean,
    linestyle="--",
    linewidth=2.5,
    label=f"Posterior mean = {elasticity_mean:.4f}"
)

ax.axvline(
    elasticity_hdi_lower,
    linestyle="--",
    linewidth=2,
    alpha=0.7
)

ax.axvline(
    elasticity_hdi_upper,
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=(
        f"95% HDI = "
        f"[{elasticity_hdi_lower:.4f}, "
        f"{elasticity_hdi_upper:.4f}]"
    )
)

ax.set_xlabel(
    "Price Elasticity of Demand",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Posterior Density",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Posterior Distribution of Price Elasticity",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

posterior_plot_path = os.path.join(
    OUTPUT_DIR,
    "elasticity_posterior_distribution.png"
)

plt.savefig(
    posterior_plot_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print(
    f"  ✓ Saved: {posterior_plot_path}"
)


# ============================================================================
# 3B-10. POSTERIOR PREDICTIVE CHECK
# ============================================================================

print("\n  Generating posterior predictive distribution...")

with elasticity_model:

    posterior_predictive = pm.sample_posterior_predictive(
        trace,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        progressbar=False
    )


trace.extend(
    posterior_predictive
)


fig, ax = plt.subplots(
    figsize=(12, 6)
)

try:

    az.plot_ppc(
        trace,
        num_pp_samples=100,
        ax=ax
    )

except Exception as ppc_error:

    print(
        f"  ⚠ Standard PPC plot unavailable: "
        f"{repr(ppc_error)}"
    )

    # Fallback diagnostic: posterior mean fitted values vs observations.
    posterior_intercept = (
        trace.posterior["intercept"]
        .values
        .mean()
    )

    posterior_elasticity = (
        trace.posterior["elasticity"]
        .values
        .mean()
    )

    predicted = (
        posterior_intercept
        + posterior_elasticity * X_global
    )

    ax.scatter(
        Y_global,
        predicted,
        alpha=0.15,
        s=10
    )

    min_val = min(
        Y_global.min(),
        predicted.min()
    )

    max_val = max(
        Y_global.max(),
        predicted.max()
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    ax.set_xlabel(
        "Observed log(units)"
    )

    ax.set_ylabel(
        "Predicted log(units)"
    )

    ax.set_title(
        "Posterior Mean Predicted vs Observed"
    )


plt.title(
    "Posterior Predictive Check — Bayesian Elasticity Model",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

ppc_plot_path = os.path.join(
    OUTPUT_DIR,
    "bayesian_elasticity_ppc.png"
)

plt.savefig(
    ppc_plot_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print(
    f"  ✓ Saved: {ppc_plot_path}"
)


# ============================================================================
# 3B-11. DEMAND CURVES BY CATEGORY
# ============================================================================

print("\n  Generating category demand curves...")


valid_categories = sorted(
    df_elasticity["category"]
    .unique()
)

n_categories = len(valid_categories)

# Dynamic grid instead of assuming exactly 3 categories.
n_cols = min(3, max(1, n_categories))
n_rows = math.ceil(
    n_categories / n_cols
)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(6 * n_cols, 5 * n_rows),
    squeeze=False
)

axes_flat = axes.flatten()


for idx, category in enumerate(valid_categories):

    ax = axes_flat[idx]

    df_cat = df_elasticity[
        df_elasticity["category"] == category
    ].copy()

    ax.scatter(
        df_cat["price_point"],
        df_cat["daily_units_sold"],
        s=35,
        alpha=0.40,
        edgecolors="black",
        linewidth=0.5
    )


    # Category elasticity.
    elasticity_for_plot = (
        category_elasticity_map.get(
            category,
            elasticity_mean
        )
    )


    # Baseline point for curve.
    baseline_price = float(
        df_cat["price_point"].median()
    )

    baseline_units = float(
        df_cat["daily_units_sold"].median()
    )


    price_min = float(
        df_cat["price_point"].min()
    )

    price_max = float(
        df_cat["price_point"].max()
    )


    if price_min < price_max:

        price_range = np.linspace(
            price_min,
            price_max,
            100
        )

        predicted_units = (
            baseline_units
            * (
                price_range
                / baseline_price
            )
            ** elasticity_for_plot
        )

        ax.plot(
            price_range,
            predicted_units,
            linestyle="--",
            linewidth=2,
            label=(
                f"Elasticity ≈ "
                f"{elasticity_for_plot:.3f}"
            )
        )


    ax.set_xlabel(
        "Selling Price ($)",
        fontweight="bold"
    )

    ax.set_ylabel(
        "Daily Units Sold",
        fontweight="bold"
    )

    ax.set_title(
        f"{category}\nDemand Relationship",
        fontweight="bold"
    )

    ax.grid(alpha=0.25)
    ax.legend()


# Remove unused axes.
for idx in range(
    n_categories,
    len(axes_flat)
):

    axes_flat[idx].remove()


plt.tight_layout()

demand_curve_path = os.path.join(
    OUTPUT_DIR,
    "demand_curves_by_category.png"
)

plt.savefig(
    demand_curve_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print(
    f"  ✓ Saved: {demand_curve_path}"
)


# ============================================================================
# 3B-12. SAVE BAYESIAN ARTIFACTS
# ============================================================================

print("\n[3B-12] Saving Phase 3B artifacts...")
print("-" * 80)


# Trace.
trace_path = os.path.join(
    OUTPUT_DIR,
    "elasticity_trace.nc"
)

trace.to_netcdf(
    trace_path
)

print(
    f"  ✓ Saved: {trace_path}"
)


# Summary.
summary_path = os.path.join(
    OUTPUT_DIR,
    "elasticity_model_summary.csv"
)

summary.reset_index().to_csv(
    summary_path,
    index=False
)

print(
    f"  ✓ Saved: {summary_path}"
)


# Category elasticity results.
category_results_path = os.path.join(
    OUTPUT_DIR,
    "category_elasticity_results.csv"
)

category_results_df.to_csv(
    category_results_path,
    index=False
)

print(
    f"  ✓ Saved: {category_results_path}"
)


# JSON summary.
bayesian_results = {

    "project_id": PROJECT_ID,

    "dataset": DATASET_NAME,

    "table": TABLE_NAME,

    "model_type": (
        "Pooled Bayesian Log-Log Price Elasticity Model"
    ),

    "model_equation": (
        "log(daily_units_sold) = "
        "intercept + elasticity * log(price_point) + error"
    ),

    "global_elasticity": {

        "mean": elasticity_mean,

        "std": elasticity_std,

        "hdi_95_lower": elasticity_hdi_lower,

        "hdi_95_upper": elasticity_hdi_upper,

        "classification": elasticity_classification,
    },

    "diagnostics": {

        "divergences": n_divergences,

        "rhat_max": rhat_max,

        "ess_bulk_min": ess_bulk_min,

        "ess_bulk_mean": ess_bulk_mean,

        "convergence_passed": bool(convergence_ok),
    },

    "category_elasticities": {},

    "data": {

        "observations": int(
            len(df_elasticity)
        ),

        "categories": int(
            df_elasticity["category"].nunique()
        ),

        "unique_price_points": int(
            df_elasticity["price_point"].nunique()
        ),
    },

    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}


for _, row in category_results_df.iterrows():

    category = row["category"]

    bayesian_results[
        "category_elasticities"
    ][category] = {

        "elasticity_mean": (
            None
            if pd.isna(row["elasticity_mean"])
            else float(row["elasticity_mean"])
        ),

        "elasticity_sd": (
            None
            if pd.isna(row["elasticity_sd"])
            else float(row["elasticity_sd"])
        ),

        "hdi_95_lower": (
            None
            if pd.isna(row["hdi_95_lower"])
            else float(row["hdi_95_lower"])
        ),

        "hdi_95_upper": (
            None
            if pd.isna(row["hdi_95_upper"])
            else float(row["hdi_95_upper"])
        ),

        "divergences": (
            None
            if pd.isna(row["divergences"])
            else int(row["divergences"])
        ),

        "rhat_max": (
            None
            if pd.isna(row["rhat_max"])
            else float(row["rhat_max"])
        ),

        "ess_bulk_min": (
            None
            if pd.isna(row["ess_bulk_min"])
            else float(row["ess_bulk_min"])
        ),

        "status": row["status"],
    }


bayesian_json_path = os.path.join(
    OUTPUT_DIR,
    "elasticity_results.json"
)

with open(
    bayesian_json_path,
    "w"
) as f:

    json.dump(
        bayesian_results,
        f,
        indent=2
    )


print(
    f"  ✓ Saved: {bayesian_json_path}"
)


print("\n" + "=" * 80)
print("✓ PHASE 3B COMPLETE")
print("=" * 80)


# ============================================================================
# PHASE 3C
# NONLINEAR CONSTRAINED PROFIT OPTIMIZATION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3C: NONLINEAR CONSTRAINED PROFIT OPTIMIZATION")
print("=" * 80)


# ============================================================================
# 3C-1. CATEGORY BASELINE DATA
# ============================================================================

print("\n[3C-1] Building category-level optimization dataset...")
print("-" * 80)


"""
Baseline values are calculated from the validated production data.

We use:

    median base price
    median unit cost
    mean daily units sold

The mean daily units is retained as the demand baseline because the
elasticity function is intended to transform the observed demand level
at the baseline price.
"""


df_category_stats = (
    df_analytics
    .groupby("category")
    .agg(
        base_price=("base_price", "median"),
        unit_cost=("unit_cost", "median"),
        baseline_units=("daily_units_sold", "mean"),
        baseline_revenue=("daily_net_revenue", "mean"),
        baseline_profit=("daily_profit", "mean"),
        observations=("category", "size"),
    )
    .reset_index()
)


# Add validated elasticity.
df_category_stats["elasticity"] = (
    df_category_stats["category"]
    .map(category_elasticity_map)
)

df_category_stats["elasticity_source"] = (
    df_category_stats["category"]
    .map(elasticity_source_map)
)


# --------------------------------------------------------------------------
# Validate optimization inputs.
# --------------------------------------------------------------------------

optimization_numeric_columns = [
    "base_price",
    "unit_cost",
    "baseline_units",
    "elasticity",
]


for column in optimization_numeric_columns:

    if not np.isfinite(
        df_category_stats[column]
    ).all():

        invalid_categories = (
            df_category_stats.loc[
                ~np.isfinite(
                    df_category_stats[column]
                ),
                "category"
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"Invalid values found in optimization column "
            f"'{column}' for categories: "
            f"{invalid_categories}"
        )


# Unit economics must allow a positive margin at some price.
if (
    df_category_stats["base_price"]
    <= df_category_stats["unit_cost"]
).any():

    problem_categories = (
        df_category_stats.loc[
            df_category_stats["base_price"]
            <= df_category_stats["unit_cost"],
            "category"
        ]
        .astype(str)
        .tolist()
    )

    print(
        "\n  ⚠ Warning:"
    )

    print(
        "    The following categories have base price "
        "less than or equal to unit cost:"
    )

    for category in problem_categories:
        print(
            f"      - {category}"
        )

    print(
        "\n    These categories may not have a feasible "
        "positive-margin optimization solution."
    )


print(
    f"  ✓ Optimization categories: "
    f"{len(df_category_stats)}"
)

print(
    "\n  Category baseline metrics:"
)

print(
    df_category_stats.to_string(
        index=False
    )
)


# ============================================================================
# 3C-2. NUMPY ARRAYS
# ============================================================================

product_names = (
    df_category_stats["category"]
    .astype(str)
    .to_numpy()
)

base_prices = (
    df_category_stats["base_price"]
    .to_numpy(dtype=float)
)

unit_costs = (
    df_category_stats["unit_cost"]
    .to_numpy(dtype=float)
)

baseline_units = (
    df_category_stats["baseline_units"]
    .to_numpy(dtype=float)
)

elasticities = (
    df_category_stats["elasticity"]
    .to_numpy(dtype=float)
)


n_products = len(
    product_names
)


# ============================================================================
# 3C-3. ECONOMIC DEMAND FUNCTION
# ============================================================================

print("\n[3C-2] Defining nonlinear demand and profit functions...")
print("-" * 80)


def calculate_expected_units(
    discounts,
    base_prices,
    baseline_units,
    elasticities
):
    """
    Calculate expected demand under discount.

    Demand model:

        Q(P) = Q0 * (P / P0)^elasticity

    Since:

        P = P0 * (1 - discount)

    we have:

        Q(discount)
            = Q0 * (1 - discount)^elasticity

    Parameters
    ----------
    discounts : array-like
        Discount fractions.

    base_prices : array-like
        Baseline prices.

    baseline_units : array-like
        Baseline daily demand.

    elasticities : array-like
        Price elasticities.

    Returns
    -------
    numpy.ndarray
        Expected daily units.
    """

    discounts = np.asarray(
        discounts,
        dtype=float
    )

    discounted_prices = (
        base_prices
        * (1.0 - discounts)
    )

    price_ratio = (
        discounted_prices
        / base_prices
    )

    expected_units = (
        baseline_units
        * np.power(
            price_ratio,
            elasticities
        )
    )

    return expected_units


def calculate_profit_by_category(
    discounts,
    base_prices,
    unit_costs,
    baseline_units,
    elasticities
):
    """
    Calculate daily profit for each category.
    """

    discounts = np.asarray(
        discounts,
        dtype=float
    )

    discounted_prices = (
        base_prices
        * (1.0 - discounts)
    )

    expected_units = calculate_expected_units(
        discounts,
        base_prices,
        baseline_units,
        elasticities
    )

    contribution_margin = (
        discounted_prices
        - unit_costs
    )

    profit = (
        expected_units
        * contribution_margin
    )

    return profit


def calculate_total_profit(
    discounts,
    base_prices,
    unit_costs,
    baseline_units,
    elasticities
):
    """
    Calculate total expected daily profit.
    """

    profit_by_category = (
        calculate_profit_by_category(
            discounts,
            base_prices,
            unit_costs,
            baseline_units,
            elasticities
        )
    )

    return float(
        np.sum(profit_by_category)
    )


# ============================================================================
# 3C-4. CONSTRAINT FUNCTIONS
# ============================================================================

def minimum_units_constraint(
    discounts
):
    """
    Constraint:

        expected_units >= MIN_EXPECTED_UNITS
    """

    expected_units = calculate_expected_units(
        discounts,
        base_prices,
        baseline_units,
        elasticities
    )

    return (
        expected_units
        - MIN_EXPECTED_UNITS
    )


def positive_margin_constraint(
    discounts
):
    """
    Constraint:

        discounted price - unit cost >= 0
    """

    discounted_prices = (
        base_prices
        * (1.0 - discounts)
    )

    return (
        discounted_prices
        - unit_costs
    )


# ============================================================================
# 3C-5. BASELINE PROFIT
# ============================================================================

print("\n[3C-3] Calculating baseline economics...")
print("-" * 80)


zero_discounts = np.zeros(
    n_products
)


baseline_expected_units = (
    calculate_expected_units(
        zero_discounts,
        base_prices,
        baseline_units,
        elasticities
    )
)


baseline_profit_by_category = (
    calculate_profit_by_category(
        zero_discounts,
        base_prices,
        unit_costs,
        baseline_units,
        elasticities
    )
)


baseline_profit = float(
    baseline_profit_by_category.sum()
)


print(
    f"  Baseline expected daily profit: "
    f"${baseline_profit:,.2f}"
)


# ============================================================================
# 3C-6. INITIAL FEASIBILITY CHECK
# ============================================================================

print("\n[3C-4] Checking optimization feasibility...")
print("-" * 80)


maximum_discount = np.full(
    n_products,
    MAX_DISCOUNT
)


units_at_max_discount = (
    calculate_expected_units(
        maximum_discount,
        base_prices,
        baseline_units,
        elasticities
    )
)


margin_at_max_discount = (
    base_prices
    * (1.0 - MAX_DISCOUNT)
    - unit_costs
)


for i, product in enumerate(product_names):

    print(
        f"  {product:<30} "
        f"units@30%={units_at_max_discount[i]:8.2f} | "
        f"margin@30%=${margin_at_max_discount[i]:7.2f}"
    )


# Check whether minimum units can be satisfied at the allowed maximum
# discount.
if (
    units_at_max_discount
    < MIN_EXPECTED_UNITS
).any():

    impossible_categories = (
        product_names[
            units_at_max_discount
            < MIN_EXPECTED_UNITS
        ]
        .tolist()
    )

    raise ValueError(
        "Optimization is infeasible under the minimum-units constraint "
        f"for categories: {impossible_categories}. "
        f"Even the maximum allowed discount of "
        f"{MAX_DISCOUNT:.0%} does not produce "
        f"{MIN_EXPECTED_UNITS:.0f} expected units/day."
    )


# Check whether maximum discount maintains positive margin.
if (
    margin_at_max_discount
    <= 0
).any():

    print(
        "\n  ⚠ Maximum discount creates zero/negative margin "
        "for some categories."
    )

    print(
        "    SLSQP will enforce the positive-margin constraint."
    )


# ============================================================================
# 3C-7. OPTIMIZATION OBJECTIVE
# ============================================================================

print("\n[3C-5] Configuring nonlinear optimization...")
print("-" * 80)


def negative_total_profit(
    discounts
):
    """
    SciPy minimizes functions.

    Therefore:

        objective = -profit
    """

    total_profit = calculate_total_profit(
        discounts,
        base_prices,
        unit_costs,
        baseline_units,
        elasticities
    )

    if not np.isfinite(total_profit):

        return 1e30

    return -total_profit


# Initial guess: no discount.
x0 = np.zeros(
    n_products,
    dtype=float
)


bounds = Bounds(
    lb=np.full(
        n_products,
        MIN_DISCOUNT
    ),
    ub=np.full(
        n_products,
        MAX_DISCOUNT
    )
)


constraints = [

    {
        "type": "ineq",
        "fun": minimum_units_constraint,
    },

    {
        "type": "ineq",
        "fun": positive_margin_constraint,
    },

]


print(
    "  Objective:"
)

print(
    "    Maximize expected daily profit"
)

print(
    "  Decision variables:"
)

print(
    f"    {n_products} category-specific discount percentages"
)

print(
    "  Method:"
)

print(
    "    SciPy SLSQP nonlinear constrained optimization"
)

print(
    "\n  Constraints:"
)

print(
    f"    {MIN_DISCOUNT:.0%} <= discount <= "
    f"{MAX_DISCOUNT:.0%}"
)

print(
    f"    Expected units >= "
    f"{MIN_EXPECTED_UNITS:.0f}"
)

print(
    "    Discounted price >/= unit cost"
)


# ============================================================================
# 3C-8. RUN OPTIMIZATION
# ============================================================================

print("\n[3C-6] Solving nonlinear optimization problem...")
print("-" * 80)


result = minimize(
    negative_total_profit,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "ftol": 1e-9,
        "maxiter": 2000,
        "disp": False,
    }
)


print(
    f"\n  Solver success : {result.success}"
)

print(
    f"  Status code    : {result.status}"
)

print(
    f"  Iterations     : {result.nit}"
)

print(
    f"  Message        : {result.message}"
)


# ============================================================================
# 3C-9. INDEPENDENT OPTIMIZATION VALIDATION
# ============================================================================

print("\n[3C-7] Independently validating optimization solution...")
print("-" * 80)


if not result.success:

    raise RuntimeError(
        "SLSQP optimization did not converge successfully.\n"
        f"Status: {result.status}\n"
        f"Message: {result.message}"
    )


optimal_discounts = np.asarray(
    result.x,
    dtype=float
)


if not np.all(
    np.isfinite(
        optimal_discounts
    )
):

    raise RuntimeError(
        "Optimization returned non-finite discount values."
    )


# Independent calculations.
optimal_prices = (
    base_prices
    * (1.0 - optimal_discounts)
)


optimal_units = (
    calculate_expected_units(
        optimal_discounts,
        base_prices,
        baseline_units,
        elasticities
    )
)


optimal_profit_by_category = (
    calculate_profit_by_category(
        optimal_discounts,
        base_prices,
        unit_costs,
        baseline_units,
        elasticities
    )
)


optimal_profit = float(
    np.sum(
        optimal_profit_by_category
    )
)


# Independent constraint checks.
discount_bounds_ok = bool(
    np.all(
        optimal_discounts
        >= MIN_DISCOUNT - 1e-8
    )
    and
    np.all(
        optimal_discounts
        <= MAX_DISCOUNT + 1e-8
    )
)


units_constraint_ok = bool(
    np.all(
        optimal_units
        >= MIN_EXPECTED_UNITS - 1e-6
    )
)


margin_constraint_ok = bool(
    np.all(
        optimal_prices
        >= unit_costs - 1e-8
    )
)


profit_finite = bool(
    np.isfinite(
        optimal_profit
    )
)


optimization_validated = bool(
    result.success
    and discount_bounds_ok
    and units_constraint_ok
    and margin_constraint_ok
    and profit_finite
)


print(
    f"  Discount bounds valid : {discount_bounds_ok}"
)

print(
    f"  Unit constraint valid : {units_constraint_ok}"
)

print(
    f"  Margin constraint valid : "
    f"{margin_constraint_ok}"
)

print(
    f"  Profit finite         : {profit_finite}"
)

print(
    f"  Final validation      : "
    f"{optimization_validated}"
)


if not optimization_validated:

    raise RuntimeError(
        "Optimization solver returned a result, but independent "
        "validation failed."
    )


# ============================================================================
# 3C-10. OPTIMIZATION RESULTS
# ============================================================================

print("\n[3C-8] Building optimization results table...")
print("-" * 80)


optimization_rows = []


for i, product in enumerate(product_names):

    discount = float(
        optimal_discounts[i]
    )

    base_price = float(
        base_prices[i]
    )

    unit_cost = float(
        unit_costs[i]
    )

    baseline_unit = float(
        baseline_units[i]
    )

    elasticity = float(
        elasticities[i]
    )

    new_price = float(
        optimal_prices[i]
    )

    expected_unit = float(
        optimal_units[i]
    )

    revenue = (
        expected_unit
        * new_price
    )

    cogs = (
        expected_unit
        * unit_cost
    )

    profit = (
        revenue
        - cogs
    )

    baseline_product_profit = (
        baseline_unit
        * (
            base_price
            - unit_cost
        )
    )

    profit_change = (
        profit
        - baseline_product_profit
    )

    optimization_rows.append(
        {
            "Product": product,

            "Elasticity": elasticity,

            "Elasticity_Source":
                elasticity_source_map[product],

            "Base_Price":
                base_price,

            "Unit_Cost":
                unit_cost,

            "Baseline_Units":
                baseline_unit,

            "Optimal_Discount_Pct":
                discount * 100.0,

            "Optimal_Price":
                new_price,

            "Expected_Units":
                expected_unit,

            "Expected_Revenue":
                revenue,

            "Expected_COGS":
                cogs,

            "Expected_Daily_Profit":
                profit,

            "Baseline_Daily_Profit":
                baseline_product_profit,

            "Daily_Profit_Change":
                profit_change,
        }
    )


optimization_results_df = pd.DataFrame(
    optimization_rows
)


print(
    optimization_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}"
    )
)


# ============================================================================
# 3C-11. PROFIT IMPACT
# ============================================================================

profit_increase = (
    optimal_profit
    - baseline_profit
)


if baseline_profit != 0:

    improvement_pct = (
        profit_increase
        / abs(baseline_profit)
        * 100.0
    )

else:

    improvement_pct = np.nan


annualized_profit_change = (
    profit_increase
    * 365.0
)


print("\n  PROFIT IMPACT")
print("-" * 80)

print(
    f"  Baseline daily profit : "
    f"${baseline_profit:,.2f}"
)

print(
    f"  Optimized daily profit: "
    f"${optimal_profit:,.2f}"
)

print(
    f"  Daily improvement     : "
    f"${profit_increase:,.2f}"
)

print(
    f"  Improvement %         : "
    f"{improvement_pct:.2f}%"
)

print(
    f"  Annualized difference : "
    f"${annualized_profit_change:,.2f}"
)


# ============================================================================
# 3C-12. SENSITIVITY ANALYSIS
# ============================================================================

print("\n[3C-9] Running discount sensitivity analysis...")
print("-" * 80)


discount_range = np.arange(
    0.00,
    MAX_DISCOUNT + 0.0001,
    0.05
)


sensitivity_rows = []


for uniform_discount in discount_range:

    uniform_discounts = np.full(
        n_products,
        uniform_discount
    )


    test_units = calculate_expected_units(
        uniform_discounts,
        base_prices,
        baseline_units,
        elasticities
    )


    test_profit = calculate_total_profit(
        uniform_discounts,
        base_prices,
        unit_costs,
        baseline_units,
        elasticities
    )


    min_units = float(
        np.min(test_units)
    )


    min_margin = float(
        np.min(
            base_prices
            * (1.0 - uniform_discount)
            - unit_costs
        )
    )


    feasible = bool(
        min_units >= MIN_EXPECTED_UNITS
        and min_margin >= 0
    )


    sensitivity_rows.append(
        {
            "uniform_discount_pct":
                uniform_discount * 100.0,

            "daily_profit":
                test_profit,

            "minimum_expected_units":
                min_units,

            "minimum_margin":
                min_margin,

            "feasible":
                feasible,
        }
    )


sensitivity_df = pd.DataFrame(
    sensitivity_rows
)


print(
    sensitivity_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================================
# 3C-13. VISUALIZATION — OPTIMAL DISCOUNTS
# ============================================================================

print("\n[3C-10] Creating Phase 3C visualizations...")
print("-" * 80)


x_positions = np.arange(
    n_products
)


fig, ax = plt.subplots(
    figsize=(
        max(10, 1.8 * n_products),
        6
    )
)


bars = ax.bar(
    x_positions,
    optimal_discounts * 100.0,
    edgecolor="black",
    linewidth=1.0,
    alpha=0.80
)


ax.set_xlabel(
    "Product Category",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Optimal Discount (%)",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Optimal Discount Strategy for Expected Profit Maximization",
    fontsize=14,
    fontweight="bold"
)

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    product_names,
    rotation=35,
    ha="right"
)

ax.set_ylim(
    0,
    max(
        MAX_DISCOUNT * 100.0 * 1.15,
        np.max(optimal_discounts * 100.0) * 1.20 + 1
    )
)

ax.grid(
    axis="y",
    alpha=0.25
)


for bar, discount in zip(
    bars,
    optimal_discounts
):

    height = bar.get_height()

    ax.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        height + 0.5,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


plt.tight_layout()


optimal_discount_plot_path = os.path.join(
    OUTPUT_DIR,
    "optimal_discount_strategy.png"
)


plt.savefig(
    optimal_discount_plot_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()


print(
    f"  ✓ Saved: "
    f"{optimal_discount_plot_path}"
)


# ============================================================================
# 3C-14. VISUALIZATION — PROFIT COMPARISON
# ============================================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


scenario_labels = [
    "Baseline\nNo Discount",
    "Optimized\nStrategy"
]


scenario_profits = [
    baseline_profit,
    optimal_profit
]


bars = ax.bar(
    scenario_labels,
    scenario_profits,
    edgecolor="black",
    linewidth=1.0,
    alpha=0.80,
    width=0.55
)


ax.set_ylabel(
    "Expected Daily Profit ($)",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Expected Profit Impact: Baseline vs Optimized Strategy",
    fontsize=14,
    fontweight="bold"
)

ax.grid(
    axis="y",
    alpha=0.25
)


max_profit_for_plot = max(
    scenario_profits
)


for bar, profit in zip(
    bars,
    scenario_profits
):

    height = bar.get_height()

    ax.text(
        bar.get_x()
        + bar.get_width() / 2.0,
        height
        + max_profit_for_plot * 0.02,
        f"${profit:,.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )


if np.isfinite(
    improvement_pct
):

    ax.text(
        0.5,
        0.90,
        f"Expected improvement: "
        f"{improvement_pct:+.2f}%",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold"
    )


plt.tight_layout()


profit_comparison_path = os.path.join(
    OUTPUT_DIR,
    "profit_improvement_comparison.png"
)


plt.savefig(
    profit_comparison_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()


print(
    f"  ✓ Saved: "
    f"{profit_comparison_path}"
)


# ============================================================================
# 3C-15. VISUALIZATION — SENSITIVITY
# ============================================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


sensitivity_x = (
    sensitivity_df[
        "uniform_discount_pct"
    ]
)

sensitivity_y = (
    sensitivity_df[
        "daily_profit"
    ]
)

feasible_mask = (
    sensitivity_df[
        "feasible"
    ]
)


ax.plot(
    sensitivity_x[
        feasible_mask
    ],
    sensitivity_y[
        feasible_mask
    ],
    "o-",
    linewidth=2.5,
    markersize=7,
    label="Feasible"
)


if (
    ~feasible_mask
).any():

    ax.plot(
        sensitivity_x[
            ~feasible_mask
        ],
        sensitivity_y[
            ~feasible_mask
        ],
        "o--",
        linewidth=2,
        markersize=7,
        label="Infeasible"
    )


# Add average optimal discount only as a reference.
average_optimal_discount = float(
    np.mean(
        optimal_discounts
    )
)


ax.axvline(
    average_optimal_discount * 100.0,
    linestyle="--",
    linewidth=2,
    label=(
        f"Mean optimized discount: "
        f"{average_optimal_discount * 100:.1f}%"
    )
)


ax.set_xlabel(
    "Uniform Discount Level (%)",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Expected Daily Profit ($)",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Sensitivity Analysis: Expected Profit vs Uniform Discount",
    fontsize=14,
    fontweight="bold"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()


sensitivity_plot_path = os.path.join(
    OUTPUT_DIR,
    "sensitivity_analysis_profit_discount.png"
)


plt.savefig(
    sensitivity_plot_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close()


print(
    f"  ✓ Saved: "
    f"{sensitivity_plot_path}"
)


# ============================================================================
# 3C-16. SAVE OPTIMIZATION ARTIFACTS
# ============================================================================

print("\n[3C-11] Saving Phase 3C artifacts...")
print("-" * 80)


optimization_csv_path = os.path.join(
    OUTPUT_DIR,
    "optimization_strategy_detailed.csv"
)


optimization_results_df.to_csv(
    optimization_csv_path,
    index=False
)


print(
    f"  ✓ Saved: "
    f"{optimization_csv_path}"
)


# --------------------------------------------------------------------------
# Optimization JSON
# --------------------------------------------------------------------------

optimization_summary = {

    "project_id": PROJECT_ID,

    "dataset": DATASET_NAME,

    "table": TABLE_NAME,

    "model_type": (
        "Nonlinear Constrained Profit Optimization"
    ),

    "optimizer": "SciPy SLSQP",

    "objective": (
        "Maximize expected daily profit"
    ),

    "demand_model": (
        "Q(P) = Q0 * (P / P0)^elasticity"
    ),

    "profit_model": (
        "Profit(P) = Q(P) * (P - UnitCost)"
    ),

    "baseline_profit_daily":
        float(baseline_profit),

    "optimal_profit_daily":
        float(optimal_profit),

    "profit_increase_daily":
        float(profit_increase),

    "improvement_percent":
        (
            None
            if not np.isfinite(improvement_pct)
            else float(improvement_pct)
        ),

    "annualized_profit_difference":
        float(annualized_profit_change),

    "solver": {

        "success":
            bool(result.success),

        "status":
            int(result.status),

        "message":
            str(result.message),

        "iterations":
            int(result.nit),

        "function_evaluations":
            int(result.nfev),

        "gradient_evaluations":
            int(result.njev),
    },

    "validation": {

        "discount_bounds_valid":
            discount_bounds_ok,

        "minimum_units_valid":
            units_constraint_ok,

        "positive_margin_valid":
            margin_constraint_ok,

        "profit_finite":
            profit_finite,

        "optimization_validated":
            optimization_validated,
    },

    "constraints": {

        "minimum_discount":
            float(MIN_DISCOUNT),

        "maximum_discount":
            float(MAX_DISCOUNT),

        "minimum_expected_units":
            float(MIN_EXPECTED_UNITS),

        "positive_margin_required":
            True,
    },

    "optimal_strategy": {},

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}


for i, product in enumerate(
    product_names
):

    optimization_summary[
        "optimal_strategy"
    ][product] = {

        "elasticity":
            float(elasticities[i]),

        "elasticity_source":
            elasticity_source_map[product],

        "base_price":
            float(base_prices[i]),

        "unit_cost":
            float(unit_costs[i]),

        "baseline_units":
            float(baseline_units[i]),

        "discount_pct":
            float(
                optimal_discounts[i]
                * 100.0
            ),

        "new_price":
            float(
                optimal_prices[i]
            ),

        "expected_units":
            float(
                optimal_units[i]
            ),

        "expected_daily_profit":
            float(
                optimal_profit_by_category[i]
            ),
    }


optimization_json_path = os.path.join(
    OUTPUT_DIR,
    "optimization_results.json"
)


with open(
    optimization_json_path,
    "w"
) as f:

    json.dump(
        optimization_summary,
        f,
        indent=2
    )


print(
    f"  ✓ Saved: "
    f"{optimization_json_path}"
)


# ============================================================================
# 3C-17. OPTIMIZATION DIAGNOSTICS JSON
# ============================================================================

optimization_diagnostics = {

    "solver_success":
        bool(result.success),

    "solver_status":
        int(result.status),

    "solver_message":
        str(result.message),

    "iterations":
        int(result.nit),

    "function_evaluations":
        int(result.nfev),

    "gradient_evaluations":
        int(result.njev),

    "independent_validation":
        bool(optimization_validated),

    "minimum_expected_units":
        float(
            np.min(
                optimal_units
            )
        ),

    "minimum_margin":
        float(
            np.min(
                optimal_prices
                - unit_costs
            )
        ),

    "maximum_discount":
        float(
            np.max(
                optimal_discounts
            )
        ),

    "minimum_discount":
        float(
            np.min(
                optimal_discounts
            )
        ),

    "mean_discount":
        float(
            np.mean(
                optimal_discounts
            )
        ),
}


optimization_diagnostics_path = os.path.join(
    OUTPUT_DIR,
    "optimization_diagnostics.json"
)


with open(
    optimization_diagnostics_path,
    "w"
) as f:

    json.dump(
        optimization_diagnostics,
        f,
        indent=2
    )


print(
    f"  ✓ Saved: "
    f"{optimization_diagnostics_path}"
)


# ============================================================================
# 3C-18. EXECUTION MANIFEST
# ============================================================================

print("\n[3C-12] Creating execution manifest...")
print("-" * 80)


execution_manifest = {

    "pipeline":
        "PHASE 3B & 3C",

    "status":
        "SUCCESS",

    "project_id":
        PROJECT_ID,

    "dataset":
        DATASET_NAME,

    "table":
        TABLE_NAME,

    "rows_loaded":
        int(len(df_analytics)),

    "elasticity_observations":
        int(len(df_elasticity)),

    "categories":
        int(n_products),

    "global_elasticity":
        float(elasticity_mean),

    "global_elasticity_hdi_95":
        [
            float(elasticity_hdi_lower),
            float(elasticity_hdi_upper),
        ],

    "bayesian_diagnostics":
        {
            "divergences":
                n_divergences,

            "rhat_max":
                rhat_max,

            "ess_bulk_min":
                ess_bulk_min,

            "convergence_passed":
                convergence_ok,
        },

    "optimization":
        {
            "baseline_profit_daily":
                float(baseline_profit),

            "optimal_profit_daily":
                float(optimal_profit),

            "profit_increase_daily":
                float(profit_increase),

            "improvement_percent":
                (
                    None
                    if not np.isfinite(
                        improvement_pct
                    )
                    else float(
                        improvement_pct
                    )
                ),

            "validated":
                optimization_validated,
        },

    "artifacts": [

        "elasticity_trace.nc",

        "elasticity_model_summary.csv",

        "elasticity_results.json",

        "category_elasticity_results.csv",

        "bayesian_elasticity_trace.png",

        "elasticity_posterior_distribution.png",

        "bayesian_elasticity_ppc.png",

        "demand_curves_by_category.png",

        "optimization_results.json",

        "optimization_strategy_detailed.csv",

        "optimal_discount_strategy.png",

        "profit_improvement_comparison.png",

        "sensitivity_analysis_profit_discount.png",

        "optimization_diagnostics.json",
    ],

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
}


manifest_path = os.path.join(
    OUTPUT_DIR,
    "execution_manifest.json"
)


with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        execution_manifest,
        f,
        indent=2
    )


print(
    f"  ✓ Saved: {manifest_path}"
)


# ============================================================================
# 3C-19. EXECUTIVE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3C EXECUTIVE SUMMARY")
print("=" * 80)


print(
    "\n"
    "┌" + "─" * 76 + "┐"
)

print(
    "│"
    + " EXPECTED PROFIT OPTIMIZATION".center(76)
    + "│"
)

print(
    "├" + "─" * 76 + "┤"
)

print(
    f"│  Baseline daily profit:      "
    f"${baseline_profit:>15,.2f}"
    + " " * 24
    + "│"
)

print(
    f"│  Optimized daily profit:     "
    f"${optimal_profit:>15,.2f}"
    + " " * 24
    + "│"
)

print(
    f"│  Daily profit change:        "
    f"${profit_increase:>15,.2f}"
    + " " * 24
    + "│"
)

print(
    f"│  Percentage improvement:     "
    f"{improvement_pct:>14.2f}%"
    + " " * 25
    + "│"
)

print(
    f"│  Annualized difference:      "
    f"${annualized_profit_change:>15,.2f}"
    + " " * 24
    + "│"
)

print(
    "├" + "─" * 76 + "┤"
)

print(
    "│"
    + " BAYESIAN ELASTICITY".center(76)
    + "│"
)

print(
    "├" + "─" * 76 + "┤"
)

print(
    f"│  Global elasticity:          "
    f"{elasticity_mean:>10.4f}"
    + " " * 35
    + "│"
)

print(
    f"│  95% HDI:                    "
    f"[{elasticity_hdi_lower:.4f}, "
    f"{elasticity_hdi_upper:.4f}]"
    + " " * 20
    + "│"
)

print(
    f"│  Demand classification:      "
    f"{elasticity_classification:<15}"
    + " " * 27
    + "│"
)

print(
    "├" + "─" * 76 + "┤"
)

print(
    "│"
    + " MODEL DIAGNOSTICS".center(76)
    + "│"
)

print(
    "├" + "─" * 76 + "┤"
)

print(
    f"│  Bayesian divergences:       "
    f"{n_divergences:<8}"
    + " " * 39
    + "│"
)

print(
    f"│  Bayesian max R-hat:         "
    f"{rhat_max:.4f}"
    + " " * 37
    + "│"
)

print(
    f"│  Bayesian minimum ESS:       "
    f"{ess_bulk_min:.0f}"
    + " " * 39
    + "│"
)

print(
    f"│  Bayesian convergence:       "
    f"{'PASSED' if convergence_ok else 'REVIEW'}"
    + " " * 31
    + "│"
)

print(
    f"│  Optimization convergence:   "
    f"{'PASSED' if result.success else 'FAILED'}"
    + " " * 28
    + "│"
)

print(
    f"│  Independent validation:     "
    f"{'PASSED' if optimization_validated else 'FAILED'}"
    + " " * 27
    + "│"
)

print(
    "└" + "─" * 76 + "┘"
)


print("\n")
print("OPTIMAL CATEGORY STRATEGY")
print("-" * 100)


display_columns = [
    "Product",
    "Elasticity",
    "Elasticity_Source",
    "Base_Price",
    "Unit_Cost",
    "Baseline_Units",
    "Optimal_Discount_Pct",
    "Optimal_Price",
    "Expected_Units",
    "Expected_Daily_Profit",
]


print(
    optimization_results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.3f}"
    )
)


# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "=" * 80)
print("✓ PHASES 3B & 3C COMPLETE")
print("=" * 80)

print(
    f"""
Pipeline status:
    ✓ Production BigQuery data loaded
    ✓ Data schema validated
    ✓ Economic variables validated
    ✓ Bayesian elasticity model completed
    ✓ Posterior diagnostics completed
    ✓ Category elasticity analysis completed
    ✓ Nonlinear optimization completed
    ✓ Optimization constraints independently validated
    ✓ Sensitivity analysis completed
    ✓ Visualization artifacts generated
    ✓ Machine-readable artifacts generated

BigQuery:
    Project : {PROJECT_ID}
    Dataset : {DATASET_NAME}
    Table   : {TABLE_NAME}

Global elasticity:
    {elasticity_mean:.4f}

Expected baseline daily profit:
    ${baseline_profit:,.2f}

Expected optimized daily profit:
    ${optimal_profit:,.2f}

Expected daily profit change:
    ${profit_increase:,.2f}

Expected percentage improvement:
    {improvement_pct:.2f}%

Output directory:
    {OUTPUT_DIR}

Execution manifest:
    execution_manifest.json

Ready for:
    Phase 4 — Dashboarding & Communication
"""
)

print("=" * 80)
