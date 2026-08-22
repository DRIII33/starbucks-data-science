 
================================================================================
INITIALIZING PHASE 3B & 3C
BAYESIAN ELASTICITY + NONLINEAR PROFIT OPTIMIZATION
================================================================================

Project:
  BigQuery Project : driiiportfolio-506303
  Dataset          : starbucks_transactions
  Table            : analytics_ready_promo_data
  Environment      : Google Colab
================================================================================

[1/8] Installing / verifying required libraries...
--------------------------------------------------------------------------------
  ✓ pandas                    available
  ✓ numpy                     available
  ✓ pymc                      available
  ✓ arviz                     available
  ✓ scipy                     available
  ✓ matplotlib                available
  ✓ google-cloud-bigquery     available
  ✓ db-dtypes                 available

[2/8] Importing libraries...
--------------------------------------------------------------------------------
  ✓ NumPy       : 2.1.3
  ✓ Pandas      : 2.2.3
  ✓ PyMC        : 5.28.5
  ✓ ArviZ       : 0.22.0
  ✓ SciPy       : 1.16.3
  ✓ All imports completed successfully

[3/8] Configuring pipeline...
--------------------------------------------------------------------------------
  BigQuery table:
    driiiportfolio-506303.starbucks_transactions.analytics_ready_promo_data

  Modeling:
    Global draws       : 2,000
    Global tune        : 1,000
    Category draws     : 1,000
    Category tune      : 500
    Target acceptance  : 0.9

  Optimization:
    Discount range     : 0% - 30%
    Minimum units      : 50

  Output directory:
    /content

[4/8] Loading production data from BigQuery...
--------------------------------------------------------------------------------
  → Inspecting table schema...
  ✓ Table found
  ✓ Available columns: 15
  ✓ Required schema validation passed

  → Executing query...
    LIMIT 100,000

  ✓ Loaded 100,000 rows
  ✓ Loaded 11 columns

[5/8] Validating production dataset...
--------------------------------------------------------------------------------
  ✓ Required columns present
  ✓ Numeric fields converted

  Missing-value summary:
    ✓ transaction_date          0
    ✓ store_id                  0
    ✓ category                  0
    ✓ treatment_group           0
    ✓ promo_id                  0
    ✓ discount_pct              0
    ✓ base_price                0
    ✓ unit_cost                 0
    ✓ daily_units_sold          0
    ✓ daily_net_revenue         0
    ✓ daily_profit              0

  Removed 0 rows with missing modeling fields

  Final validated dataset:
    Rows       : 100,000
    Categories : 3
    Date range : 2022-01-01 00:00:00 → 2023-10-29 00:00:00

  Price statistics:
       base_price  price_point  daily_units_sold  discount_pct
count  100000.000   100000.000          100000.0    100000.000
mean        4.167        3.716           183.421         0.108
std         1.027        1.088            58.878         0.137
min         3.000        2.010              55.0         0.000
25%         3.000        3.000             138.0         0.000
50%         4.000        3.685             181.0         0.000
75%         5.500        4.400             223.0         0.200
max         5.500        5.500             436.0         0.330

================================================================================
PHASE 3B: BAYESIAN PRICE ELASTICITY MODELING
================================================================================

[3B-1] Preparing elasticity dataset...
--------------------------------------------------------------------------------
  ✓ Valid elasticity observations: 100,000
  ✓ Categories: 3
  ✓ Price range: $2.01 - $5.50
  ✓ Unit range: 55 - 436

[3B-2] Checking price variation...
--------------------------------------------------------------------------------
  ✓ Global unique price points: 9

  Category price variation:
   category  n_obs  unique_prices  min_price  max_price
     Bakery  33334              3      2.680        4.0
Drip Coffee  33333              3      2.010        3.0
Frappuccino  33333              3      3.685        5.5

[3B-3] Building pooled Bayesian elasticity model...
--------------------------------------------------------------------------------
  X shape: (100000,)
  Y shape: (100000,)

  Model:
    log(units) = intercept + elasticity × log(price) + error

  Interpretation:
    elasticity = % change in quantity / % change in price

  → Sampling global posterior...
                                                                                                                   
  Progress                      Draw   Divergences   Step size   Grad evals   Speed           Elapsed   Remaining  
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━   3000   0             0.161       31           20.03 draws/s   0:02:29   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━   3000   0             0.191       7            20.56 draws/s   0:02:25   0:00:00    
                                                                                                                   



  ✓ Global Bayesian sampling completed

[3B-4] Evaluating Bayesian convergence...
--------------------------------------------------------------------------------

                mean        sd   hdi_3%   hdi_97%  mcse_mean   mcse_sd     ess_bulk     ess_tail     r_hat
intercept   6.412323  0.002181  6.40845  6.416691   0.000058  0.000047  1432.518231  1438.750188  1.004890
elasticity -0.988311  0.001680 -0.99154 -0.985184   0.000045  0.000036  1414.596830  1331.803839  1.004373
sigma       0.161575  0.000366  0.16089  0.162273   0.000009  0.000008  1613.898645  1357.707322  1.001029

  Diagnostics:
    Divergences : 0
    Max R-hat   : 1.0049
    Min ESS     : 1415
    Mean ESS    : 1487

  ✓ Bayesian convergence checks passed

[3B-5] Extracting posterior elasticity...
--------------------------------------------------------------------------------

  GLOBAL PRICE ELASTICITY
    Posterior mean : -0.988311
    Posterior SD   : 0.001680
    95% HDI        : [-0.991769, -0.985184]

    ✓ Estimated elasticity is negative, consistent with conventional demand behavior.
    Demand classification: INELASTIC

    Interpretation:
    A 1% increase in price is associated with approximately a 0.99% change in quantity in the opposite direction.

[3B-6] Estimating category-specific elasticity...
--------------------------------------------------------------------------------

  Category: Bakery
    Observations : 33,334
    Price points : 3
    Elasticity: -1.0315
    95% HDI: [-1.0414, -1.0215]
    Divergences: 0
    R-hat max: 1.0013
    ESS min: 525
    Status: CONVERGED

  Category: Drip Coffee
    Observations : 33,333
    Price points : 3
    Elasticity: -0.6026
    95% HDI: [-0.6119, -0.5924]
    Divergences: 0
    R-hat max: 1.0044
    ESS min: 748
    Status: CONVERGED

  Category: Frappuccino
    Observations : 33,333
    Price points : 3
    Elasticity: -1.2986
    95% HDI: [-1.3092, -1.2883]
    Divergences: 0
    R-hat max: 1.0086
    ESS min: 579
    Status: CONVERGED

  Category elasticity results:
   category  n_observations  unique_price_points  elasticity_mean  elasticity_sd  hdi_95_lower  hdi_95_upper  divergences  rhat_max  ess_bulk_min    status
     Bakery           33334                    3        -1.031472       0.005176     -1.041351     -1.021482            0  1.001307     524.74176 CONVERGED
Drip Coffee           33333                    3        -0.602595       0.005166     -0.611879     -0.592421            0  1.004356     748.13991 CONVERGED
Frappuccino           33333                    3        -1.298623       0.005313     -1.309224     -1.288350            0  1.008569     579.23535 CONVERGED

[3B-7] Preparing validated elasticities for optimization...
--------------------------------------------------------------------------------

  Elasticities entering optimization:
    Bakery                           -1.0315 (category_specific_bayesian)
    Drip Coffee                      -0.6026 (category_specific_bayesian)
    Frappuccino                      -1.2986 (category_specific_bayesian)

[3B-8] Creating Phase 3B visualizations...
--------------------------------------------------------------------------------
  ✓ Saved: /content/bayesian_elasticity_trace.png
  ✓ Saved: /content/elasticity_posterior_distribution.png

  Generating posterior predictive distribution...
  ✓ Saved: /content/bayesian_elasticity_ppc.png

  Generating category demand curves...
  ✓ Saved: /content/demand_curves_by_category.png

[3B-12] Saving Phase 3B artifacts...
--------------------------------------------------------------------------------
  ✓ Saved: /content/elasticity_trace.nc
  ✓ Saved: /content/elasticity_model_summary.csv
  ✓ Saved: /content/category_elasticity_results.csv
  ✓ Saved: /content/elasticity_results.json

================================================================================
✓ PHASE 3B COMPLETE
================================================================================

================================================================================
PHASE 3C: NONLINEAR CONSTRAINED PROFIT OPTIMIZATION
================================================================================

[3C-1] Building category-level optimization dataset...
--------------------------------------------------------------------------------
  ✓ Optimization categories: 3

  Category baseline metrics:
   category  base_price  unit_cost  baseline_units  baseline_revenue  baseline_profit  observations  elasticity          elasticity_source
     Bakery         4.0        1.2      186.778994        648.303073       424.168280         33334   -1.031472 category_specific_bayesian
Drip Coffee         3.0        0.5      233.155582        614.128422       497.550632         33333   -0.602595 category_specific_bayesian
Frappuccino         5.5        1.5      130.327603        617.357676       421.866272         33333   -1.298623 category_specific_bayesian

[3C-2] Defining nonlinear demand and profit functions...
--------------------------------------------------------------------------------

[3C-3] Calculating baseline economics...
--------------------------------------------------------------------------------
  Baseline expected daily profit: $1,627.18

[3C-4] Checking optimization feasibility...
--------------------------------------------------------------------------------
  Bakery                         units@30%=  269.84 | margin@30%=$   1.60
  Drip Coffee                    units@30%=  289.06 | margin@30%=$   1.60
  Frappuccino                    units@30%=  207.11 | margin@30%=$   2.35

[3C-5] Configuring nonlinear optimization...
--------------------------------------------------------------------------------
  Objective:
    Maximize expected daily profit
  Decision variables:
    3 category-specific discount percentages
  Method:
    SciPy SLSQP nonlinear constrained optimization

  Constraints:
    0% <= discount <= 30%
    Expected units >= 50
    Discounted price >/= unit cost

[3C-6] Solving nonlinear optimization problem...
--------------------------------------------------------------------------------

  Solver success : True
  Status code    : 0
  Iterations     : 5
  Message        : Optimization terminated successfully

[3C-7] Independently validating optimization solution...
--------------------------------------------------------------------------------
  Discount bounds valid : True
  Unit constraint valid : True
  Margin constraint valid : True
  Profit finite         : True
  Final validation      : True

[3C-8] Building optimization results table...
--------------------------------------------------------------------------------
    Product  Elasticity          Elasticity_Source  Base_Price  Unit_Cost  Baseline_Units  Optimal_Discount_Pct  Optimal_Price  Expected_Units  Expected_Revenue  Expected_COGS  Expected_Daily_Profit  Baseline_Daily_Profit  Daily_Profit_Change
     Bakery     -1.0315 category_specific_bayesian      4.0000     1.2000        186.7790                0.0000         4.0000        186.7790          747.1160       224.1348               522.9812               522.9812               0.0000
Drip Coffee     -0.6026 category_specific_bayesian      3.0000     0.5000        233.1556                0.0000         3.0000        233.1556          699.4667       116.5778               582.8890               582.8890              -0.0000
Frappuccino     -1.2986 category_specific_bayesian      5.5000     1.5000        130.3276                0.0000         5.5000        130.3276          716.8018       195.4914               521.3104               521.3104              -0.0000

  PROFIT IMPACT
--------------------------------------------------------------------------------
  Baseline daily profit : $1,627.18
  Optimized daily profit: $1,627.18
  Daily improvement     : $0.00
  Improvement %         : 0.00%
  Annualized difference : $0.00

[3C-9] Running discount sensitivity analysis...
--------------------------------------------------------------------------------
 uniform_discount_pct  daily_profit  minimum_expected_units  minimum_margin  feasible
                 0.00      1,627.18                  130.33            2.50      True
                 5.00      1,596.04                  139.30            2.35      True
                10.00      1,561.85                  149.44            2.20      True
                15.00      1,524.07                  160.95            2.05      True
                20.00      1,481.98                  174.13            1.90      True
                25.00      1,434.67                  189.36            1.75      True
                30.00      1,380.94                  207.11            1.60      True

[3C-10] Creating Phase 3C visualizations...
--------------------------------------------------------------------------------
  ✓ Saved: /content/optimal_discount_strategy.png
  ✓ Saved: /content/profit_improvement_comparison.png
  ✓ Saved: /content/sensitivity_analysis_profit_discount.png

[3C-11] Saving Phase 3C artifacts...
--------------------------------------------------------------------------------
  ✓ Saved: /content/optimization_strategy_detailed.csv
  ✓ Saved: /content/optimization_results.json
  ✓ Saved: /content/optimization_diagnostics.json

[3C-12] Creating execution manifest...
--------------------------------------------------------------------------------
  ✓ Saved: /content/execution_manifest.json

================================================================================
PHASE 3C EXECUTIVE SUMMARY
================================================================================

┌────────────────────────────────────────────────────────────────────────────┐
│                        EXPECTED PROFIT OPTIMIZATION                        │
├────────────────────────────────────────────────────────────────────────────┤
│  Baseline daily profit:      $       1,627.18                        │
│  Optimized daily profit:     $       1,627.18                        │
│  Daily profit change:        $           0.00                        │
│  Percentage improvement:               0.00%                         │
│  Annualized difference:      $           0.00                        │
├────────────────────────────────────────────────────────────────────────────┤
│                             BAYESIAN ELASTICITY                            │
├────────────────────────────────────────────────────────────────────────────┤
│  Global elasticity:             -0.9883                                   │
│  95% HDI:                    [-0.9918, -0.9852]                    │
│  Demand classification:      INELASTIC                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                              MODEL DIAGNOSTICS                             │
├────────────────────────────────────────────────────────────────────────────┤
│  Bayesian divergences:       0                                              │
│  Bayesian max R-hat:         1.0049                                     │
│  Bayesian minimum ESS:       1415                                       │
│  Bayesian convergence:       PASSED                               │
│  Optimization convergence:   PASSED                            │
│  Independent validation:     PASSED                           │
└────────────────────────────────────────────────────────────────────────────┘


OPTIMAL CATEGORY STRATEGY
----------------------------------------------------------------------------------------------------
    Product  Elasticity          Elasticity_Source  Base_Price  Unit_Cost  Baseline_Units  Optimal_Discount_Pct  Optimal_Price  Expected_Units  Expected_Daily_Profit
     Bakery      -1.031 category_specific_bayesian       4.000      1.200         186.779                 0.000          4.000         186.779                522.981
Drip Coffee      -0.603 category_specific_bayesian       3.000      0.500         233.156                 0.000          3.000         233.156                582.889
Frappuccino      -1.299 category_specific_bayesian       5.500      1.500         130.328                 0.000          5.500         130.328                521.310

================================================================================
✓ PHASES 3B & 3C COMPLETE
================================================================================

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
    Project : driiiportfolio-506303
    Dataset : starbucks_transactions
    Table   : analytics_ready_promo_data

Global elasticity:
    -0.9883

Expected baseline daily profit:
    $1,627.18

Expected optimized daily profit:
    $1,627.18

Expected daily profit change:
    $0.00

Expected percentage improvement:
    0.00%

Output directory:
    /content

Execution manifest:
    execution_manifest.json

Ready for:
    Phase 4 — Dashboarding & Communication

================================================================================
