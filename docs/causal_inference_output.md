
================================================================================
INITIALIZING PHASE 3A: CAUSAL INFERENCE MODELING
================================================================================

[1/7] Installing / verifying required libraries...
--------------------------------------------------------------------------------
  ✓ dowhy                        available
  ✓ econml                       available
  ✓ networkx                     available
  ✓ pydot                        available
  ✓ lightgbm                     available
  ✓ scipy                        available
  ✓ matplotlib                   available
  ✓ google-cloud-bigquery        available

[2/7] Importing libraries...
--------------------------------------------------------------------------------
  ✓ All libraries imported successfully

[3/7] Loading data from BigQuery...
--------------------------------------------------------------------------------
  Query source: driiiportfolio-506303.starbucks_transactions.analytics_ready_promo_data
  ✓ Loaded 100,000 rows
  ✓ Columns: 15

[4/7] Validating source schema and preparing causal sample...
--------------------------------------------------------------------------------
ERROR:dowhy.causal_graph:Error: Pygraphviz cannot be loaded. No module named 'pygraphviz'
Trying pydot ...
  ✓ Treatment: 20,325 treated / 59,226 control
  ✓ Outcome: daily_net_revenue
  ✓ Explicit adjustment variables: 7
  ✓ DML features after encoding: 9
  ✓ Rows retained: 79,551
  ✓ Rows dropped for required fields: 0

  Summary Statistics:
--------------------------------------------------------------------------------
  Control mean outcome:     $625.33
  Treatment mean outcome:   $642.58
  Naive ATE:                $17.26
  Note: the naive difference is an unadjusted association, not a causal estimate.

[5/7] Constructing causal graph and identifying effect...
--------------------------------------------------------------------------------
  ✓ Causal graph successfully parsed
  ✓ DoWhy causal effect identification completed
  ✓ DML adjustment matrix constructed explicitly

[6/7] Estimating causal effect with EconML LinearDML...
--------------------------------------------------------------------------------
  Fitting full validated sample...
  X shape: (79551, 9)
  T shape: (79551,)
  Y shape: (79551,)
  ✓ DML fit completed successfully

  CAUSAL EFFECT ESTIMATES:
--------------------------------------------------------------------------------
  Average Treatment Effect: $12.32
  95% Confidence Interval: [$10.63, $14.01]
  Standard Error: $0.8639
  Test Statistic: 14.2616
  p-value: 3.79651e-46

  INTERPRETATION:
--------------------------------------------------------------------------------
  ✓ PROMO_20 is estimated to increase daily net revenue by $12.32.
    Direction of estimated causal effect: POSITIVE
  Relative effect versus sample baseline: 1.96%

  STATISTICAL SIGNIFICANCE:
--------------------------------------------------------------------------------
  ✓ Effect is statistically significant at α = 0.05

[7/7] Estimating heterogeneous treatment effects...
--------------------------------------------------------------------------------

  Conditional Average Treatment Effect by Market Segment:
--------------------------------------------------------------------------------
                 mean    std    min    max  count
market_segment                                   
Rural           12.15  47.11 -70.47  89.69  18675
Suburban        10.95  47.12 -71.84  93.85  35760
Urban           14.40  46.11 -78.86  94.31  25116

  Conditional Average Treatment Effect by Product Category:
--------------------------------------------------------------------------------
              mean   std    min    max  count
category                                     
Bakery       19.86  8.88  -4.29  59.15  26517
Drip Coffee -47.54  9.07 -78.86   7.57  26517
Frappuccino  64.64  6.19  46.53  94.31  26517

  Robustness and overlap diagnostics:
--------------------------------------------------------------------------------
  Treatment prevalence: 0.2555
  Propensity min/max: 0.0248 / 0.8667
  Propensity 1%/99%: 0.0383 / 0.6213
  Overlap assessment: No severe empirical overlap violation detected
  Heuristic bias scale: $0.3593
  NOTE: this is a diagnostic, not a formal omitted-variable sensitivity bound.

  Creating visualizations...
--------------------------------------------------------------------------------
  ✓ causal_graph_structure.png
  ✓ ate_comparison_causal_vs_naive.png
  ✓ hte_by_market_segment.png
  ✓ cate_distribution.png
  ✓ outcome_distribution_comparison.png
  ✓ confounder_balance_smd.png
  ✓ propensity_overlap.png

  Saving output artifacts...
--------------------------------------------------------------------------------
  ✓ heterogeneous_treatment_effects.csv
  ✓ confounder_balance_smd.csv
  ✓ causal_inference_results.json

================================================================================
✓ PHASE 3A COMPLETE: CAUSAL INFERENCE MODELING
================================================================================

CAUSAL INFERENCE EXECUTIVE SUMMARY
----------------------------------

Treatment:
    PROMO_20 vs CONTROL

Outcome:
    Daily Net Revenue

Sample:
    79,551 observations

Treatment:
    20,325

Control:
    59,226

DML Causal ATE:
    $12.32

95% Confidence Interval:
    [$10.63, $14.01]

Standard Error:
    $0.8639

Test Statistic:
    14.2616

p-value:
    3.79651e-46

Statistically Significant:
    YES

Relative Effect:
    1.96%

Naive ATE:
    $17.26

Difference:
    $-4.93

Empirical Overlap:
    No severe empirical overlap violation detected

Output Directory:
    /content/phase_3a_outputs

Generated artifacts:
  ✓ ate_comparison_causal_vs_naive.png                 32.8 KB
  ✓ cate_distribution.png                              54.3 KB
  ✓ causal_graph_structure.png                        127.4 KB
  ✓ causal_inference_results.json                       2.0 KB
  ✓ confounder_balance_smd.csv                          0.2 KB
  ✓ confounder_balance_smd.png                         44.5 KB
  ✓ heterogeneous_treatment_effects.csv              5204.0 KB
  ✓ hte_by_market_segment.png                          38.5 KB
  ✓ outcome_distribution_comparison.png                70.0 KB
  ✓ propensity_overlap.png                             36.2 KB

================================================================================
✓ PHASE 3A DELIVERABLES READY FOR PHASE 3B & 3C
================================================================================
