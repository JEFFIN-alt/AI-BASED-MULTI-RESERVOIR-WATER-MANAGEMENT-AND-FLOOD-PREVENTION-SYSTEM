# Phase 13: Validation Report

## 1. Files Created
**Figures:**
- `results/thesis_visualizations/figures/v3_actual_vs_predicted_1d.png`
- `results/thesis_visualizations/figures/v3_actual_vs_predicted_3d.png`
- `results/thesis_visualizations/figures/v3_actual_vs_predicted_7d.png`
- `results/thesis_visualizations/figures/v3_scatter_1d.png`
- `results/thesis_visualizations/figures/v3_scatter_3d.png`
- `results/thesis_visualizations/figures/v3_scatter_7d.png`
- `results/thesis_visualizations/figures/v3_training_history.png`
- `results/thesis_visualizations/figures/model_comparison_r2.png`
- `results/thesis_visualizations/figures/model_comparison_mae.png`
- `results/thesis_visualizations/figures/spatial_ablation_r2.png`
- `results/thesis_visualizations/figures/v3_period_comparison_r2.png`
- `results/thesis_visualizations/figures/v3_period_comparison_mae.png`
- `results/thesis_visualizations/figures/v3_reservoir_r2_1d.png`
- `results/thesis_visualizations/figures/v3_reservoir_r2_3d.png`
- `results/thesis_visualizations/figures/v3_reservoir_r2_7d.png`
- `results/thesis_visualizations/figures/v3_residuals_1d.png`
- `results/thesis_visualizations/figures/v3_residuals_3d.png`
- `results/thesis_visualizations/figures/v3_residuals_7d.png`
- `results/thesis_visualizations/figures/v3_extreme_flow_limitation.png`
- `results/thesis_visualizations/figures/june_2025_structural_discontinuity.png`

**Tables (CSV):**
- `results/thesis_visualizations/tables/final_v3_metrics.csv`
- `results/thesis_visualizations/tables/final_model_comparison.csv`
- `results/thesis_visualizations/tables/final_period_comparison.csv`
- `results/thesis_visualizations/tables/v3_per_reservoir_metrics.csv`

**Reports:**
- `results/thesis_visualizations/reports/PHASE_13_RESULTS_SUMMARY.md`
- `results/thesis_visualizations/reports/PHASE_13_VALIDATION.md`
- `results/thesis_visualizations/README.md`

## 2. Files Inspected / Source Artifacts Used
- `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` (Primary source for actual vs predicted values)
- `results/lstm_pytorch_v3_logtarget/training_history.csv` (Source for training loss curves)
- `results/final_model_comparison/final_model_comparison_original_units.csv` (Source for baseline comparisons)
- `results/lstm_pytorch_v3_logtarget/v3_period_comparison.csv` (Source for Jan-Apr vs Jun-Aug period differences)
- `results/v3_final_audit/V3_FINAL_AUDIT_REPORT.md` (Source of truth for integrity, GNN metrics, and constraints)

## 3. Metric Consistency Checks
- ✅ No new models were trained.
- ✅ No test set data or targets were altered.
- ✅ Recalculated R² and MAE for LSTM V3 matching audited values exactly (e.g., 1d R² = 0.759).
- ✅ `generate_thesis_package.py` completed execution with exit code 0.
- ✅ All tables are completely free of NaN/Inf values.
- ✅ All generated plots use original inflow units.

## 4. Warnings
- Seaborn threw a deprecation warning regarding the `palette` argument without a `hue` assignment in barplots. This is a purely cosmetic library warning (v0.14.0 future change) and does not affect the correctness of the generated plots.

## 5. Unavailable Analyses
As mandated by the Final Audit Report, the following analyses were **not** fabricated and do not appear in this package:
- High-flow percentage capture rates (81.4%, 77.7%, etc.), as no reproducible script exists for these numbers.
- Absolute confirmation of a "unit conversion" in June 2025, which remains an unverified hypothesis. The visualizations correctly label this period strictly as an "Observed Structural Discontinuity".
