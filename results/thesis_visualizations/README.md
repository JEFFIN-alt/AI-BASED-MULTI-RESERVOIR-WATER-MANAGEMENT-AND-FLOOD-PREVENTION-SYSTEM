# Thesis Visualizations & Results Package

This directory contains the final, verified, presentation-ready visualizations and data tables for the Kerala Multi-Reservoir Inflow Forecasting project. 

## Integrity Guarantee
- **No models were trained or modified** to generate these visualizations.
- **No data was altered.** All plots reflect the frozen test set evaluations produced during the model development phase.
- **Source Artifacts Used:** 
  - `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv`
  - `results/lstm_pytorch_v3_logtarget/training_history.csv`
  - `results/final_model_comparison/final_model_comparison_original_units.csv`
  - `results/lstm_pytorch_v3_logtarget/v3_period_comparison.csv`
  - `results/v3_final_audit/V3_FINAL_AUDIT_REPORT.md`

## Structure
- `figures/`: High-resolution (300 DPI) PNG files for inclusion in thesis documents and presentations.
- `tables/`: Clean CSV representations of metrics for tabular inclusion.
- `reports/`: Markdown summaries interpreting the results with scientifically conservative language.

## Recommended Figures for the Thesis (Top 5)
1. **`figures/model_comparison_r2.png`**: Establishes V3 as the premier model across all baselines.
2. **`figures/spatial_ablation_r2.png`**: Visually demonstrates the negative results of the GNN spatial ablations compared to V3.
3. **`figures/v3_actual_vs_predicted_1d.png`**: The strongest demonstration of the model's predictive capability.
4. **`figures/v3_period_comparison_r2.png`**: Illustrates the difference in predictive capability between the dry and monsoon regimes.
5. **`figures/v3_extreme_flow_limitation.png`**: Provides honest, visual documentation of the peak-dampening limitation of the V3 model.

## Recommended Figures for the Presentation (Top 3)
1. **`figures/model_comparison_r2.png`**: Quick, clear bar chart showing baseline dominance.
2. **`figures/v3_scatter_1d.png`**: Clean scatter plot with R² and MAE, visually proving model alignment to the ideal y=x line.
3. **`figures/june_2025_structural_discontinuity.png`**: Highlights the critical data regime shift that defined the project's complex error behavior.

## Known Limitations
- The "June 2025 Structural Discontinuity" is plotted as an observed error spike; it is not confirmed to be a unit conversion.
- Non-reproducible high-flow statistics mentioned in earlier project stages (e.g., "81.4% capture") were purposefully excluded from this package following the Phase 12 audit.
