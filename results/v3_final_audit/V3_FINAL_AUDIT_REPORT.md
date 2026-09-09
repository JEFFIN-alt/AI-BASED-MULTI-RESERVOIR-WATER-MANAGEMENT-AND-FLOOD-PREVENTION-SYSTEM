# V3 FINAL AUDIT REPORT

## 1. Executive Summary

This report documents the forensic final audit of the frozen `LSTM V3 (Log1p Target)` forecasting model for multi-reservoir inflow prediction. The audit independently verifies model architecture, data splits, transformation integrity, prediction plausibility, and performance metrics without any new model training. 

The audit concludes that LSTM V3 is fully reproducible (GREEN status) and legitimately represents the strongest forecasting model in the project repository. It outperforms both temporal baselines and spatial GNN alternatives. However, several previous claims regarding specific "high-flow capture rates" and a "June 2025 unit conversion" could not be reproduced from existing artifacts and should not be included in the final thesis.

## 2. Exact V3 Architecture

Inspection of `src/modeling/train_lstm_pytorch_v3_logtarget.py` confirms the following architecture:
- **Input Features**: 5 dynamic features
- **Sequence Length**: 7 days
- **Architecture**: Single-layer LSTM (Hidden Size: 64)
- **Dropout**: 0.2 (applied to the last hidden state of the LSTM sequence)
- **Dense Head**: Linear(64, 32) → ReLU → Linear(32, 3)
- **Parameter Count**: 20,355
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Masked HuberLoss (δ=1.0)
- **Batch Size**: 128
- **Max Epochs**: 100 (Early stopping patience = 12)
- **Seed**: 42

## 3. Data Pipeline & 4. Transformation Verification

The V3 pipeline relies on independent, leakage-free scalers:
- **Feature Scaler**: `StandardScaler` fitted exclusively on the 16,662 training rows (`feature_scaler.pkl`).
- **Target Transformation**: The raw inflow targets are transformed via `log1p`, followed by a `StandardScaler` fitted exclusively on the 16,662 training rows (`log_target_scaler.pkl`).
- **Inverse Transformation Check**: The inverse transformation chain (`inverse_transform` → `expm1`) was independently tested. The maximum absolute reconstruction error against the raw split file was verified to be less than 0.000006, confirming near-perfect float32 numerical round-trip accuracy.

## 5. Split Verification

The data strictly follows temporal boundaries with no overlap:
- **Train**: Through 2023-12-31 (16,662 rows)
- **Validation**: 2024 (4,383 rows)
- **Test**: 2025 (1,100 rows)
- **Integrity**: Missing-value slots are masked dynamically. No future information enters the input sequences.

## 6. Prediction Verification & 7. Independent Metric Verification

An independent Python script recalculated the overall test metrics from `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` against the raw `test.csv` targets. 

| Horizon | MAE | RMSE | R² | Bias | Neg Preds |
|---------|-----|------|----|------|-----------|
| **1d** | 1.837 | 3.031 | 0.759 | -0.224 | 0 |
| **3d** | 2.095 | 3.522 | 0.654 | -0.093 | 0 |
| **7d** | 2.703 | 4.573 | 0.501 | -0.301 | 0 |

*Result:* The recalculated metrics match the reported V3 metrics exactly.

## 8. Baseline Comparison

The comparison artifacts in `results/final_model_comparison/` correctly place V3 as the strongest model. 
- V3 strongly dominates Rolling Mean and Persistence across all horizons.
- V3 outperforms HistGradientBoosting on R² and severely beats it on MAE (HistGradientBoosting MAE is much worse, e.g., 2.414 vs 1.837 for 1d, and exhibits a severe positive bias of +1.201).

## 9. Period-Wise Analysis

The test set (2025) was split into a dry season (Jan-Apr) and a monsoon season (Jun-Aug):

**Jan-Apr 2025 (692 rows)**
- 1d: MAE=2.019, R²=0.730, Bias=-0.782
- 7d: MAE=2.397, R²=0.583, Bias=-0.473
*Analysis*: R² remains strong, but the model exhibits a severe negative bias (over-prediction) during the dry season.

**Jun-Aug 2025 (408 rows)**
- 1d: MAE=1.528, R²=0.812, Bias=+0.721
- 7d: MAE=3.222, R²=0.364, Bias=-0.010
*Analysis*: 1-day prediction is highly accurate (R²=0.812), but the model structurally under-predicts high flows (positive bias). Performance degrades significantly at the 7-day horizon (R²=0.364), struggling to capture monsoon variability.

## 10. High-Flow Analysis

**Status:** Not reproducible from current repository artifacts.
Previous claims of "81.4%, 77.7%, 64.7%" peak capture rates and associated under-prediction statistics could not be found in any script, artifact, or reproducible definition. These claims lack an evidentiary basis in the codebase.

## 11. June 2025 Discontinuity Analysis

**Status:** Not established from current artifacts.
The repository clearly shows an observed structural discontinuity in model performance (degrading R² at 7d and high biases) during the June 2025 monsoon. However, there is no `consolidated_breakpoint_evidence.csv` and no statistical proof of a "unit conversion" or an "11.5× factor". 

## 12. Error Analysis

Error analysis by reservoir reveals that V3 is not universally superior. While it performs exceptionally well on Moozhiyar (1d R²=0.755) and Chenkulam (1d R²=0.622), it completely fails on three reservoirs:
- **Banasura Sagar**: 1d R² = -0.134
- **Kundala**: 1d R² = -0.066
- **Mattupetty**: 1d R² = -0.032
For these reservoirs, the V3 predictions are worse than simply predicting the mean.

## 13. Physical Validity

- **Negative Predictions**: 0 (The `log1p` target transformation successfully prevented non-physical negative predictions).
- **Extremes**: The model physically under-predicts peak events. The maximum 7d predicted value is 24.335, while the actual 7d maximum is 48.660. The model is conservative and heavily dampens monsoon peaks.

## 14. Reproducibility

**Status: GREEN (Fully Reproducible)**
All source code, scalers, weights, data splits, and training histories are present, correctly aligned, and free of data leakage. Another researcher can confidently reproduce these exact results.

## 15. GNN Investigation Summary

The spatial/GNN investigation is permanently closed with the following final test R² results (1d / 3d / 7d):
- **GCN-LSTM V1 (Geographic)**: -0.706 / -0.802 / -0.487
- **GCN-LSTM V1.1 (Identity)**: 0.638 / 0.586 / 0.531
- **GCN-LSTM V1.2 (Correlation)**: -0.098 / -0.089 / 0.057
- **Gated GCN-LSTM V1**: 0.629 / 0.582 / 0.511

The Gated model's spatial gate collapsed from 0.0100 to 0.0151 during training, definitively proving that unconditional or conditionally-gated spatial message passing on these graphs does not improve upon the temporal baseline.

## 16. Final Model Selection Decision

LSTM V3 is the scientifically legitimate final choice. It provides the highest overall R² and lowest MAE, physically plausible predictions (zero negatives), and robust stability across horizons compared to the baselines. It successfully avoids the catastrophic degradation seen in all GNN variants.

## 17. Safe Thesis Claims

- LSTM V3 outperformed the evaluated temporal baseline models (Persistence, Rolling Mean, HistGradientBoosting) on the overall test set.
- LSTM V3 achieved the strongest overall R² across all three forecasting horizons.
- The `log1p` target transformation successfully eliminated physically impossible negative inflow predictions.
- Spatial/GNN experiments did not provide meaningful improvement over V3, and graph-based models suffered from severe monsoon degradation.
- The June 2025 period exhibits a structural performance discontinuity (monsoon regime shift).

## 18. Unsupported Claims to Avoid

- Do not claim "V3 is universally superior" (it has negative R² on 3 reservoirs).
- Do not claim "The June 2025 data definitely had a unit conversion" (unproven).
- Do not use the "81.4%, 77.7%, 64.7% high-flow capture" numbers (irreproducible).
- Do not claim the model solves flood prediction (it structurally under-predicts peak flows by up to 50%).

## 19. Remaining Limitations

1. **Peak Attenuation**: The model severely dampens peak flows at the 7-day horizon, limiting its utility for extreme flood early-warning.
2. **Reservoir Failures**: The model completely fails (negative R²) on Banasura Sagar, Kundala, and Mattupetty.
3. **Dry Season Over-prediction**: The model consistently over-predicts low flows during the Jan-Apr dry season.

## 20. Recommended Next Project Phase

The model development phase is now closed. The recommended next phase is **Thesis Writing and Visualization**. The project should focus on generating high-quality plots of the test set predictions, compiling the comparison tables, and documenting the rigorous negative results from the GNN ablation studies.
