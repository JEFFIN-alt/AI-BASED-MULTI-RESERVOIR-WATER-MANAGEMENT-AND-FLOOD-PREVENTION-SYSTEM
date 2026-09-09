# Phase 13: Thesis Results Summary

## 1. Final Model
The final forecasting model for the Kerala Multi-Reservoir Inflow Forecasting project is **LSTM V3 (Log1p Target)**. It is a temporal-only LSTM model featuring a single LSTM layer (hidden size: 64), a dropout layer (0.2), and a fully connected head to predict inflow over 1-day, 3-day, and 7-day horizons simultaneously. A critical component is the `log1p` transformation of the target variable prior to modeling, which enforces non-negative predictions.

## 2. Overall Test Performance
Evaluated on the independent 2025 test set (1,100 samples) in original inflow units, LSTM V3 achieved the following overall metrics:
- **1-Day Horizon**: MAE = 1.837, RMSE = 3.031, R² = 0.759
- **3-Day Horizon**: MAE = 2.095, RMSE = 3.522, R² = 0.654
- **7-Day Horizon**: MAE = 2.703, RMSE = 4.573, R² = 0.501
The model produced zero negative predictions.

## 3. Baseline Comparison
LSTM V3 achieved the strongest overall performance among the evaluated models.
- It substantially outperformed Persistence and a Rolling Mean baseline across all horizons.
- Compared to HistGradientBoosting (HGB), V3 achieved a higher R² and a significantly lower MAE (e.g., 1d MAE of 1.837 vs HGB's 2.414). HGB also exhibited severe positive bias (over-prediction) which V3 avoided.

## 4. Period-wise Performance
The test period encompasses two distinct hydrological regimes: the dry season (Jan-Apr 2025) and the monsoon season (Jun-Aug 2025).
- **Dry Season**: V3 maintained strong correlation (1d R² = 0.730) but exhibited a structural negative bias (Bias = -0.782), indicating consistent over-prediction of minimal/zero flows.
- **Monsoon Season**: V3 demonstrated exceptional 1-day predictive performance (1d R² = 0.812). However, performance severely degraded at the 7-day horizon (7d R² = 0.364), accompanied by a positive bias (+0.994), reflecting the difficulty of predicting long-term monsoon variability.

## 5. Per-Reservoir Observations
Prediction accuracy degraded under certain reservoir conditions. V3 excels on reservoirs like Moozhiyar (1d R² = 0.755) and Chenkulam (1d R² = 0.622). However, the model structurally fails on three reservoirs: Banasura Sagar, Kundala, and Mattupetty, where it recorded negative R² values at the 1-day horizon, performing worse than the mean.

## 6. Error Behavior
Residual analysis reveals that prediction errors are largely zero-centered but exhibit heavy tails, primarily driven by under-prediction during the monsoon season and over-prediction during the extreme dry season.

## 7. Extreme-Flow Limitation
LSTM V3 tends to dampen extreme inflow magnitudes, particularly at longer horizons. For example, during the peak 7-day inflow event on the test set, the actual observed inflow was ~48.6, but the model conservatively predicted ~24.3 (a 50% under-prediction). The model should not be considered a reliable tool for absolute extreme-flood peak magnitude forecasting.

## 8. June 2025 Discontinuity
An observable structural discontinuity occurs around June 2025. The rolling mean absolute error (MAE) sharply increases as the monsoon activates. While previous hypotheses suggested a possible unit-conversion error in the data collection pipeline, no verifiable evidence currently supports this claim. The discontinuity is conservatively treated as a naturally occurring, high-variance monsoon regime shift.

## 9. GNN Investigation Summary
Spatial-model experiments did not produce a meaningful improvement over V3. Ablation studies incorporating Graph Convolutional Networks (GCN) using geographic distance (k-NN) and historical inflow correlation graphs resulted in catastrophic R² collapses during the monsoon. An ablation using an identity graph recovered performance, proving that the unconditional spatial message-passing mechanism was actively destructive. A final experiment using a learned scalar gate to conditionally fuse spatial and temporal representations resulted in the gate collapsing to near-zero, meaning the model learned to ignore the spatial graph entirely.

## 10. Physical Validity
The `log1p` transformation strictly prevents negative (non-physical) inflow predictions, a critical success factor for the model's physical validity.

## 11. Reproducibility
The final model and its associated metrics are fully reproducible. The training pipeline, independent scaler objects, data splits, and test predictions are strictly isolated with zero data leakage.

## 12. Important Limitations
1. **Dampened Peaks**: V3 systematically under-predicts the highest magnitude monsoon events.
2. **Reservoir Heterogeneity**: The model struggles to generalize to reservoirs with highly irregular or near-zero variance profiles (e.g., Banasura Sagar).
3. **Horizon Degradation**: 7-day forecasting in the monsoon period remains highly uncertain.

## 13. Thesis-Safe Interpretation
LSTM V3 represents a robust, highly verified baseline for multi-reservoir inflow forecasting in Kerala. It significantly outperforms non-deep-learning baselines and avoids the catastrophic instability observed in complex spatial GNN architectures. While it demonstrates strong overall 1-day predictive performance, its utility as an extreme flood-warning system is limited by its tendency to conservatively dampen high-flow peaks.
