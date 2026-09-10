# STEP 0 — Spatial Residual Diagnostic Report

## 1. Purpose

Determine whether the frozen LSTM V3 model's prediction residuals contain any exploitable spatial signal — specifically, whether a reservoir's prediction error correlates with the concurrent inflow at its correlation-graph neighbors. This evidence is needed to decide whether one final GNN experiment is scientifically justified.

## 2. Exact Files Used

| File | Path |
|------|------|
| V3 predictions | `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` |
| Test split | `data/processed/splits/test.csv` |
| Correlation graph edges | `data/processed/graph/graph_D_correlation_v1_2/edges.csv` |
| Correlation graph metadata | `data/processed/graph/graph_D_correlation_v1_2/metadata.json` |

## 3. Exact Graph Filepath

```
data/processed/graph/graph_D_correlation_v1_2/edges.csv
```

- 16 reservoirs, 41 undirected edges
- Pearson inflow correlation ≥ 0.55, minimum overlap 365 days
- Built using TRAINING DATA ONLY (`leakage_safe: true`)
- 2 connected components (1 isolated node: Pamba)
- **This graph was NOT rebuilt. The pre-existing artifact was used as-is.**

## 4. Data Alignment Checks

| Check | Result |
|-------|--------|
| 16 reservoirs match across graph / V3 / test split | ✅ |
| V3 predictions (1100 rows) = test.csv (1100 rows) | ✅ |
| Date columns aligned row-by-row | ✅ |
| Reservoir columns aligned row-by-row | ✅ |
| target_1d actual values match | ✅ |
| target_3d actual values match | ✅ |
| target_7d actual values match | ✅ |
| Pamba isolated (0 neighbors) | ✅ |
| No NaN/Inf in final correlation inputs | ✅ (0 detected) |

Excluded rows: 80 of 1100 overall (all Pamba rows — zero neighbors, correctly excluded from mean).

## 5. Full Correlation Table

| Horizon | Period | Pearson r | p-value | Spearman r | p-value | N_total | N_valid | N_excluded |
|---------|--------|-----------|---------|------------|---------|---------|---------|------------|
| target_1d | Overall | +0.0219 | 0.4840 | +0.0303 | 0.3333 | 1100 | 1020 | 80 |
| target_1d | Jan-Apr 2025 | −0.0404 | 0.3047 | −0.0602 | 0.1258 | 692 | 648 | 44 |
| target_1d | **Jun-Aug 2025** | **+0.2620** | **<0.0001** | **+0.2935** | **<0.0001** | 408 | 372 | 36 |
| target_3d | Overall | +0.1027 | 0.0010 | +0.0693 | 0.0268 | 1100 | 1020 | 80 |
| target_3d | Jan-Apr 2025 | +0.0346 | 0.3789 | −0.0403 | 0.3059 | 692 | 648 | 44 |
| target_3d | **Jun-Aug 2025** | **+0.3082** | **<0.0001** | **+0.3483** | **<0.0001** | 408 | 372 | 36 |
| target_7d | Overall | +0.1585 | <0.0001 | +0.1420 | <0.0001 | 1100 | 1020 | 80 |
| target_7d | Jan-Apr 2025 | +0.0608 | 0.1219 | +0.0170 | 0.6654 | 692 | 648 | 44 |
| target_7d | **Jun-Aug 2025** | **+0.3148** | **<0.0001** | **+0.4038** | **<0.0001** | 408 | 372 | 36 |

## 6. Pearson Results

- **Overall**: Only target_7d clears the |r| ≥ 0.15 threshold (r = 0.159). 1d and 3d do not.
- **Jan-Apr 2025**: No horizon clears the threshold. All |r| < 0.07.
- **Jun-Aug 2025**: All three horizons show clear signal — r = 0.262 (1d), 0.308 (3d), 0.315 (7d). All highly significant (p < 0.0001).

## 7. Spearman Results

- **Overall**: Only target_7d approaches the threshold (ρ = 0.142). 1d and 3d do not.
- **Jan-Apr 2025**: No horizon clears the threshold. All |ρ| < 0.07.
- **Jun-Aug 2025**: All three horizons show clear signal — ρ = 0.294 (1d), 0.348 (3d), 0.404 (7d). All highly significant (p < 0.0001). The Spearman correlations are **even stronger** than Pearson, suggesting a robust monotonic relationship, not just linear.

## 8. Overall Interpretation

V3's residuals are **weakly correlated** with neighbor inflow when measured across the full test period. This is because the signal is almost entirely absent during the dry season and concentrated in the monsoon. The overall statistics are diluted by the large dry-season sample (648/1020 valid rows) where no spatial signal exists.

## 9. Jan-Apr 2025 Interpretation

> **NO CLEAR SIGNAL** across all horizons.

During the dry season, V3's errors are essentially unrelated to neighbor inflow. This makes physical sense: during low-flow periods, inflows are small and reservoir-to-reservoir hydrological coupling is minimal. There is nothing for a GNN to exploit here.

## 10. Jun-Aug 2025 Interpretation

> **SIGNAL FOUND** across ALL horizons, with increasing strength at longer horizons.

| Horizon | Pearson r | Spearman ρ |
|---------|-----------|------------|
| 1d | 0.262 | 0.294 |
| 3d | 0.308 | 0.348 |
| 7d | 0.315 | 0.404 |

During the monsoon, when a reservoir's neighbors are experiencing high inflow, V3 systematically **under-predicts** that reservoir's inflow (positive residual = prediction > actual, but positive correlation with neighbor inflow means higher neighbor inflow → larger positive residual). The signal strengthens with forecast horizon, which is consistent with the hypothesis that spatial propagation effects become more important over longer time windows.

## 11. June 2025 Regime Shift Concentration

The spatial signal is **entirely concentrated in the Jun-Aug 2025 monsoon period**:

| Horizon | Jan-Apr |Pearson| | Jun-Aug |Pearson| | Ratio |
|---------|---------|---------|-------|
| 1d | 0.040 | 0.262 | **6.5×** |
| 3d | 0.035 | 0.308 | **8.8×** |
| 7d | 0.061 | 0.315 | **5.2×** |

This is consistent with the previously documented June 2025 structural discontinuity. The monsoon regime activates hydrological coupling between reservoirs that does not exist during dry months. V3, being a per-reservoir LSTM with no spatial awareness, cannot capture this coupling.

## 12. Verdicts Per Horizon

| Horizon | Period | Verdict |
|---------|--------|---------|
| target_1d | Overall | NO CLEAR SIGNAL |
| target_1d | Jan-Apr 2025 | NO CLEAR SIGNAL |
| target_1d | **Jun-Aug 2025** | **SIGNAL FOUND** |
| target_3d | Overall | NO CLEAR SIGNAL |
| target_3d | Jan-Apr 2025 | NO CLEAR SIGNAL |
| target_3d | **Jun-Aug 2025** | **SIGNAL FOUND** |
| target_7d | **Overall** | **SIGNAL FOUND** |
| target_7d | Jan-Apr 2025 | NO CLEAR SIGNAL |
| target_7d | **Jun-Aug 2025** | **SIGNAL FOUND** |

> [!IMPORTANT]
> **Summary**: 4 of 9 horizon×period cells show SIGNAL FOUND. All 4 involve the monsoon period, and the signal strengthens with forecast horizon. The evidence suggests that V3's monsoon-period errors contain exploitable spatial structure — but only during high-flow conditions, and any GNN architecture would need to selectively leverage neighbor information rather than uniformly mixing signals across all seasons.
