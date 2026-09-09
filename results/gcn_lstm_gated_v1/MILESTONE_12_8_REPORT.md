# Milestone 12.8 — Gated GCN-LSTM V1: Final GNN Experiment Report

## 1. Research Question

Can selective spatial fusion — where a learned gate controls how much neighbor information enters the prediction — improve the frozen LSTM V3 baseline without the destructive unconditional message passing that caused V1, V1.1, and V1.2 to fail?

## 2. Motivation from Previous Experiments

| Model | Graph | 1d R² | 3d R² | 7d R² | Outcome |
|-------|-------|-------|-------|-------|---------|
| LSTM V3 | None | 0.759 | 0.654 | 0.501 | **Frozen baseline** |
| GCN-LSTM V1 | Geographic k-NN | −0.706 | −0.802 | −0.487 | FLOP |
| GCN-LSTM V1.1 | Identity (self-loops) | 0.638 | 0.586 | 0.531 | Controlled ablation |
| GCN-LSTM V1.2 | Correlation (r≥0.55) | −0.098 | −0.089 | 0.057 | FLOP |

The identity graph (no spatial edges) recovered most performance, while both real graphs caused catastrophic degradation. A Step 0 residual diagnostic then found that V3's monsoon-period errors correlate with neighbor inflow (Spearman ρ up to 0.40 in Jun-Aug), suggesting exploitable spatial structure exists but only during high-flow regimes.

## 3. Integrity Checks

All 12 pre-training checks PASSED:

1. ✅ 16 reservoirs
2. ✅ Graph nodes match canonical ordering
3. ✅ All edges reference valid nodes
4. ✅ Graph leakage-safe (train-only)
5. ✅ 41 undirected edges
6. ✅ 82 bidirectional directed edges
7. ✅ Tensor shapes: X=(B,16,7,5), y=(B,16,3)
8. ✅ Masks unchanged (train=16662, val=4383, test=1100)
9. ✅ Target transformation round-trip verified
10. ✅ Dates: 1131/285/74, non-overlapping
11. ✅ No NaN/Inf
12. ✅ No test information enters training

## 4. Exact Architecture

```
DUAL-BRANCH GATED GCN-LSTM (45,636 parameters)

LOCAL BRANCH:
  Input (B*16, 7, 5) → LSTM(5→64) → h_local (B*16, 64)

SPATIAL BRANCH:
  Input (B, 7, 16, 5) → GCNConv(5→32, self_loops=True, normalize=True) → ReLU
  → reshape (B*16, 7, 32) → LSTM(32→64) → h_spatial (B*16, 64)

GATED FUSION:
  h = h_local + sigmoid(α) × h_spatial
  α initialized to −4.595 → sigmoid(α) ≈ 0.01

FC HEAD:
  h → Linear(64→32) → ReLU → Linear(32→3)
```

The local branch processes raw 5-feature sequences independently per node — architecturally equivalent to V3's LSTM. The spatial branch mirrors V1/V1.2's GCN→LSTM path. The scalar gate (1 parameter) controls how much spatial information enters the final representation.

## 5. Exact Graph Used

```
data/processed/graph/graph_D_correlation_v1_2/edges.csv
```
- 16 reservoirs, 41 undirected edges (82 directed)
- Pearson inflow correlation ≥ 0.55, minimum 365 days overlap
- Training data only, leakage verified
- NOT rebuilt for this experiment

## 6. Masking Implementation

- Missing reservoir-date slots are zero-padded in X (all 5 features = 0)
- GCNConv with `add_self_loops=True, normalize=True` uses symmetric normalization
- Zero-padded neighbors contribute zero vectors to aggregation (mild signal dilution)
- This is identical masking behavior to V1/V1.1/V1.2
- Loss mask ensures zero-padded nodes do NOT contribute to gradients
- The gate mechanism does NOT change masking behavior

## 7. Training Configuration

| Parameter | Value | Same as V1/V1.1/V1.2? |
|-----------|-------|------------------------|
| Seed | 42 | ✅ |
| Batch size | 32 | ✅ |
| Max epochs | 100 | ✅ |
| Learning rate | 0.001 | ✅ |
| Early stopping | 12 epochs | ✅ |
| Optimizer | Adam | ✅ |
| Loss | Masked Huber (δ=1.0) | ✅ |
| Device | CPU | ✅ |

**Training result**: Early stopping at epoch 37, best epoch 25, best val loss 0.0660.

## 8. Overall Results (Original Inflow Units)

| Model | Horizon | MAE | RMSE | R² | Bias | Neg Preds |
|-------|---------|-----|------|----|------|-----------|
| **Gated V1** | target_1d | 2.343 | 3.765 | **0.629** | +0.024 | 0 (0%) |
| **Gated V1** | target_3d | 2.354 | 3.869 | **0.582** | +0.150 | 0 (0%) |
| **Gated V1** | target_7d | 2.736 | 4.524 | **0.511** | +0.047 | 0 (0%) |

## 9. Period-Wise Results

| Period | Horizon | MAE | RMSE | R² | Bias |
|--------|---------|-----|------|----|------|
| Jan-Apr 2025 | target_1d | 2.291 | 3.784 | 0.657 | −1.097 |
| Jan-Apr 2025 | target_3d | 2.348 | 3.957 | 0.614 | −0.739 |
| Jan-Apr 2025 | target_7d | 2.566 | 4.323 | 0.540 | −0.511 |
| Jun-Aug 2025 | target_1d | 2.432 | 3.731 | 0.536 | +1.926 |
| Jun-Aug 2025 | target_3d | 2.364 | 3.715 | 0.463 | +1.659 |
| Jun-Aug 2025 | target_7d | 3.024 | 4.846 | 0.459 | +0.994 |

> [!NOTE]
> No monsoon collapse. All Jun-Aug R² values are positive (0.46–0.54), unlike V1 (−1.7 to −3.1) and V1.2 (−0.5 to −1.4).

## 10. Gate Diagnostics

| Metric | Value |
|--------|-------|
| Initial α | −4.595 |
| Initial gate | 0.0100 |
| Final α | −4.178 |
| Final gate | **0.0151** |
| Gate type | Scalar (1 parameter) |

**Interpretation: The gate COLLAPSED.** It moved from 0.010 to 0.015 — essentially remaining near zero throughout training. The model learned that spatial information from the correlation graph does NOT improve upon the local temporal representation.

The gate did not close further (it marginally opened), which suggests the spatial branch was not actively harmful when gated this low, but the gradient signal was insufficient to justify opening it further.

## 11. Comparison with V3

| Horizon | Gated R² | V3 R² | Delta | Gated MAE | V3 MAE | Delta |
|---------|----------|-------|-------|-----------|--------|-------|
| 1d | 0.629 | **0.759** | **−0.130** | 2.343 | **1.837** | +0.506 |
| 3d | 0.582 | **0.654** | **−0.072** | 2.354 | **2.095** | +0.259 |
| 7d | **0.511** | 0.501 | **+0.010** | 2.736 | **2.703** | +0.033 |

**V3 wins on 2 of 3 horizons.** The gated model marginally exceeds V3 on 7d R² (+0.010) but this is within noise and MAE is still worse.

### Full R² Comparison

| Horizon | Gated V1 | V3 | V1.1 | V1.2 | V1 |
|---------|----------|-----|------|------|----|
| 1d | 0.629 | **0.759** | 0.638 | −0.098 | −0.706 |
| 3d | 0.582 | **0.654** | 0.586 | −0.089 | −0.802 |
| 7d | 0.511 | **0.501** | 0.531 | 0.057 | −0.487 |

## 12. Experiment Classification

### **PARTIAL SUCCESS**

**Rationale:**
- ✅ Clearly improves over V1 (geographic) and V1.2 (correlation) — avoids catastrophic negative R²
- ✅ No monsoon collapse (all Jun-Aug R² > 0.45)
- ✅ Comparable to V1.1 (identity graph) — slight differences within noise
- ❌ Does NOT beat V3 overall (loses on 1d and 3d R², wins marginally on 7d)
- ❌ Gate collapsed to ~0.015 — spatial contribution is negligible
- ❌ The 7d improvement (+0.010 R²) is within noise, not meaningful

**V3 remains the primary forecasting model.**

## 13. Limitations

1. **Gate collapsed**: The scalar gate effectively ignored spatial information. This could mean (a) the correlation graph is not informative enough, (b) the GCN architecture cannot extract useful spatial representations, or (c) the spatial signal is too weak relative to the local temporal signal.

2. **Graph staleness**: The correlation graph was built from training data (through 2023). The June 2025 regime shift may have altered inter-reservoir relationships.

3. **Masking confound**: Zero-padded missing nodes dilute GCN aggregation, potentially weakening an already marginal spatial signal.

4. **Single seed**: Only seed=42 was tested. Results may vary with different initialization.

5. **Scalar gate**: A node-wise or feature-wise gate might have revealed reservoir-specific spatial dependencies, but would have been harder to interpret with only 1 run.

## 14. Final Conclusion

We evaluated geographic and inflow-correlation graph constructions under a GCN-LSTM framework. Both substantially underperformed the strong temporal LSTM baseline (V3), while an identity-graph control recovered much of the lost performance. A residual diagnostic nevertheless revealed neighbor-related structure in V3's monsoon-period residuals (Spearman ρ up to 0.40). A final selective spatial-fusion experiment with a learned gate was conducted. The gate effectively collapsed toward zero (0.015), and the model's performance remained below V3 on 2 of 3 horizons. The gated architecture successfully avoided the catastrophic degradation of unconditional GCN models but could not improve upon the frozen temporal baseline.

**The evidence does not support the hypothesis that GCN-based spatial information improves multi-reservoir inflow forecasting in this dataset and architecture.**

## 15. Explicit Stopping Decision

### **STOP ALL GNN EXPERIMENTATION.**

The GNN branch is frozen as a rigorous negative result. The controlled progression was:

1. Geographic graph → FLOP (catastrophic R²)
2. Identity graph → Recovery (confirms GCN aggregation is the problem)
3. Correlation graph → FLOP (graph construction is not the problem)
4. Gated fusion → PARTIAL SUCCESS (gate collapses, spatial info ignored)

Four experiments with three different graphs and two architectural approaches consistently show that spatial information cannot improve upon the strong temporal LSTM V3 baseline in this reservoir forecasting task.

No fifth graph. No attention mechanism. No hyperparameter sweep. No "one more GNN."

**LSTM V3 is confirmed as the final forecasting model.**

## Files Created

| File | Purpose |
|------|---------|
| `src/modeling/train_gcn_lstm_gated_v1.py` | Training script |
| `models/gcn_lstm_gated_v1/best_model.pt` | Model checkpoint |
| `results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_metrics.csv` | Overall metrics |
| `results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_period_metrics.csv` | Period metrics |
| `results/gcn_lstm_gated_v1/training_history.csv` | Epoch-by-epoch history |
| `results/gcn_lstm_gated_v1/gate_diagnostics.json` | Gate analysis |

## Files NOT Modified

- ✅ `models/lstm_pytorch_v3_logtarget/` — untouched
- ✅ `results/lstm_pytorch_v3_logtarget/` — untouched
- ✅ `results/final_model_comparison/` — untouched
- ✅ `data/processed/` — untouched
- ✅ All previous GCN experiment artifacts — untouched
