# MILESTONE 11.20 — FINAL MODEL COMPARISON

Pure evaluation/consolidation step. No model was trained. No dataset or
train/validation/test split was modified. All numbers below were computed
in this session directly from raw predictions vs. `test.csv` ground truth —
not copied from any prior report.

## 1. Dataset / test-set verification

- `test.csv`: **1100 rows**, date range **2025-01-01 to 2025-08-22**.
- No NaN or Inf in any target column (`target_1d`, `target_3d`, `target_7d`).
- Test set spans Jan–Aug 2025: **692 rows** in Jan–Apr 2025, **408 rows** in
  Jun–Aug 2025 (remaining rows fall in May 2025, outside both named windows).

## 2. Row alignment verification

| model | n rows | row count OK | date+reservoir alignment OK | no NaN | no Inf |
|---|---|---|---|---|---|
| Persistence | 1100 | ✅ | ✅ | ✅ | ✅ |
| Rolling Mean | 1100 | ✅ | ✅ | ✅ | ✅ |
| HistGradientBoosting | 1100 | ✅ | ✅ | ✅ | ✅ |
| LSTM V3 | 1100 | ✅ | ✅ | ✅ | ✅ |

**Data-provenance notes, since this matters for trusting the numbers below:**

- No prediction files existed in this session for **Persistence** or
  **Rolling Mean** — no `data/processed/baselines/` directory was present.
  Both were computed directly from `test.csv`'s existing `inflow_day_1..7`
  columns (Persistence = `inflow_day_7`; Rolling Mean = mean of the 7-day
  window). The formulas were reverse-verified against your reference
  numbers and matched to 5+ decimal places before being trusted — they were
  not assumed.
- The uploaded **`tree_baseline_test_predictions.csv`** (HistGradientBoosting)
  was in **scaled space**, not original inflow units — its own `actual_target_*`
  column did not match `test.csv`'s targets directly. This was caught by
  correlation/residual checking: the scaled "actual" column is an exact
  affine transform of the true original-unit target (R=1.0, max fit residual
  <1e-15, verified independently per horizon). That transform was inverted
  to recover original-unit predictions; ground-truth agreement after
  inversion was confirmed to **<1e-6** against `test.csv` for all three
  horizons before any metric was computed.
- **LSTM V3** predictions (`test_predictions_original_units.csv`) were
  already in original units; its own embedded true-value column was
  confirmed to agree with `test.csv` to <1e-6.

All four models' independently-computed metrics matched the user-supplied
reference numbers to within floating-point precision (max diff <0.00005) —
full verification table is in the script output; nothing below is assumed.

## 3. Overall model comparison — ORIGINAL INFLOW UNITS

| model | target | MAE | RMSE | R² | Bias | neg. count | neg. % |
|---|---|---|---|---|---|---|---|
| LSTM V3 | 1d | **1.7564** | **3.0593** | **0.7549** | -0.339 | 0 | 0.0% |
| HistGradientBoosting | 1d | 2.4142 | 3.2979 | 0.7151 | 1.201 | 0 | 0.0% |
| Persistence | 1d | 1.8248 | 3.3036 | 0.7141 | -0.069 | 0 | 0.0% |
| Rolling Mean | 1d | 2.2362 | 3.9981 | 0.5813 | 0.244 | 0 | 0.0% |
| LSTM V3 | 3d | **1.9827** | **3.5080** | **0.6565** | -0.241 | 0 | 0.0% |
| Persistence | 3d | 2.1428 | 3.8783 | 0.5802 | 0.059 | 0 | 0.0% |
| HistGradientBoosting | 3d | 3.0669 | 4.3281 | 0.4772 | 1.984 | 0 | 0.0% |
| Rolling Mean | 3d | 2.4574 | 4.4075 | 0.4578 | 0.371 | 0 | 0.0% |
| LSTM V3 | 7d | **2.5417** | **4.4300** | **0.5316** | -0.314 | 0 | 0.0% |
| Persistence | 7d | 2.7912 | 4.8623 | 0.4357 | -0.246 | 0 | 0.0% |
| Rolling Mean | 7d | 2.9431 | 5.2067 | 0.3530 | 0.066 | 0 | 0.0% |
| HistGradientBoosting | 7d | 4.2949 | 5.9819 | 0.1459 | 3.163 | 0 | 0.0% |

## 4. Best model by horizon (primary: R², secondary: MAE/RMSE)

| horizon | rank 1 | rank 2 | rank 3 | rank 4 |
|---|---|---|---|---|
| target_1d | **LSTM V3** (R²=0.7549) | HistGradientBoosting (0.7151) | Persistence (0.7141) | Rolling Mean (0.5813) |
| target_3d | **LSTM V3** (R²=0.6565) | Persistence (0.5802) | HistGradientBoosting (0.4772) | Rolling Mean (0.4578) |
| target_7d | **LSTM V3** (R²=0.5316) | Persistence (0.4357) | Rolling Mean (0.3530) | HistGradientBoosting (0.1459) |

LSTM V3 wins on both primary (R²) and secondary (MAE, RMSE) metrics at all
three horizons — a full sweep, not just an R² edge. HistGradientBoosting
degrades sharply at longer horizons (R² collapses from 0.72 → 0.15 between
1d and 7d) and carries a large, growing positive bias (+1.2 → +3.2), unlike
the other three models.

## 5. Period-wise comparison

| period | model | target | R² | MAE | RMSE |
|---|---|---|---|---|---|
| Jan-Apr 2025 | **LSTM V3** | 1d | **0.7217** | 1.9962 | 3.4080 |
| Jan-Apr 2025 | HistGradientBoosting | 1d | 0.7035 | 2.4956 | 3.5175 |
| Jan-Apr 2025 | Rolling Mean | 1d | 0.6941 | 2.1072 | 3.5728 |
| Jan-Apr 2025 | Persistence | 1d | 0.6569 | 2.2692 | 3.7839 |
| Jun-Aug 2025 | **Persistence** | 1d | **0.8288** | 1.0709 | 2.2672 |
| Jun-Aug 2025 | LSTM V3 | 1d | 0.8157 | 1.3499 | 2.3525 |
| Jun-Aug 2025 | HistGradientBoosting | 1d | 0.7223 | 2.2762 | 2.8876 |
| Jun-Aug 2025 | Rolling Mean | 1d | 0.2857 | 2.4549 | 4.6309 |
| Jan-Apr 2025 | **LSTM V3** | 3d | **0.6512** | 2.1186 | 3.7595 |
| Jan-Apr 2025 | Rolling Mean | 3d | 0.6184 | 2.2737 | 3.9322 |
| Jan-Apr 2025 | HistGradientBoosting | 3d | 0.5965 | 2.7383 | 4.0438 |
| Jan-Apr 2025 | Persistence | 3d | 0.5435 | 2.4634 | 4.3012 |
| Jun-Aug 2025 | **Persistence** | 3d | **0.6433** | 1.5990 | 3.0288 |
| Jun-Aug 2025 | LSTM V3 | 3d | 0.6421 | 1.7523 | 3.0340 |
| Jun-Aug 2025 | HistGradientBoosting | 3d | 0.1147 | 3.6243 | 4.7716 |
| Jun-Aug 2025 | Rolling Mean | 3d | -0.0167 | 2.7690 | 5.1135 |
| Jan-Apr 2025 | **LSTM V3** | 7d | **0.5670** | 2.4283 | 4.1965 |
| Jan-Apr 2025 | Rolling Mean | 7d | 0.5210 | 2.5653 | 4.4139 |
| Jan-Apr 2025 | Persistence | 7d | 0.4183 | 2.8828 | 4.8639 |
| Jan-Apr 2025 | HistGradientBoosting | 7d | 0.3552 | 3.6880 | 5.1210 |
| Jun-Aug 2025 | **LSTM V3** | 7d | **0.4687** | 2.7339 | 4.8002 |
| Jun-Aug 2025 | Persistence | 7d | 0.4555 | 2.6358 | 4.8595 |
| Jun-Aug 2025 | Rolling Mean | 7d | 0.0767 | 3.5838 | 6.3281 |
| Jun-Aug 2025 | HistGradientBoosting | 7d | -0.1989 | 5.3242 | 7.2108 |

**This is where the "V3 wins everywhere" story breaks down slightly, and
it's worth stating plainly:** in the Jun–Aug 2025 window (the post-June-9
low/near-zero-flow regime identified in the earlier provenance milestones),
**Persistence edges out LSTM V3 at 1d (R² 0.829 vs 0.816) and 3d (R² 0.643
vs 0.642, effectively a tie)**. LSTM V3 only regains a clear lead at 7d in
that window. This is consistent with a regime where inflow is close to
flat/near-zero day over day — a naive "tomorrow = today" prediction becomes
very hard to beat at short horizons precisely because there is little
genuine dynamics left to model. Rolling Mean and HistGradientBoosting both
collapse badly in Jun–Aug (Rolling Mean R² goes negative at 3d; HGB goes
negative at 7d), overpredicting a regime that is systematically lower-flow
than their historically-averaged training behavior.

## 6. Physical validity / negative prediction comparison

| model | 1d neg. | 3d neg. | 7d neg. |
|---|---|---|---|
| Persistence | 0 | 0 | 0 |
| Rolling Mean | 0 | 0 | 0 |
| HistGradientBoosting | 0 | 0 | 0 |
| **LSTM V3** | **0** | **0** | **0** |

All four models produce zero negative flow predictions on this test set —
Persistence and Rolling Mean are structurally non-negative by construction
(they echo/average non-negative observed inflow), HistGradientBoosting
apparently avoided negatives here despite no explicit non-negativity
constraint, and LSTM V3's `expm1` inversion guarantees non-negativity by
construction (this was the specific finding of Milestone 11.18b, where the
prior V2 model had produced up to 20% negative predictions at 7d). Physical
validity is a wash on this comparison — it no longer discriminates between
models the way it did against V2.

## 7. Final conclusion

**LSTM V3 is the best-performing model overall**, ranking first on R², MAE,
and RMSE simultaneously at all three horizons on the full test set. This
milestone's numbers support treating V3 as the project's final model — but
with three caveats stated explicitly rather than glossed over:

1. **A higher aggregate R² does not mean V3 is universally superior.**
   Broken out by period, Persistence is competitive with — and at 1d/3d in
   the Jun–Aug 2025 window, slightly better than — LSTM V3. The aggregate
   win is real but not uniform across time; a report claiming V3 dominates
   *everywhere* would overstate the evidence in this table.
2. **The June 9, 2025 structural discontinuity (documented in Milestones
   11.18–11.19) remains an unresolved dataset limitation, not something
   this milestone fixes or explains.** The test set spans both the pre- and
   post-break regime, and all models' pre/post (here: Jan–Apr vs Jun–Aug)
   performance is reported separately for exactly this reason — pooling
   them into one number would hide the fact that HistGradientBoosting and
   Rolling Mean effectively fail (near-zero or negative R²) specifically in
   the post-break window, while LSTM V3 and Persistence degrade more
   gracefully. This asymmetry is itself informative but should not be read
   as proof of what caused the discontinuity.
3. **The exact cause of the discontinuity is still unproven** — Milestone
   11.19 ruled out a storage-only or inflow-only explanation but could not
   establish whether the joint collapse in inflow/outflow reflects a real
   hydrological change or a reporting/feed change. This milestone's period
   split is consistent with that earlier finding (models trained on
   pre-break dynamics do worse post-break) but adds no new evidence about
   the cause itself.
4. **The Aug 2025–Jul 2026 data extension remains excluded** from both
   training and this evaluation, as instructed. No model in this comparison
   has been tested against that period, and this report makes no claim
   about performance beyond the existing test window (through 2025-08-22).

**Recommended framing for the BTech report:** present LSTM V3 as the
selected final model based on the full-test-set ranking, but include the
period-split table as evidence that the win margin narrows (and briefly
reverses at short horizons) in the anomalous post-June-2025 regime — this
strengthens the report's credibility rather than weakening it, since it
shows the discontinuity was accounted for rather than ignored.

## Files produced

- `results/final_model_comparison/final_model_comparison_original_units.csv`
- `results/final_model_comparison/final_model_ranking.csv`
- `results/final_model_comparison/final_period_comparison.csv`
- `results/final_model_comparison/MILESTONE_11.20_report.md` (this file)

---

## MILESTONE 11.20 COMPLETE

### Final ranking table (primary: R² desc | secondary: MAE, RMSE asc)

| horizon | rank | model | R² | MAE | RMSE |
|---|---|---|---|---|---|
| 1d | 1 | **LSTM V3** | 0.7549 | 1.7564 | 3.0593 |
| 1d | 2 | HistGradientBoosting | 0.7151 | 2.4142 | 3.2979 |
| 1d | 3 | Persistence | 0.7141 | 1.8248 | 3.3036 |
| 1d | 4 | Rolling Mean | 0.5813 | 2.2362 | 3.9981 |
| 3d | 1 | **LSTM V3** | 0.6565 | 1.9827 | 3.5080 |
| 3d | 2 | Persistence | 0.5802 | 2.1428 | 3.8783 |
| 3d | 3 | HistGradientBoosting | 0.4772 | 3.0669 | 4.3281 |
| 3d | 4 | Rolling Mean | 0.4578 | 2.4574 | 4.4075 |
| 7d | 1 | **LSTM V3** | 0.5316 | 2.5417 | 4.4300 |
| 7d | 2 | Persistence | 0.4357 | 2.7912 | 4.8623 |
| 7d | 3 | Rolling Mean | 0.3530 | 2.9431 | 5.2067 |
| 7d | 4 | HistGradientBoosting | 0.1459 | 4.2949 | 5.9819 |
