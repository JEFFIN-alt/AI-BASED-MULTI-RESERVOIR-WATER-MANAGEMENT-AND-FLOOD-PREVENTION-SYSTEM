# STAGE 5 REPORT — FORECAST PIPELINE INTEGRATION & DATA PROVENANCE

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 5 boundary (Stage 6 NOT started)
**Objective:** Make the live LSTM forecast pipeline explicit, unit-correct, provenance-aware and scientifically honest.

---

## 0. ⚠️ STOP-AND-REPORT — MISSING PHYSICAL QUANTITIES (Req. 25)

This is the headline finding and it requires your sign-off.

The live Digital Twin **cannot legitimately produce 2 of the 5 features the frozen
LSTM V3 requires.** They are reported here rather than invented.

| Missing quantity | Frozen V3 expectation | Live `ReservoirNetwork` capability | Why it cannot be derived |
|---|---|---|---|
| **`water_level`** | reservoir stage in **METRES** (train range 41.9 – 1758.65 m; scaler mean 789.30, scale 449.05) | exposes `storage` (MCM) and `storage_fraction` **only** | There is **no elevation–storage (rating) curve** anywhere in the repository. Metres cannot be obtained from storage without fabricating physics. |
| **`rainfall`** | **millimetres** (train range 0 – 264.6 mm) | no rainfall state at all | Rainfall is an external meteorological input; the reservoir network does not model it. |

**The previous live code violated this in two ways:**

1. It passed `get_simulated_water_level_proxy()` — a **storage percentage (0–100)** —
   into the `water_level` slot that was trained on **metres**. That is not a unit
   conversion; it is a different physical quantity. A value of `50` becomes a
   z-score of `(50 − 789.30)/449.05 ≈ −1.65`, i.e. a plausible-looking but
   physically meaningless input.
2. It passed a hard-coded `0.0` for `rainfall` — a **silent zero-fill** of a
   missing real-world measurement (explicitly forbidden by Req. 15).

**Resolution implemented (and needing your confirmation):**

* No conversion was invented. No metres were fabricated.
* `water_level` and `rainfall` are now **explicitly marked** `UNAVAILABLE` in every
  provenance record, with a machine-readable reason.
* The demonstration twin may still render a forecast (Req. 10), using an
  **explicitly labelled `SYNTHETIC_DEMO` placeholder** — a documented statistic of
  the frozen training split, constant per reservoir, never presented as a
  measurement and never derived from storage.
* Setting `AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS=0` disables the placeholder path
  entirely, and the live forecast then reports `FORECAST_UNAVAILABLE` naming the
  exact missing quantities.

**Please confirm** that the labelled `SYNTHETIC_DEMO` placeholder is acceptable for
the demonstration twin, or instruct that the live forecast should always report
`FORECAST_UNAVAILABLE` until a real water-level and rainfall source is integrated.

---

## 1. Live LSTM Pipeline — BEFORE Changes

```
GlobalSimulationState._record_history()          (every step)
    row = [manual_inflows[v_name],               <-- COMMANDED baseline, not observed
           get_simulated_water_level_proxy(),    <-- storage PERCENT into a METRES slot
           res_obj.state.storage_mcm,            <-- MCM  (unit-correct)
           0.0,                                   <-- SILENT zero-fill for rainfall
           res_obj.state.release_mcm_day]        <-- MCM/day (unit-correct)
    (fallback values 50.0 / inflow when res_obj is missing -> fabricated)
    buffer seeded with 7 IDENTICAL rows at construction -> fabricated 7-day history

GlobalSimulationState._run_ml_pipeline()
    np.array(buffer) -> DataFrame(columns=[inflow, water_level, live_storage, rainfall, total_outflow])
    LiveForecaster.predict(df)
        -> flat 51-col row -> feature_scaler.transform -> (1,7,5) tensor
        -> LSTM -> log_target_scaler.inverse_transform -> expm1 -> MCM/day
    on ANY exception: lstm_forecasts[v] = {1d: manual_inflows[v], 3d: ..., 7d: ...}
        <-- ECHO of the commanded inflow, presented as "LSTM forecast"
    if not ml_ready: same echo fallback

    GNN: same 5-column buffer passed RAW (unscaled) to a model whose contract
         requires ALREADY-SCALED inputs
    control: gnn_adapter.get_control_forecasts(..., "lstm_primary")
```

**Defects present before Stage 5**

| # | Defect | Severity |
|---|---|---|
| D1 | storage **percent** substituted for `water_level` **metres** | Critical |
| D2 | `rainfall` silently zero-filled | High |
| D3 | `inflow` taken from the *commanded* baseline, not the authoritative state | High |
| D4 | 7-day buffer pre-seeded with 7 identical rows (fabricated history) | High |
| D5 | echo fallbacks present fabricated numbers as LSTM output | High |
| D6 | one try/except coupled the validated LSTM to the experimental GNN | Medium |
| D7 | GNN fed unscaled inputs contrary to its own documented contract | Medium |
| D8 | the "consecutive dates" check warned unconditionally and never validated | Medium |
| D9 | no provenance anywhere in the forecast payload | Critical |

---

## 2. Live LSTM Pipeline — AFTER Changes

```
GlobalSimulationState._record_history()          (every authoritative step)
    { step_index, inflow <- state.inflow_mcm_day,          MCM/day  SIMULATED
      live_storage <- state.storage_mcm,                   MCM      SIMULATED
      total_outflow <- state.release_mcm_day }             MCM/day  SIMULATED
    buffer NOT pre-seeded; 7 DISTINCT step indices required

GlobalSimulationState._forecast_one_reservoir(v_name, p_name)
    1. warm-up gate   -> <7 distinct steps  => WARMUP_INSUFFICIENT_HISTORY (no fabrication)
    2. feature provenance via build_live_feature_inputs(...)
       water_level / rainfall => UNAVAILABLE  (or SYNTHETIC_DEMO when opted in)
    3. if any UNAVAILABLE  => FORECAST_UNAVAILABLE + exact missing quantities
    4. if LSTM not loaded  => FORECAST_UNAVAILABLE + MODEL_NOT_LOADED
    5. LiveForecaster.predict(df, input_provenance=summary)
       - scaler contract verified at load time (51 cols, frozen order)
       - contiguity of the 7-day window now actually validated
       - returns provenance block with the forecast
    6. exceptions => FORECAST_UNAVAILABLE + INFERENCE_ERROR (never an echo)

Advisory GNN (separate, optional, never control)
    _build_gnn_history() -> scaled (7,5) windows in FEATURE_ORDER

Control: gnn_adapter.get_control_forecasts(..., "lstm_primary") + provenance merge
```

---

## 3. Exact Feature Mapping (Req. 7, 8)

Frozen LSTM V3 input tensor: **(N, 7, 5)** — 7 daily timesteps × 5 dynamic features,
in this exact order.

| # | Frozen feature | Live source | Mapping verdict |
|---|---|---|---|
| 1 | `inflow` | `ReservoirNode.state.inflow_local` (MCM/day) | ✅ legitimate, unit-correct |
| 2 | `water_level` | — none — | ❌ **UNAVAILABLE** (no rating curve) |
| 3 | `live_storage` | `ReservoirNode.state.storage` (MCM) | ✅ legitimate, unit-correct |
| 4 | `rainfall` | — none — | ❌ **UNAVAILABLE** (not modelled) |
| 5 | `total_outflow` | `ReservoirNode.state.total_outflow` = controlled_release + spill (MCM/day) | ✅ legitimate, unit-correct |

**Not mapped (no trained counterpart):** `inflow_routed`, `controlled_release`,
`spill`, `gate_position`, `transmission_loss`, `capacity`, `max_release`,
`storage_fraction`.

**Deliberate non-mapping:** there is no mapping from `storage` / `storage_fraction`
to `water_level`. That mapping would require an elevation–storage curve, which does
not exist in this repository.

---

## 4. Units for Every Feature

| Feature | Unit | Source of the unit |
|---|---|---|
| `inflow` | MCM/day | `src/common/units.py::CANONICAL_STATE_UNITS`, frozen train pipeline |
| `water_level` | **metres** | frozen train data (per-reservoir ranges match published Kerala dam levels: Idamalayar ~132–166 m, Idukki ~703–732 m, Ponmudi ~684–708 m, Anayirankal ~1188–1207 m) |
| `live_storage` | MCM | frozen train data (Idukki max 1845.364 ≈ published live capacity) |
| `rainfall` | millimetres | frozen train data (max 264.6 mm) |
| `total_outflow` | MCM/day | frozen train data |
| `forecast_1d/3d/7d` | **MCM/day** | inverse transform target of the frozen model |

Twin display units (`m³/s`, storage ratio) are separate and unchanged; the twin's
`water_level` field is a **display ratio** (`level_unit: proxy_ratio`), explicitly
documented in the adapter as *not* a model feature.

---

## 5. Which Features Are Real / Simulated / Unavailable

The vocabulary is closed (`src/modeling/v3_feature_contract.py::FeatureProvenance`)
so no ambiguous terminology is introduced:

| Provenance | Meaning | Live demo | Validated offline evaluation |
|---|---|---|---|
| `MEASURED_HISTORICAL` | a real observation | never | all 5 |
| `SIMULATED` | produced by the authoritative `ReservoirNetwork` | `inflow`, `live_storage`, `total_outflow` | never |
| `SYNTHETIC_DEMO` | documented synthetic placeholder, demo only | `water_level`, `rainfall` (opt-in) | never |
| `UNAVAILABLE` | physical quantity not available | `water_level`, `rainfall` when demo mode is off | never |

**No live feature is ever labelled as a real measurement.**

---

## 6. Forecast Provenance Design (Req. 11, 12, 13)

### 6.1 Per-feature record
Every feature fed to the model carries:
`{name, unit, value, provenance, is_simulated, is_available, note}`.

### 6.2 Per-forecast record
Every live forecast carries:

```
forecast_source            = "FROZEN_LSTM_V3"
forecast_unit              = "MCM/day"
horizons                   = ["forecast_1d","forecast_3d","forecast_7d"]
model_version              = "LSTM_V3_LOGTARGET"
forecast_status            = DEMONSTRATION_ONLY | WARMUP_INSUFFICIENT_HISTORY
                           | FORECAST_UNAVAILABLE
forecast_provenance        = SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL
                           | REAL_MEASUREMENT_INPUTS_FROZEN_MODEL
                           | REQUIRED_FEATURE_UNAVAILABLE | MODEL_NOT_LOADED
                           | INFERENCE_ERROR | INSUFFICIENT_HISTORY
is_simulated               = true
validated_metrics_apply    = false          <-- reported V3 metrics do NOT apply
input_provenance           = {features, unavailable, synthetic_demo, simulated,
                              measured_historical, all_real_measurements}
unavailable_features       = ["water_level","rainfall"]   (when applicable)
```

`forecast_status` becomes `VALIDATED` **only** if every input is
`MEASURED_HISTORICAL` — which the live path can never produce by construction.

### 6.3 Twin payload
Per reservoir: `forecast_status`, `forecast_provenance`, `forecast_source`,
`forecast_is_simulated`, `forecast_validated_metrics_apply`,
`forecast_unavailable_features`.

Top level: a `forecast_provenance` block that states the two pipelines explicitly —

```
validated_evaluation : "historical held-out data -> frozen LSTM V3"
live_path            : "authoritative simulation state -> simulation/demo feature inputs -> frozen LSTM V3"
live_forecasts_are_validated : false
```

This preserves the distinction required by Req. 13 and prevents the live
demonstration forecast from being read as the validated evaluation.

---

## 7. Files Modified

| File | Change |
|---|---|
| `src/modeling/v3_feature_contract.py` | **NEW** — frozen feature contract, provenance vocabulary, availability statements, synthetic-demo placeholder table, scaler verification |
| `src/modeling/inference.py` | Scaler contract verified at load; provenance returned with every forecast; real contiguity validation; `ForecastUnavailableError`; refuses UNAVAILABLE inputs. **Numeric path unchanged** |
| `src/dashboard/api/state_manager.py` | Buffer rebuilt from authoritative state (3 simulated features only, distinct step indices, no pre-seeding); per-reservoir forecast with full provenance; LSTM/GNN initialisation decoupled; GNN given contract-correct scaled inputs; echo fallbacks removed |
| `src/dashboard/sim_bridge.py` | Forecast provenance passed through into the twin state |
| `src/dashboard/twin_component/state_adapter.py` | Per-reservoir + top-level forecast provenance exposed in the twin payload |
| `tests/test_stage5_forecast_provenance.py` | **NEW** — 38 tests |

**NOT modified (Req. 1, 2):** `best_model.pt`, `log_target_scaler.pkl`,
`data/processed/scaled/feature_scaler.pkl`, frozen V3 prediction CSVs,
`results/phase15_v3_validation/*`, `src/controller/*` (MPC, SafetyLayer, objective),
`src/network_env/reservoir_network.py`, `src/network_env/topology_config.yaml`.

---

## 8. Tests Added (Req. 17)

`tests/test_stage5_forecast_provenance.py` — **38 tests**:

| Group | What it proves |
|---|---|
| Frozen artifacts (3) | model/scalers unchanged byte-for-byte after load + inference; Phase 15.3 manifest still holds; architecture is the frozen one (input 5, output 3) |
| Feature contract (6) | exact feature order; 7-timestep sequence; 51 scaler columns with the 35 dynamic first in feature-major order; a reordered/wrong-size scaler is **rejected**; permutation materially changes output (why pinning matters); static columns cannot influence output |
| Units & horizons (6) | `forecast_unit == "MCM/day"`; horizons 1d/3d/7d; feature units published (metres / mm / MCM / MCM/day); forecasts finite across input scales |
| Missing / invalid data (5) | missing column, wrong window length, NaN history and **gapped dates** all raise; UNAVAILABLE inputs raise rather than substitute |
| No storage→level conversion (3) | no proxy call in the live path; **`water_level` is provably independent of storage** across 8 storage magnitudes; the missing physics is documented |
| Live payload provenance (7) | every live forecast is `DEMONSTRATION_ONLY` / `is_simulated` / `validated_metrics_apply=False`; never `VALIDATED`; simulated vs synthetic features labelled; every feature carries unit + provenance; `inflow`/`live_storage`/`total_outflow` match the authoritative network state; twin payload exposes provenance |
| Warm-up (3) | no fabricated history at construction; `WARMUP` before 7 distinct steps; repeated identical steps do **not** satisfy the window |
| Strict mode (1) | with placeholders disabled → `FORECAST_UNAVAILABLE` naming `water_level` + `rainfall` and the missing physics |
| GNN separation (3) | control policy is `lstm_primary` only; control forecasts are `LSTM_V3`; LSTM path independent of GNN availability |
| Buffer semantics (2) | buffer holds only the 3 simulated features; step indices are distinct |

---

## 9. Complete Test Results (Req. 23)

| Suite | Result |
|---|---|
| **Complete test suite** | ✅ **292 passed, 0 failed** (58.7 s) |
| Stage 3 + 4 + 5 regression run | ✅ **107 passed, 0 failed** |
| ↳ Stage 3 authoritative-network | ✅ 28/28 |
| ↳ Stage 4 single authoritative simulation | ✅ 41/41 |
| ↳ Stage 5 forecast provenance | ✅ 38/38 |
| Frozen artifact verification | ✅ exit 0 |

Suite growth: 254 (end of Stage 4) → **292** (Stage 5, +38).

---

## 10. Frozen Artifact Verification (Req. 2, 23)

`scripts/stage3_verify_frozen_artifacts.py` → **ALL FROZEN ARTIFACTS UNCHANGED** (exit 0)

| Artifact | raw | LF-normalised | Status |
|---|---|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | MATCH | — | ✅ |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | MATCH | MATCH | ✅ |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | CRLF only | **MATCH** | ✅ |

Additionally, `test_frozen_model_and_scalers_unchanged_by_loading_and_inference`
hashes the four artefacts around a load + inference cycle and asserts no byte changed.

### 10.1 Protected Phase 15.3 artefacts (Req. 24)

✅ All 6 files in `results/phase15_v3_validation/` are **byte-for-byte unchanged**
(compared against the Stage 3 baseline hashes). Nothing in Stage 5 writes there;
the Phase 15.3 scripts were not executed.

---

## 11. Stage 3 / Stage 4 Regression Results (Req. 23)

| Regression | Result |
|---|---|
| Stage 3 authoritative topology preserved (A→B→C→D, delays 2/1/1, attenuation 0.90/0.85/0.80) | ✅ 28/28 |
| Stage 4 single authoritative simulation, WS from the same instance, frontend cannot inject state | ✅ 41/41 |
| Frozen `ReservoirNetwork` physics | ✅ untouched |

---

## 12. Scientific-Claim Impact

| Claim | Before Stage 5 | After Stage 5 |
|---|---|---|
| "Live forecasts are LSTM V3 predictions" | implied as equivalent to the validated evaluation | now explicitly `DEMONSTRATION_ONLY`, `validated_metrics_apply: false` |
| `water_level` input | silently a storage **percentage** where **metres** were required | marked `UNAVAILABLE` / `SYNTHETIC_DEMO`; no false physical meaning |
| `rainfall` input | silently `0.0` | marked `UNAVAILABLE` / `SYNTHETIC_DEMO` |
| 7-day history | 7 copies of `t=0` at start | requires 7 genuinely distinct simulated steps; otherwise `WARMUP` |
| Fallback forecasts | echoed commanded inflow, labelled as model output | removed; explicit `FORECAST_UNAVAILABLE` |
| Reported V3 metrics (MAE/RMSE/R²) | could be read as describing the live numbers | explicitly scoped to the historical held-out evaluation only |

**Net effect:** the live demonstration is now *honest about being a demonstration*.
No validated claim is weakened, and no previously-implied false claim survives.

---

## 13. Remaining Issues

1. **`water_level` (metres) is still not available live.** The demo uses a labelled
   placeholder. A real integration needs a per-reservoir elevation–storage (rating)
   curve — this is a data-acquisition task, not a code task. **Needs your decision (§0).**
2. **`rainfall` (mm) is still not available live.** Same situation; needs a
   meteorological feed or an explicitly modelled rainfall input.
3. **Scale mismatch between virtual and real reservoirs.** The virtual cascades
   (A = 10.82, B = 21.26, C = 348.29, D = 401.43 MCM) are far smaller than the real
   reservoirs whose historical `live_storage` trained the model (Idukki alone up to
   1845 MCM). Units are correct, but the operating point is not representative — a
   further reason the live forecast cannot claim validated skill.
4. **GNN inputs were corrected** (now scaled per its documented contract). The GNN
   remains an unvalidated negative result and is advisory only.
5. **`configs/simulation/four_reservoir_demo.json`** still carries a dead routing-delay
   block on the live path (carried over from Stage 3); untouched per Req. 13.
6. **`data_bridge.py`** still loads its own `LiveForecaster` for the Streamlit
   analytics view; that path is read-only but does not yet carry the provenance
   block. Separate from the Digital Twin live path.
7. No manual browser verification was performed.

---

## 14. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Synthetic placeholders could be mistaken for real measurements if a UI shows them without the provenance label | Medium | Provenance is in the payload; a UI badge is a Stage 6 candidate. Strict mode (`AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS=0`) removes them entirely. |
| 2 | Live forecasts now show `WARMUP` for the first 7 steps instead of instant numbers | Low | Honest by design; the twin renders null forecasts. |
| 3 | `water_level` placeholders sit inside the training distribution, so forecasts *look* plausible — plausibility is not validity | Medium | `validated_metrics_apply: false` on every payload; documented in §12. |
| 4 | Live forecast values changed noticeably (no more percent-in-metres-slot) | Low | Expected: the previous numbers were computed from a physically meaningless input. |
| 5 | `feature_scaler.pkl` is not in the frozen manifest; only its column contract is pinned | Low | The loader verifies the 51-column contract; a byte hash could be added to the manifest in a later stage. |
| 6 | `sklearn` 1.9.0 → 1.9.1 unpickle warning on the frozen scalers | Low | Pre-existing; scalers are read-only and the contract is verified. |

---

## 15. Recommendation

1. **Decide §0** — confirm the labelled `SYNTHETIC_DEMO` placeholder for the
   demonstration twin, or require `FORECAST_UNAVAILABLE` until real
   `water_level` / `rainfall` sources exist. Everything else in Stage 5 is
   independent of this choice.
2. **Accept Stage 5.** The live forecast pipeline is now explicit, unit-correct,
   provenance-aware and honest about the two physical quantities it cannot produce.
3. **Before any external demonstration**, show the forecast status alongside the
   numbers (a small badge reading "DEMONSTRATION — model inputs simulated") so no
   reader mistakes the live numbers for the validated evaluation.
4. **Stage 6 candidates (in priority order)**
   * a UI provenance badge sourced from `forecast_provenance`;
   * an elevation–storage curve for the four mapped reservoirs (data acquisition),
     which would make `water_level` legitimately available;
   * a rainfall input (observed feed or an explicitly modelled series);
   * extend `scripts/stage3_verify_frozen_artifacts.py` to also hash
     `data/processed/scaled/feature_scaler.pkl`.
5. **Do NOT** integrate MPC (Stage 7) or the SafetyLayer (Stage 8) — neither was
   touched, and GNN output is still kept out of the control path.

---

*End of Stage 5 Report. Stopped at the Stage 5 boundary — Stage 6 not started.*
