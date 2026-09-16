# STAGE 7 REPORT — LIVE MPC INTEGRATION

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 7 boundary (Stage 8 NOT started)
**Goal:** Integrate the validated Phase 15.3 MPC into the live authoritative simulation path, behind a provenance gate.

---

## 1. Exact Files Changed

### Created

| File | Purpose |
|---|---|
| `src/controller/live_mpc_orchestrator.py` | **NEW** — the provenance gate + the ONE authoritative live controller path (`LiveMPCOrchestrator`, `LiveControlDecision`, `ControllerStatus`) |
| `tests/test_stage7_live_mpc_integration.py` | **NEW** — 40 Stage 7 tests |

### Modified

| File | Change |
|---|---|
| `src/dashboard/api/state_manager.py` | AI mode now uses `LiveMPCOrchestrator` instead of `ForecastAwareController`; controller provenance published in the state payload |
| `src/dashboard/twin_component/state_adapter.py` | Top-level `control` block; never invents a favourable controller status |
| `src/dashboard/web/index.html` | HUD shows the ACTUAL controller status, forecast eligibility and safety-layer status |
| `src/dashboard/sim_bridge.py` | `compute_ai_recommendation` documented as **NON-AUTHORITATIVE** |
| `src/network_env/live_forecast_adapter.py` | Additive: `node_id_map` / `for_network()`, `parse_provenance_string()`, `provenance_from_snapshot()` |
| `src/network_env/__init__.py` | Export the new adapter helpers |

### NOT modified (hard boundary respected)

`src/controller/mpc_controller.py` · `src/controller/safety.py` · `src/controller/objective.py` · `src/network_env/reservoir_network.py` · `src/network_env/v3_forecast_adapter.py` · `models/lstm_pytorch_v3_logtarget/*` · `data/processed/scaled/feature_scaler.pkl` · `results/lstm_pytorch_v3_logtarget/*` · `results/phase15_v3_validation/*`

**No second MPC implementation was created.** The orchestrator holds one
`MPCController` and calls `decide()` unchanged.

---

## 2. Live Control Path Before / After

### Before Stage 7

```
Frozen LSTM V3 → live forecast dict
                        ↓
        ForecastAwareController            ← RULE-BASED (risk thresholds)
        SimBridge.compute_ai_recommendation
                        ↓  percent gates, A/B/C only
        D kept at the manual 100% value
                        ↓
                ReservoirNetwork → WebSocket → Twin
```
The AI-mode gate decisions were made by a threshold rule engine, yet the UI
labelled the mode "AI / MPC". There was **no provenance gate**: any forecast —
including a fabricated echo — would have driven it.

### After Stage 7

```
Frozen LSTM V3
     ↓
LiveForecastAdapter                        (Stage 6, unmodified contract)
     ↓
NetworkForecastSnapshot
     ↓
PROVENANCE GATE   (validated_metrics_apply == true for EVERY reservoir)   ← NEW
     ↓
MPCController.decide()                     (validated Phase 15.3, UNMODIFIED)
     ↓
[ Stage 8 SafetyLayer ]                    ← NOT INTEGRATED (explicitly reported)
     ↓
ReservoirNetwork → FastAPI → WebSocket → Three.js Twin
```

---

## 3. Proof the MPC Is Actually Authoritative

| Evidence | Test |
|---|---|
| The orchestrator's gate positions equal a **direct** `MPCController.decide()` call, to 1e-12, for all four reservoirs — i.e. the boundary does not post-process the decision | `test_orchestrator_output_equals_a_direct_mpc_call` |
| With an eligible forecast, `controller_status = ACTIVE`, `mpc_status ∈ {OPTIMAL,…}`, `candidates_evaluated > 0`, `control_applied = True` | `test_validated_forecast_reaches_the_mpc` |
| **The gate positions actually applied to the live network are the MPC's**, and they differ from the pre-step gates | `test_mpc_actually_changes_live_gate_decisions` |
| The old rule-based advisor is never consulted in AI mode (its call raises if touched) | `test_live_ai_step_does_not_use_the_legacy_advisor` |
| `compute_ai_recommendation` no longer appears in the live state manager at all | `test_legacy_forecast_aware_controller_is_not_called_by_the_live_path` |
| The GNN cannot make control decisions (nothing GNN-related is imported) | `test_gnn_cannot_make_control_decisions` |
| The controller provenance reaches `/api/state`, the twin payload and the HUD | `test_control_provenance_is_in_the_api_state`, `test_control_provenance_is_in_the_twin_payload`, `test_twin_ui_displays_the_controller_state` |

**One authoritative live controller path.** `controller_type = "MPC"`,
`controller_status = ACTIVE | BLOCKED | UNAVAILABLE`. A rule-based controller is
never labelled as MPC; a blocked MPC is never labelled ACTIVE.

---

## 4. Proof Demonstration Forecasts Cannot Drive the MPC

The gate requires, **for every controlled reservoir**:

1. the node exists in the snapshot,
2. `validated_metrics_apply is True` (exact identity, never truthy-coerced),
3. declared status is `VALIDATED` (not `DEMONSTRATION_ONLY`, not warm-up, not unavailable),
4. declared unit is `MCM/day`,
5. every horizon the MPC consumes (`1d`, `3d`, `7d`) is available, finite and non-negative.

On failure the orchestrator **does not call the MPC at all**
(`mpc_status = "NOT_INVOKED"`), reports the exact reasons, holds the REAL current
gates and sets `control_applied = False`.

| Case | Result | Test |
|---|---|---|
| `DEMONSTRATION_ONLY` | BLOCKED, `NOT_INVOKED`, reasons include `NOT_VALIDATED_METRICS` + `STATUS_NOT_VALIDATED:DEMONSTRATION_ONLY` | `test_demonstration_only_forecast_blocks_the_mpc` |
| `validated_metrics_apply=false` (even with status `VALIDATED`) | BLOCKED | `test_validated_metrics_apply_false_blocks_the_mpc` |
| Warm-up / unavailable | BLOCKED | `test_warmup_and_unavailable_statuses_block_the_mpc` |
| Missing Idukki | BLOCKED (§5) | `test_missing_idukki_forecast_blocks_and_is_not_fabricated` |
| NaN / ±Inf | BLOCKED (`HORIZON_UNAVAILABLE` / `INVALID_VALUE`) | `test_nonfinite_forecast_is_rejected`, `test_nonfinite_value_in_a_bare_snapshot_is_rejected` |
| Unit mismatch (`m3/s`) | BLOCKED (`UNIT_MISMATCH`) — both at the adapter and, independently, from the snapshot's provenance string | `test_unit_mismatch_is_rejected`, `test_unit_mismatch_declared_in_provenance_is_rejected` |
| No snapshot at all | BLOCKED (`NO_FORECAST_SNAPSHOT`) — fails closed | `test_eligibility_is_fail_closed_without_any_snapshot` |
| Blocked decision | gate positions equal the **real current** gate positions; nothing invented | `test_blocked_decision_holds_current_gates_without_fabricating` |

**Consequence for the current live twin:** because Stage 5 forecasts are
`DEMONSTRATION_ONLY`, the live AI path is **BLOCKED by design** — pinned by
`test_live_ai_mode_with_demo_forecasts_blocks_and_holds_gates`. Demonstration
forecasts remain available for visualisation, and can never silently become
authoritative control input.

---

## 5. Treatment of the Missing Idukki (Reservoir D) Forecast

* Reservoir D is a **full, first-class member** of the MPC state — it is not
  pinned, excluded, or held at a manual value. With an eligible forecast the MPC
  returns a D gate like any other reservoir
  (`test_all_four_reservoirs_reach_the_mpc_state`, `test_mpc_actually_changes_live_gate_decisions`).
* The live pipeline produces **no forecast for D**, and none is invented. The
  Stage 6 adapter emits an explicit `ReservoirForecast` for D with all three
  targets `None` and status `FORECAST_UNAVAILABLE` (`issue = MISSING_FORECAST`).
* Because the gate requires every reservoir to be eligible, a missing D forecast
  **blocks the whole MPC** — it does not cause a zero, an average, a
  carry-forward, or a partial "control the other three" fallback.
* Reasons read e.g. `Virtual Reservoir D:MISSING_FORECAST`,
  `Virtual Reservoir D:NOT_VALIDATED_METRICS`, `Virtual Reservoir D:HORIZON_UNAVAILABLE:1d`.

This is the fail-explicit behaviour requested: no fabricated forecast, no silent
degradation.

---

## 6. Test Counts

| Suite | Result |
|---|---|
| **Stage 7 tests** | ✅ **40 passed** |
| **Complete test suite** | ✅ **388 passed, 0 failed** (49.6 s) |
| Stage 3 regression | ✅ 28/28 |
| Stage 4 regression | ✅ 41/41 |
| Stage 5 regression | ✅ 38/38 |
| Stage 6 regression | ✅ 56/56 |
| Stage 3+4+5+6+7 combined | ✅ **203 passed, 0 failed** |

Suite growth: 348 (end of Stage 6) → **388** (Stage 7, +40).

Stage 7 coverage maps to the 15 requested items:

| Req. | Test |
|---|---|
| 1 Validated → MPC receives snapshot | `test_validated_forecast_reaches_the_mpc` |
| 2 DEMONSTRATION_ONLY → blocked | `test_demonstration_only_forecast_blocks_the_mpc` |
| 3 `validated_metrics_apply=false` → blocked | `test_validated_metrics_apply_false_blocks_the_mpc` |
| 4 Missing Idukki → no fabrication | `test_missing_idukki_forecast_blocks_and_is_not_fabricated` |
| 5 Non-finite → rejected | `test_nonfinite_forecast_is_rejected`, `test_nonfinite_value_in_a_bare_snapshot_is_rejected` |
| 6 Unit mismatch → rejected | `test_unit_mismatch_is_rejected`, `test_unit_mismatch_declared_in_provenance_is_rejected` |
| 7 MPC changes live gates | `test_mpc_actually_changes_live_gate_decisions` |
| 8 Legacy advisor not authoritative | `test_legacy_forecast_aware_controller_is_not_called_by_the_live_path`, `test_live_ai_step_does_not_use_the_legacy_advisor`, `test_sim_bridge_advisor_is_marked_non_authoritative` |
| 9 Browser cannot inject state | `test_no_state_write_endpoint_exists`, `test_forged_control_block_is_ignored_by_command_endpoints` |
| 10 All four reservoirs reach the MPC | `test_all_four_reservoirs_reach_the_mpc_state` |
| 11 Canonical units | `test_canonical_units_are_preserved`, `test_no_new_implicit_unit_conversions_are_introduced` |
| 12 Provenance reaches WS/UI | `test_control_provenance_is_in_the_api_state`, `…twin_payload`, `test_twin_ui_displays_the_controller_state`, `test_twin_payload_never_labels_an_unknown_controller_as_active_mpc` |
| 13 Frozen artifacts untouched | `test_mpc_integration_does_not_modify_frozen_artifacts`, `test_validated_mpc_and_safety_files_are_unmodified` |
| 14 Stage 3–6 intact | `test_stage3_topology_is_unchanged`, `test_demonstration_badge_and_strict_mode_still_present` |
| 15 Validated snapshot passes through unchanged | `test_validated_snapshot_values_pass_through_unchanged` |

Plus the requested behavioural test:
**`test_same_state_and_validated_forecast_gives_the_same_mpc_decision`** — two
independent identical networks + identical validated forecasts + fresh
controllers produce **byte-identical gate positions** (deterministic).

---

## 7. Frozen-Artifact Status

`scripts/stage3_verify_frozen_artifacts.py` → **exit 0 — ALL FROZEN ARTIFACTS UNCHANGED**

| Artifact | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | ✅ UNCHANGED |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | ✅ UNCHANGED |
| `data/processed/scaled/feature_scaler.pkl` | ✅ read-only |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | ✅ UNCHANGED (LF-normalised match) |

Additionally hashed around a full orchestrator run inside the test suite.

### 7.1 Phase 15.3 regression verification

`scripts/stage3_phase15_3_reproduction.py` re-run (shielded, redirected output) → **exit 0**:

* **Protected artifacts: 6/6 UNCHANGED** (before/after hash comparison)
* **Reproduction fidelity: 3/3 files byte-for-byte IDENTICAL** to the protected run
* Reproduced metrics identical: overflow events 7 → 0, overflow volume 10.75 → 0 MCM,
  downstream violations 8 → 0, peak downstream flow 60 → 30, total release 2017.25 → 2031.00,
  forecast dates used 74, mass-balance residual −6.82e−13

---

## 8. Remaining Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **The live AI path is BLOCKED in the current build**, because all live forecasts are `DEMONSTRATION_ONLY`. This is the intended safety outcome, not a defect — but it means the live twin cannot demonstrate MPC-driven release until real telemetry exists. | Expected; documented. |
| 2 | Reservoir D (Idukki) has no live forecast, which independently blocks the MPC. | Same root cause as #1. |
| 3 | The MPC's **internal** `SafetyLayer` (gate bounds + rate limiting) is part of the validated artifact and still runs inside `decide()`. The **Stage 8 post-MPC safety boundary on the live path does not exist.** | Stated explicitly as `safety_layer_status = NOT_INTEGRATED_STAGE_7`. |
| 4 | Live MPC performance is untested at scale: the E2E test uses one step. No closed-loop soak test over many steps exists. | Stage 8/9 candidate. |
| 5 | `get_adapted_state()` still recomputes the ML pipeline on every read, including every WebSocket broadcast (pre-existing since Stage 4). Adding MPC to the AI path makes each `step()` heavier. | Performance risk; caching is a good Stage 8 candidate. |
| 6 | `SimBridge.compute_ai_recommendation` still exists (retained as an offline helper, marked NON-AUTHORITATIVE). It is no longer called by the live path, but it remains available. | Low; pinned by tests. |
| 7 | The `node_id_map`/`for_network()` addition to the Stage 6 adapter exists because the live facade network uses live names as node ids while the canonical validated network uses `Reservoir_A…D`. This dual convention is a documented wart inherited from Stage 3. | Low; explicit and tested. |
| 8 | The MPC grid search is 1296 candidates × 3 rollout steps per decision. Fine for a demo, expensive for high-frequency control. | Performance only. |

---

## 9. Statement on the Safety Layer

> **The SafetyLayer is NOT integrated in Stage 7.**
>
> The post-MPC safety boundary on the live path does not exist yet. The live
> architecture is `Forecast → MPC → ReservoirNetwork`, with a clearly marked
> future boundary at `Forecast → MPC → [Stage 8 SafetyLayer] → ReservoirNetwork`.
>
> Every control payload reports `safety_layer_status = "NOT_INTEGRATED_STAGE_7"`,
> and the Digital Twin HUD displays `SAFETY LAYER — NOT INTEGRATED`.
>
> The `SafetyLayer` instance that the validated MPC constructs internally is part
> of the frozen Phase 15.3 artifact and is unchanged; it is **not** the Stage 8
> live safety boundary, and this report does not present the MPC as
> safe-by-construction on the live path.

---

## 10. Requirement Compliance

| # | Requirement | Status |
|---|---|---|
| 1–4 | Do not modify MPC / SafetyLayer / ReservoirNetwork / frozen artifacts | ✅ verified by hash + source assertions |
| — | Use the existing MPC; no second implementation | ✅ `MPCController` instantiated once and called as-is |
| — | MPC receives exactly `NetworkForecastSnapshot` | ✅ |
| — | All four reservoirs in the MPC state; D not pinned/excluded | ✅ |
| — | Fail explicitly when a required forecast is unavailable | ✅ BLOCKED + reasons |
| — | Canonical units (MCM, MCM/day, fraction); percent only at the API/UI | ✅ single boundary in `src/common/units.py` |
| — | ONE authoritative live controller path | ✅ `LiveMPCOrchestrator`; legacy advisor removed from the live path |
| — | Browser JS / GNN cannot decide or inject state | ✅ |
| — | SafetyLayer NOT integrated; boundary marked | ✅ §9 |
| — | Control provenance exposed | ✅ `controller_type`, `controller_status`, `forecast_control_eligible`, `forecast_provenance`, `safety_layer_status` |
| — | Do not label a rule-based controller as MPC | ✅ |
| — | Do not start Stage 8 / hardware / RL / change the GNN role / redesign the MPC | ✅ |

---

## 11. Recommendation

1. **Accept Stage 7.** The validated MPC is now the single authoritative live
   controller, gated on forecast provenance, with D as a full member and no
   fabricated inputs anywhere.
2. **Expect the live AI path to be BLOCKED** until real `water_level` (metres) and
   `rainfall` (mm) sources exist, i.e. until the Stage 5 hardware-readiness item
   lands. The gate is doing its job. For a demonstration of MPC-driven control,
   use the offline validated pipeline (Phase 15.3) rather than the live twin.
3. **Stage 8 priorities**
   * integrate the SafetyLayer as the post-MPC live boundary and switch
     `safety_layer_status` from `NOT_INTEGRATED_STAGE_7` to its real status;
   * cache the ML pipeline so `get_adapted_state()` is not recomputed per broadcast;
   * add a multi-step closed-loop soak test for the live MPC path.
4. **Do NOT** implement hardware, RL/MARL, or change the GNN role.

---

*End of Stage 7 Report. Stopped at the Stage 7 boundary — Stage 8 not started.*
