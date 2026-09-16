# STAGE 8 REPORT — SAFETY LAYER INTEGRATION

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 8 boundary (Stage 9 NOT started)
**Goal:** Integrate the validated `SafetyLayer` into the live MPC control path so that
it is the final software safety boundary between the MPC's proposed action and the
authoritative reservoir simulation.

Target architecture (achieved):

```
Frozen LSTM V3
    ↓
LiveForecastAdapter
    ↓
NetworkForecastSnapshot
    ↓
PROVENANCE GATE
    ↓
MPCController
    ↓
SAFETY LAYER          ← this stage
    ↓
ReservoirNetwork
    ↓
FastAPI / WebSocket
    ↓
Three.js Digital Twin
```

---

## 1. Exact Files Changed

### Created

| File | Purpose |
|---|---|
| `tests/test_stage8_safety_layer_integration.py` | **NEW** — 38 Stage 8 tests (36 functions; one parametrized ×3) |
| `results/phase15_stage8_safety_layer/PHASE_15_STAGE8_REPORT.md` | **NEW** — this report |

### Modified

| File | Change |
|---|---|
| `src/controller/live_mpc_orchestrator.py` | **The integration.** `SafetyLayer` is now instantiated and invoked inside `decide()`; `LiveControlDecision` gains `safety_layer_integrated`, `safety_layer_version`, `safety_layer_status`, `safety_is_safe`, `safety_violations`, `safety_modified`, `proposed_gate_positions_fraction`; `status_dict()` reports them; module doc records the layer's ACTUAL guarantees. |
| `src/dashboard/api/state_manager.py` | **Comment/doc only** — the stale "SafetyLayer is NOT integrated yet (Stage 8)" note replaced with an accurate Stage 8 statement, and `_apply_ai_control()` documents that `decision.gate_positions_pct` is the SafetyLayer's output. **No behavioural change.** |
| `src/dashboard/twin_component/state_adapter.py` | `_control_block()` now carries `safety_modified` (default `None` = *unknown*, never `False`) and `proposed_gate_positions_fraction`; docstrings corrected. |
| `src/dashboard/web/index.html` | Two new HUD rows — `SAFETY LAYER` (`data-ref="safety"`) and `SAFETY ACTION` (`data-ref="safetymod"`); `_updateHUD` colours `SAFE` green / `CORRECTED` amber and shows `MODIFIED` / `UNCHANGED` / `NOT APPLIED`. Display only, no logic. |
| `src/dashboard/app.py` | Streamlit panel (read-only) shows `Safety layer` status and a caption when the layer MODIFIED the proposal. Cannot write back. |
| `tests/test_stage7_live_mpc_integration.py` | 4 assertions updated from the Stage 7 placeholder contract (`NOT_INTEGRATED_STAGE_7`) to the real Stage 8 contract. No Stage 7 behaviour asserts were weakened. |

### NOT modified (hard boundary respected)

`src/controller/mpc_controller.py` · `src/controller/safety.py` · `src/controller/objective.py` ·
`src/network_env/reservoir_network.py` · `src/network_env/v3_forecast_adapter.py` ·
`models/lstm_pytorch_v3_logtarget/*` · `data/processed/scaled/feature_scaler.pkl` ·
`results/lstm_pytorch_v3_logtarget/*` · `results/phase15_v3_validation/*` · `src/hardware/*`

* **No second safety system was created.** The orchestrator holds exactly one
  `SafetyLayer` instance — the validated `src/controller/safety.py` class.
* **No MPC redesign.** `MPCController.decide()` is called unchanged: same objective,
  same candidate grid, same optimization logic.

---

## 2. Exact Live Path Before / After

### Before Stage 8

```
Frozen LSTM V3 → LiveForecastAdapter → NetworkForecastSnapshot
                                             ↓
                                    PROVENANCE GATE
                                             ↓
                                    MPCController.decide()
                                    (which internally ran SafetyLayer
                                     once, but nothing verified the object
                                     that actually reached the network)
                                             ↓
                            decision.gate_positions_pct  ────────┐
                                                                  ↓
                                                         ReservoirNetwork
                                                                  ↓
                                                    state → WebSocket → Twin
```
The MPC's raw output was written straight into `gate_commands`. The boundary
existed only inside the MPC, and the live payload reported
`safety_layer_status = "NOT_INTEGRATED_STAGE_7"`.

### After Stage 8

```
Frozen LSTM V3 → LiveForecastAdapter → NetworkForecastSnapshot
                                             ↓
                                    PROVENANCE GATE        (Stage 7, unchanged)
                                             ↓
                                    MPCController.decide() (validated Phase 15.3, UNMODIFIED)
                                             ↓  raw proposal
                                    SafetyLayer.validate() (validated Phase 15.3, UNMODIFIED) ← NEW BOUNDARY
                                             ↓  validated_gates  ← the ONLY thing applied
                                     decision.gate_positions_pct / _fraction
                                             ↓
                                     ReservoirNetwork
                                             ↓
                                     state → FastAPI → WebSocket → Three.js Twin
```

The exact orchestrator code that implements the boundary:

```python
proposed_fraction = {str(k): v for k, v in (mpc_decision.gate_positions or {}).items()}
t0 = time.perf_counter()
safety_result = self.safety.validate(proposed_fraction, current_fraction, node_ids)
self.last_safety_latency_ms = (time.perf_counter() - t0) * 1000.0

applied_fraction = {nid: float(safety_result.validated_gates[nid]) for nid in node_ids}
applied_pct = {nid: units.gate_fraction_to_percent(g) for nid, g in applied_fraction.items()}
```

`applied_fraction` — **the SafetyLayer's output** — is what populates
`gate_positions_fraction` / `gate_positions_pct`, and those are the only values
`state_manager._apply_ai_control()` writes into `gate_commands`.

The MPC's raw proposal is kept **for audit only**, in
`proposed_gate_positions_fraction`, kept transport-safe (finite floats or `None`)
and deliberately never routed through the unit boundary.

---

## 3. Exact SafetyLayer Checks Actually Present

Read from `src/controller/safety.py` (**not assumed**). `SafetyLayer.validate()`
receives `(proposed_gates, current_gates, node_ids)` and returns
`SafetyCheckResult(is_safe, validated_gates, violations, status)`.

### 3.1 Checks that ACTUALLY exist

| # | Check | Implemented? | Exact behaviour | Stage 8 evidence |
|---|---|---|---|---|
| 1 | **All required nodes present** | ✅ YES | A node absent from the proposal is replaced with its **current** gate: `current_gates.get(nid, 0.1)`. Violation: `"<nid>: missing gate command — using fallback"` | `test_missing_node_in_the_proposal_is_filled_from_the_current_gate` (D=0.25 current → 0.25 applied, proposal recorded as `None`) |
| 2 | **Finite numeric values / NaN-Inf rejection** | ✅ YES | Non-`int`/`float`, `NaN` or `±Inf` ⇒ value replaced by **0.1** (not clamped, not passed through). Violation: `"<nid>: invalid gate value <v> — clamped to 0.1"` | `test_non_finite_gate_is_replaced_by_the_conservative_value` (NaN, +Inf, −Inf parametrized), `test_non_numeric_gate_is_replaced` (`"wide open"`) |
| 3 | **Gate bounds [0, 1]** | ✅ YES | Finite gate clamped with `max(0.0, min(1.0, gate))`. Violation: `"<nid>: gate <o> clamped to <g>"` | `test_gate_bounds_are_enforced` |
| 4 | **Rate-of-change limit** | ✅ YES | `max_gate_change_per_step` (default **0.5**, and the live orchestrator passes the MPC's own `config.max_gate_change`). `abs(delta) > limit` ⇒ truncated to `prev ± limit`, then re-clamped. Violation: `"<nid>: gate movement <d> exceeds limit <l> — rate-limited to <g>"` | `test_rate_limit_is_enforced` |
| 5 | **Violations reported** | ✅ YES | Human-readable strings accumulated in `violations`; propagated verbatim into `LiveControlDecision.safety_violations` and the twin payload | every corrected-case test |
| 6 | **Safe action selection** | ⚠️ PARTIAL | The layer does **not** search for a safe action. It performs **per-node correction**: fill-from-current → 0.1 for invalid → clamp → rate-limit. That is the whole mechanism. | `test_safety_layer_output_equals_a_direct_validate_call` |
| 7 | **Emergency / fallback behaviour** | ⚠️ EXISTS BUT NOT REACHABLE FROM `validate()` | `SafetyLayer.emergency_fallback(node_ids)` returns `{nid: 0.1}`. It is a `@staticmethod` and is called by `MPCController` **only** when no feasible candidate exists. `validate()` never calls it, and its `else: status = "EMERGENCY"` branch is **unreachable through `validate()`**, because every `nid in node_ids` always ends up in `validated` (either the fallback path or the main path). | `test_emergency_status_is_unreachable_through_validate`, `test_safety_layer_source_is_unmodified` |

Status mapping inside `validate()`:

* `len(violations) == 0` → `"SAFE"`, `is_safe = True`
* violations present and all nodes validated → `"CORRECTED"`, `is_safe = False`
* otherwise → `"EMERGENCY"` (unreachable via `validate()`, see above)

### 3.2 Checks that are NOT implemented — REPORTED, NOT ADDED

The Stage 8 instruction was explicit: *"If the existing SafetyLayer does not handle
a condition safely, STOP and report the gap rather than silently adding unrelated
behavior."* The following were **not** implemented in the validated layer, and
**nothing was added to fill them**:

| Check requested in the brief | Present? | Notes |
|---|---|---|
| Gate bounds | ✅ | see 3.1 #3 |
| NaN / Inf rejection | ✅ | see 3.1 #2 |
| Rate-of-change limits | ✅ | see 3.1 #4 |
| Invalid action handling | ⚠️ | A *value* that is invalid is handled (→ 0.1). A **malformed container** (e.g. `proposed_gates = None` or a list) is **not** handled — the layer assumes a `Mapping` and would raise `TypeError` on `nid not in proposed_gates`. The orchestrator guarantees a `dict`, so this cannot happen on the live path; it is a gap in the layer itself. |
| **Release bounds** | ❌ **NOT IMPLEMENTED** | No release/max-release/flow check exists anywhere in `safety.py`. Test `test_release_and_downstream_capacity_checks_are_not_implemented` pins its absence (`"max_release" not in source`). |
| **Downstream capacity constraints** | ❌ **NOT IMPLEMENTED** | The **class docstring claims** *"4. Downstream capacity (estimated from proposed releases)"* — **there is no code backing it**. This false claim is pinned by `test_release_and_downstream_capacity_checks_are_not_implemented` (`assert "Downstream capacity" in source`). Downstream capacity *is* enforced elsewhere — by `objective.evaluate_trajectory(...)` inside the MPC's scoring and by `ReservoirNetwork`'s own downstream warnings — but **not** by the SafetyLayer. |
| **Reservoir safety constraints (storage / headroom / overflow)** | ❌ **NOT IMPLEMENTED** | `"storage" not in source`. The layer is purely gate-level; it cannot see storage. |
| **Minimum environmental flow** | ❌ **NOT IMPLEMENTED** | No such concept in the layer. |
| **Prior-state validation** | ❌ **NOT IMPLEMENTED** | `current_gates` is used *only* for the rate limit and the missing-node fallback; it is never validated for plausibility. Edge case: `prev = current_gates.get(nid, gate)` means **if the current gate for a node is missing, no rate limiting is applied at all** for that node. |
| **Impossible-state detection** | ❌ **NOT IMPLEMENTED** | No storage/flow/level sanity check exists. |

**Consequence, stated plainly:** the live SafetyLayer guarantees that every gate
command reaching `ReservoirNetwork` is a finite number in `[0, 1]`, present for all
four reservoirs, and within `max_gate_change` of the current gate. It does **not**
guarantee release bounds, downstream-capacity compliance, storage safety, or
minimum flow. Those remain the MPC objective's and `ReservoirNetwork`'s
responsibility, exactly as in the validated Phase 15.3 pipeline. No claim to the
contrary is made anywhere in the Stage 8 code, UI, or this report.

---

## 4. Proof the Raw MPC Output Cannot Bypass the SafetyLayer

| # | Evidence | Test |
|---|---|---|
| 1 | **Applied gates == a direct `SafetyLayer.validate()` call on the raw proposal**, to `1e-12`, for all four reservoirs. The boundary does not post-process, re-order or "improve" the layer's output. | `test_safety_layer_output_equals_a_direct_validate_call` |
| 2 | With a monkeypatched **unsafe** MPC proposal, applied ≠ proposed, `safety_layer_status = CORRECTED`, `safety_is_safe = False`, `safety_modified = True`, violations non-empty, and every applied gate is finite and inside `[0, 1]`. | `test_unsafe_mpc_action_is_constrained_by_the_safety_layer` |
| 3 | With an unsafe proposal, the **applied** gates are `{A: 0.5, B: 0.0, C: 0.1, D: 0.5}` for the raw proposal `{A: 5.0, B: −3.0, C: NaN, D: 0.9}` — i.e. A is rate-limited to **0.5** (not 1.0), B is clamped to **0.0** (not negative), C is substituted with **0.1** (not a crash, not NaN), D is rate-limited to **0.5** (not 0.9). | smoke probe + `test_unsafe_mpc_action_is_constrained_by_the_safety_layer` |
| 4 | The SafetyLayer instance used **is** `src/controller/safety.py` — `type(orchestrator.safety) is SafetyLayer` — and it is configured with the same `max_gate_change` as the MPC's own instance. | `test_the_safety_layer_used_is_the_validated_implementation` |
| 5 | The layer is invoked **exactly once** per active decision, with all four live node ids and all four proposed gates. | `test_valid_mpc_action_reaches_the_safety_layer` |
| 6 | **Source-level**: `decision.gate_positions_pct` (the SafetyLayer's output) is what `state_manager` applies; `self.mpc_orchestrator.decide(` is the only decision call; the legacy rule-based advisor is absent; the GNN is never assigned into `gate_commands`; `ai_recommendations` no longer exists. | `test_only_the_orchestrator_applies_gates_to_the_live_network` |
| 7 | The **GNN cannot bypass** the boundary: the orchestrator imports nothing GNN-related (AST-verified), and the GNN result never reaches `gate_commands`. | `test_gnn_cannot_bypass_the_safety_layer` |
| 8 | The **browser cannot bypass** the boundary: `POST`/`PUT /api/state` → **405**; posting a forged `control: {safety_layer_status: "SAFE", control_applied: true}` to a command endpoint does **not** change the reported safety provenance. | `test_browser_cannot_bypass_the_safety_layer`, `test_forged_control_payload_does_not_change_safety_provenance` |
| 9 | The SafetyLayer source does **not** reference the orchestrator (no circularity, no self-modification): `"live_mpc_orchestrator" not in safety.py`. | `test_safety_layer_source_is_unmodified` |

### 4.1 The requested behavioural integration test (not a unit test)

`test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network`
drives the **authoritative** `state_manager.sim_state` end to end:

1. `sim.mode = "MANUAL"`, gates forced to `0.0`, `sim.step()` → network gates are genuinely `0.0`.
2. Switch to AI mode with a **validated** forecast bundle (gate → `ACTIVE`) and monkeypatch
   the MPC to return the unsafe proposal `{A: 5.0, B: −3.0, C: NaN, D: 0.9}`.
3. `sim.step()`.
4. Assertions:
   * the **decision** carries the safe action `{A: 0.5, B: 0.0, C: 0.1, D: 0.5}` (`safety_layer_status = CORRECTED`, `safety_modified = True`);
   * `ReservoirNetwork`'s `state.gate_position` **equals that safe action** for all four reservoirs;
   * `ReservoirNetwork` did **not** receive the raw proposal (A ≠ 1.0, D ≠ 0.9);
   * the resulting physical state **reflects** it: `A.state.controlled_release == 0.5 × A.max_release`.

The mirror-image test `test_demo_forecast_end_to_end_leaves_gates_untouched` proves the
blocked path changes **nothing** at the network.

---

## 5. Proof Demonstration Forecasts Remain Blocked (SafetyLayer Never Rescues Them)

The Stage 7 provenance gate stays **upstream** and untouched. The SafetyLayer is
**never** treated as a way to make an invalid forecast valid — in fact it is never
*invoked* in that case.

| Evidence | Result | Test |
|---|---|---|
| `DEMONSTRATION_ONLY` forecasts | `controller_status = BLOCKED`, `mpc_status = "NOT_INVOKED"`, `safety_layer_status = "NOT_APPLIED_MPC_BLOCKED"`, `control_applied = False`, gates = the **real current** gates | `test_demonstration_forecast_blocks_and_safety_does_not_rescue` |
| SafetyLayer invocation on a blocked forecast | **Zero calls** — a monkeypatched `validate` records `[]` | `test_safety_layer_is_not_invoked_when_the_mpc_is_blocked` |
| End-to-end with the authoritative sim | gates before == gates after; no fabrication | `test_demo_forecast_end_to_end_leaves_gates_untouched` |
| Network-side effect of a blocked forecast | none — SafetyLayer holds the current gates and reports `NOT_APPLIED_MPC_BLOCKED` | `test_blocked_decision_holds_current_gates_without_fabricating` (Stage 7, still green) |

Three explicit "not applied" states exist so a blocked path can never be confused
with a safe one:

* `NOT_APPLIED_MPC_BLOCKED` — the provenance gate refused the forecast
* `NOT_APPLIED_MPC_ERROR` — the MPC raised
* `NOT_APPLIED_ADAPTER_ERROR` — the forecast adapter raised

---

## 6. Treatment of the Missing Idukki (Reservoir D) Forecast

Unchanged from Stage 7, and now verified with the SafetyLayer in place:

* D is a **full, first-class** member of the MPC state and of the safety boundary —
  never pinned, never excluded, never held at a manual value.
* The live pipeline produces **no forecast for D**, and none is invented: the
  Stage 6 adapter emits an explicit `ReservoirForecast` with all three targets
  `None`, status `FORECAST_UNAVAILABLE`, `issue = MISSING_FORECAST`.
* Because the gate requires **every** reservoir to be eligible, a missing D
  forecast **blocks the whole coordinated MPC**. There is no zero, no average, no
  carry-forward, no "control the other three" fallback.
* `safety_layer_status = NOT_APPLIED_MPC_BLOCKED`; the SafetyLayer is not invoked.
* Reasons read `Virtual Reservoir D:MISSING_FORECAST` (plus `…:NOT_VALIDATED_METRICS`,
  `…:HORIZON_UNAVAILABLE:1d`), surfaced in the API/WebSocket/twin payload.

Pinned by `test_missing_idukki_forecast_blocks_upstream` (Stage 8) and
`test_missing_idukki_forecast_blocks_and_is_not_fabricated` (Stage 7).

---

## 7. All-Four-Reservoir Behaviour; Reservoir D Is Not Pinned at 100%

* `A = Anayirankal → B = Ponmudi → C = Idamalayar → D = Idukki`, processing order
  `['Virtual Reservoir A', 'Virtual Reservoir B', 'Virtual Reservoir C', 'Virtual Reservoir D']`.
* Every mapping in the decision — `gate_positions_fraction`, `gate_positions_pct`
  and `proposed_gate_positions_fraction` — carries **exactly the four live node ids**
  (`test_all_four_reservoirs_pass_through_the_safety_layer`).
* **The legacy "D pinned at 100%" behaviour is gone.** From a current gate of `0.0`
  with the MPC proposing `0.9`, D is rate-limited to **0.5** — asserted to be
  numerically **not** `1.0` (`test_reservoir_d_is_not_pinned_to_the_legacy_100_percent`).
* Safety applies to the **complete coordinated action**: the layer is called once
  with all four node ids, so a violation on any single reservoir corrects the same
  proposal that carries D's command.
* `test_only_the_orchestrator_applies_gates_to_the_live_network` proves the old
  rule-based advisor and the GNN cannot write gate commands.

---

## 8. Tests and Counts

| Suite | Result |
|---|---|
| **Stage 8 tests** (`tests/test_stage8_safety_layer_integration.py`) | ✅ **38 passed** |
| Stage 7 tests (4 assertions updated to the Stage 8 contract) | ✅ **40 passed** |
| **Complete test suite** | ✅ **426 passed, 0 failed** (63.7 s) |
| Stage 3 regression | ✅ 28/28 |
| Stage 4 regression | ✅ 41/41 |
| Stage 5 regression | ✅ 38/38 |
| Stage 6 regression | ✅ 56/56 |
| Stage 7 regression | ✅ 40/40 |
| Stage 8 | ✅ 38/38 |
| Stage 3+4+5+6+7+8 combined | ✅ **241 passed, 0 failed** |

Suite growth: 388 (end of Stage 7) → **426** (Stage 8, +38).

### 8.1 Coverage against the 20 requested test areas

| Req. | Requirement | Stage 8 test(s) |
|---|---|---|
| 1 | Valid MPC action reaches SafetyLayer | `test_valid_mpc_action_reaches_the_safety_layer`, `test_the_safety_layer_used_is_the_validated_implementation` |
| 2 | SafetyLayer output, not raw MPC output, reaches `ReservoirNetwork` | `test_safety_layer_output_equals_a_direct_validate_call`, `test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network`, `test_only_the_orchestrator_applies_gates_to_the_live_network` |
| 3 | Safe MPC action passes unchanged | `test_safe_mpc_action_passes_through_unchanged` |
| 4 | Unsafe action constrained per the ACTUAL implementation | `test_unsafe_mpc_action_is_constrained_by_the_safety_layer` |
| 5 | Gate bounds enforced | `test_gate_bounds_are_enforced` |
| 6 | Release bounds **where implemented** | `test_release_and_downstream_capacity_checks_are_not_implemented` — **honest gap record: not implemented** (see §3.2) |
| 7 | NaN/Inf behaviour verified | `test_non_finite_gate_is_replaced_by_the_conservative_value` (×3), `test_non_numeric_gate_is_replaced` |
| 8 | Invalid action behaviour verified | `test_non_numeric_gate_is_replaced`, `test_missing_node_in_the_proposal_is_filled_from_the_current_gate`, `test_emergency_status_is_unreachable_through_validate` |
| 9 | All four reservoirs included | `test_all_four_reservoirs_pass_through_the_safety_layer` |
| 10 | D/Idukki not pinned to legacy 100% | `test_reservoir_d_is_not_pinned_to_the_legacy_100_percent` |
| 11 | Demonstration forecasts remain blocked upstream | `test_demonstration_forecast_blocks_and_safety_does_not_rescue`, `test_safety_layer_is_not_invoked_when_the_mpc_is_blocked`, `test_demo_forecast_end_to_end_leaves_gates_untouched` |
| 12 | Missing Idukki forecast remains blocked upstream | `test_missing_idukki_forecast_blocks_upstream` |
| 13 | Browser cannot bypass SafetyLayer | `test_browser_cannot_bypass_the_safety_layer`, `test_forged_control_payload_does_not_change_safety_provenance` |
| 14 | GNN cannot bypass SafetyLayer | `test_gnn_cannot_bypass_the_safety_layer` |
| 15 | Controller provenance reports MPC + SafetyLayer status accurately | `test_controller_provenance_reports_mpc_and_safety_status`, `test_safety_layer_boundary_is_reported_on_an_active_decision` (Stage 7) |
| 16 | Safety modifications visible in state/WebSocket payload | `test_safety_state_is_visible_in_the_twin_payload`, `test_twin_ui_displays_safety_status_and_applied_vs_proposed`, `test_control_payload_is_json_serialisable_even_for_an_unsafe_proposal` |
| 17 | Canonical units remain correct | `test_canonical_units_are_preserved`, `test_no_implicit_unit_conversions_in_the_orchestrator` |
| 18 | Deterministic same-input/same-state behaviour | `test_whole_path_is_deterministic_with_a_validated_forecast`, `test_safety_layer_is_deterministic_for_the_same_inputs` |
| 19 | Stage 3–7 tests remain green | ✅ 203/203 (28 + 41 + 38 + 56 + 40) |
| 20 | Frozen Phase 15.3 artifacts unchanged | `test_safety_integration_does_not_modify_frozen_artifacts`, `test_validated_mpc_and_safety_are_not_modified_by_stage8` |

Plus the required behavioural integration test:
`test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network` (see §4.1)
and its mirror `test_demo_forecast_end_to_end_leaves_gates_untouched`.

### 8.2 A real bug found and fixed during this stage

Building the *display copy* of the raw MPC proposal called
`units.gate_fraction_to_percent()` on a `NaN` proposal value. The unit boundary
**correctly raises** `UnitContractError` ("NaN/Inf gate values are rejected rather
than clamped, because clamping a NaN would silently fail OPEN") — which crashed
`decide()` and would have taken down the WebSocket feed.

Fixed: the audit copy `proposed_gate_positions_fraction` is now **transport-safe**
(finite float, or `None` when the proposal was missing / non-numeric / non-finite)
and is **not** routed through the unit boundary; the misleading
`proposed_gate_positions_pct` field was removed. Pinned by
`test_control_payload_is_json_serialisable_even_for_an_unsafe_proposal`.

---

## 9. Frozen-Artifact Status

`scripts/stage3_verify_frozen_artifacts.py` → **exit 0 — `VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`**

| Artifact | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | ✅ UNCHANGED (SHA256 `448cb9659a91ea21…`) |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | ✅ UNCHANGED (`63d7325e2ad61bde…`) |
| `data/processed/scaled/feature_scaler.pkl` | ✅ read-only, verification MATCH |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | ✅ UNCHANGED (LF-normalised byte-identical; raw hash differs only by the known `core.autocrlf` artefact) |

Additionally, `test_safety_integration_does_not_modify_frozen_artifacts` hashes the
three frozen files around a full orchestrator run with an **unsafe** proposal, and
`test_validated_mpc_and_safety_are_not_modified_by_stage8` asserts by AST/source that
`mpc_controller.py` and `safety.py` contain no reference to the orchestrator and that
`SafetyLayer` is still the class the MPC uses.

---

## 10. Phase 15.3 Regression Status

`scripts/stage3_phase15_3_reproduction.py` re-run **shielded** (its `OUTPUT_DIR`
redirected, protected directory hash-guarded) → **exit 0**:

* **Protected artifacts: 6/6 UNCHANGED** (before/after SHA256 comparison)
* **Reproduction fidelity: 3/3 files byte-for-byte IDENTICAL** to the protected run
* Reproduced metrics identical to the Phase 15.3 baseline:

| Metric | Baseline | MPC | Diff |
|---|---|---|---|
| overflow_events | 7.00 | 0.00 | −7.00 BETTER |
| overflow_volume_mcm | 10.75 | 0.00 | −10.75 BETTER |
| ds_violations | 8.00 | 0.00 | −8.00 BETTER |
| peak_ds_flow | 60.00 | 30.00 | −30.00 BETTER |
| total_release_mcm | 2017.25 | 2031.00 | +13.75 WORSE |
| forecast_dates_used | N/A | 74 | — |

* Mass-balance residual identical in both runs: **−6.82e−13**
* V3 integrity inside the run: **PASSED** (all three V3 artifacts MATCH)

The validated MPC and the validated SafetyLayer therefore behave **exactly** as
before Stage 8.

---

## 11. Performance / Latency Observations

Per the instruction *"Do not optimize prematurely. First establish correctness and
deterministic behavior. Record whether SafetyLayer introduces meaningful per-step
latency"*, **no optimization was performed**. Measurements taken:

| Measurement | Value |
|---|---|
| In-path SafetyLayer latency, measured inside `decide()` over **20** decisions | **min 0.004 ms · mean 0.00622 ms · max 0.0107 ms** |
| Bare `SafetyLayer.validate()` (5000 calls, all four reservoirs) | **0.002784 ms / call** |
| Whole `LiveMPCOrchestrator.decide()` (1296-candidate grid × 3-step rollout) | **≈ 660 ms** |
| Test assertions | in-path latency `< 5 ms`; bare `validate()` `< 1 ms/call` |

**Conclusion:** the SafetyLayer is ~3 µs per call and **≈ 0.001 %** of the cost of a
single `decide()` call. It introduces **no meaningful per-step latency** — it is
three to five orders of magnitude cheaper than the MPC grid search it guards.

The SafetyLayer is **not** on any hot loop other than `decide()` (once per simulated
step), so no caching or vectorization is warranted. `last_safety_latency_ms` is
retained on the orchestrator for future instrumentation.

---

## 12. Remaining Limitations

| # | Limitation | Impact / note |
|---|---|---|
| 1 | **The SafetyLayer does not check release bounds, downstream capacity, storage headroom or minimum environmental flow.** The class docstring's "downstream capacity" claim is **false** (no backing code). | Reported, **not** silently fixed — per the Stage 8 instruction. These remain the MPC objective's and `ReservoirNetwork`'s responsibility. Any future claim of "safety-layer-enforced downstream capacity" must be preceded by actually implementing it. |
| 2 | `EMERGENCY` is **unreachable through `validate()`**; `emergency_fallback()` is only reached from `MPCController`'s "no feasible candidate" branch (status `EMERGENCY`, all gates → 0.1). | The live payload can therefore only ever report `SAFE`, `CORRECTED`, or one of the three `NOT_APPLIED_*` states — unless the MPC itself returns the emergency proposal, which the boundary then re-validates normally. |
| 3 | **The live AI path is still BLOCKED.** All live forecasts are `DEMONSTRATION_ONLY` (Stage 5) and Reservoir D has no live forecast at all, so the provenance gate refuses. | Intended safety outcome, not a defect. It does mean the live twin cannot currently *demonstrate* MPC-driven release; the SafetyLayer's behaviour on the live path is proven by the behavioural integration test with an injected validated bundle. |
| 4 | The SafetyLayer ends up applied **twice** per step: once inside the validated `MPCController.decide()` (unmodified Phase 15.3 behaviour) and once at the live boundary. | Idempotent on the current validated path (a gate already rate-limited to exactly the limit is not re-triggered), which is why a valid decision reports `SAFE` / `modified = False`. The **boundary pass is the authoritative one**, because it validates the object that actually reaches the network rather than trusting the producer. |
| 5 | `safety_modified` is **`False`** for a normally-validated decision, which can read as "the layer did nothing". | This is accurate: the layer did not need to change anything. `safety_layer_status` (`SAFE`) is the field that states the layer ran and found no violations. |
| 6 | `proposed_gate_positions_fraction` entries may be **`None`** (missing, non-numeric, NaN/Inf proposals). | By design: the raw value is not a valid gate and must not be coerced. Consumers (UI/API clients) must handle `None`. |
| 7 | The validated layer assumes `proposed_gates` is a mapping; a `None`/list input raises `TypeError` rather than failing closed. | The orchestrator always passes a `dict`, so the live path is safe. Recorded as a layer-level gap. |
| 8 | Rate limiting uses `current_gates.get(nid, gate)` — **if the current gate for a node is absent, no rate limit is applied** for that node. | Not reachable on the live path (gates are always read from `ReservoirNetwork`), but worth recording. |
| 9 | `get_adapted_state()` still recomputes the ML pipeline on **every** read, including every WebSocket broadcast (pre-existing since Stage 4). `step()` is now heavier because of the MPC. | Performance risk; caching is a good future item. Not addressed in Stage 8 (no premature optimization). |
| 10 | The end-to-end behavioural test covers a **single** step. There is no multi-step closed-loop soak test of the live path. | Correctness for one step is proven; long-run behaviour is not. |
| 11 | Dual node-id convention: the live facade network uses `Virtual Reservoir A…D` while the canonical validated network uses `Reservoir_A…D`. | Documented wart inherited from Stage 3; handled explicitly by `LiveForecastAdapter.for_network()`. |
| 12 | The MPC grid search (1296 candidates × 3 rollout steps) dominates per-step cost. | Fine for a demo, expensive for high-frequency control. Performance only. |

---

## 13. Hardware Status — Explicit Statement

> **Hardware is NOT connected.** Stage 8 is **software-only**.
>
> * No ESP32 / serial / MQTT / Modbus / actuator adapter was created or modified.
> * `src/hardware/` was **not** touched.
> * No GPIO, no sensor polling, no valve or gate actuation of any kind is performed.
> * The twin payload continues to report
>   `hardware_status = { esp32: NOT_CONNECTED, water_level_sensor: NOT_CONNECTED, flow_sensor: NOT_CONNECTED, gate_actuator: NOT_CONNECTED }`.
> * The only physical quantity the system moves is the **simulated** gate position
>   of `ReservoirNetwork`.

The boundary was nevertheless designed so that future hardware integration can
slot in **downstream of the SafetyLayer**:

```
MPC
 ↓
SafetyLayer                     ← already the final software boundary
 ↓
Hardware / Simulation command boundary   ← future (NOT implemented)
 ↓
ReservoirNetwork (today)   |   Actuator command (future)
```

The SafetyLayer therefore remains **upstream of any future actuator command**, and
the no-bypass guarantees in §4 (orchestrator-only application, browser 405,
GNN advisory-only) apply equally to a future hardware adapter. **The hardware
adapter was deliberately not implemented.**

---

## 14. Stage 9 Status — Explicit Statement

> **Stage 9 has NOT started.**
>
> As required by the hard boundary:
>
> * ❌ Stage 9 not started — no work of any kind beyond the Stage 8 boundary.
> * ❌ Hardware not implemented.
> * ❌ RL / MARL not implemented.
> * ❌ GNN behaviour not changed (still advisory-only; still cannot influence control).
> * ❌ MPC not redesigned — objective, candidate grid and optimization logic untouched.
> * ❌ Frozen Phase 15.3 artifacts not modified (verified by SHA256).
>
> Work stops here, at the Stage 8 boundary.

---

## Appendix A — Stage 8 Requirement Compliance Summary

| Requirement | Status |
|---|---|
| Obtain the MPC's proposed action, pass it through the **existing** validated SafetyLayer, apply the SafetyLayer output to `ReservoirNetwork` | ✅ 4-step path in `decide()`; applied == direct `validate()` output to 1e-12 |
| Use the existing SafetyLayer, not a second safety system | ✅ one `SafetyLayer` instance, class unmodified |
| Do not redesign MPC / change objective / grid / optimization / frozen implementation | ✅ `MPCController` called unchanged; source assertions + hash |
| Determine and document what the SafetyLayer **actually** checks | ✅ §3.1 (7 behaviours, incl. the unreachable `EMERGENCY` branch) |
| Do not assume a check exists because it would be desirable; report gaps | ✅ §3.2 — release bounds, downstream capacity, storage, minimum flow, prior-state and impossible-state checks are **absent and were not added**; the false "downstream capacity" docstring claim is pinned by a test |
| Provenance gate stays upstream; SafetyLayer must never make an invalid forecast valid | ✅ §5 — the layer is not even invoked for a blocked forecast |
| `DEMONSTRATION_ONLY` → MPC blocked | ✅ |
| Missing Idukki → coordinated MPC blocked, no fabricated forecast | ✅ §6 |
| `validated_metrics_apply = false` → MPC blocked | ✅ (Stage 7 gate, still green) |
| All four reservoirs first-class; D not pinned at 100% | ✅ §7 |
| Exactly one authoritative path; browser display-only; GNN advisory-only; no frontend injection; no duplicate controller; no legacy `ForecastAwareController` | ✅ §4 #6–#8 |
| Canonical units (MCM, MCM/day, metres, gate fraction); percent only at API/UI; no implicit conversions | ✅ `units.gate_fraction_to_percent` only; no `/100.0`, no `*100.0` in the orchestrator |
| Fail-closed: follow the **actual** validated behaviour; do not invent a fallback | ✅ §3 — NaN/Inf → 0.1, missing node → current gate, out-of-range → clamp, big jump → rate-limit; nothing else invented |
| Design the boundary so hardware can later sit downstream of the SafetyLayer; do not implement it | ✅ §13 |
| 20 testing requirements incl. a real behavioural integration test | ✅ §8, §4.1 |
| UI updated to display controller / safety layer / eligibility / safety status / applied vs proposed; no JS intelligence; no UI bypass | ✅ `SAFETY LAYER` + `SAFETY ACTION` HUD rows; Streamlit read-only panel; `POST/PUT /api/state` → 405 |
| Record whether the SafetyLayer introduces meaningful per-step latency | ✅ §11 — ≈ 3 µs/call, ~0.001 % of a decision |
| Frozen Phase 15.3 artifacts unchanged | ✅ §9 |

---

*End of Stage 8 Report. Stopped at the Stage 8 boundary — Stage 9 not started.*
