# STAGE 10 REPORT — DOWNSTREAM CAPACITY SAFETY BOUNDARY

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 10 boundary (Stage 11 NOT started)
**Goal:** Establish a real downstream-capacity safety boundary for the LIVE
four-reservoir control path, so the live controller cannot apply an action that
drives the flow below the terminal reservoir above the authoritative downstream
capacity (50.0 MCM/day).

---

## 1. Exact Files Changed

### Created

| File | Purpose |
|---|---|
| `src/controller/downstream_capacity_guard.py` | **NEW** — `DownstreamCapacityGuard`, `DownstreamCapacityResult`, the boundary statuses and the failure policy |
| `tests/test_stage10_downstream_capacity.py` | **NEW** — 40 Stage 10 tests |
| `results/phase15_stage10_downstream_capacity/PHASE_15_STAGE10_REPORT.md` | **NEW** — this report |

### Modified

| File | Change |
|---|---|
| `src/controller/live_mpc_orchestrator.py` | The boundary is invoked inside `decide()` **after** the SafetyLayer; the guard's output becomes the applied action. `LiveControlDecision` gains the downstream status/flow/capacity/latency block, `safety_layer_gate_positions_fraction` (the SafetyLayer's own output, to complete the audit chain), and the `final_safe_control_action_*` records. Blocked/error branches report `NOT_APPLIED_*` and fabricate nothing. |
| `src/dashboard/api/state_manager.py` | Passes the inflows that will actually be applied to `decide()` (so the prediction is about the step that really happens); the adapter-error branch reports `NOT_APPLIED_ADAPTER_ERROR` for the downstream boundary as well. |
| `src/dashboard/twin_component/state_adapter.py` | `_control_block()` normalises the new downstream fields; an absent control block reports `downstream_status = UNKNOWN` with `None` (never `True`) for `capacity_achieved` / `protection_modified`, so the twin can never imply protection it did not receive. |
| `src/dashboard/web/index.html` | Three new HUD rows — `DOWNSTREAM CAP`, `PREDICTED FLOW`, `FINAL ACTION` — displaying the backend's status/values verbatim. No decision logic in JavaScript. |
| `tests/test_stage8_safety_layer_integration.py` | 4 assertions updated from "the SafetyLayer output is final" to the Stage 10 contract (see §12.2). |
| `tests/test_stage9_four_reservoir_coordination.py` | 1 behavioural test updated for the same reason. |

### NOT modified (hard boundary respected)

`src/controller/safety.py` · `src/controller/mpc_controller.py` · `src/controller/objective.py` ·
`src/network_env/reservoir_network.py` · `src/network_env/topology_config.yaml` ·
`src/modeling/*` · `models/lstm_pytorch_v3_logtarget/*` · `data/processed/scaled/feature_scaler.pkl` ·
`results/lstm_pytorch_v3_logtarget/*` · `results/phase15_v3_validation/*` · `src/hardware/*`

* The SafetyLayer is **not** modified, and is **not** made to claim a downstream-capacity check
  it does not implement.
* No second reservoir simulator: the guard **clones** the authoritative `ReservoirNetwork`.
* No routing equation is duplicated or replaced.
* The MPC's objective, candidate grid and optimization logic are untouched.

---

## 2. Exact Downstream Safety Architecture

```
Frozen LSTM V3 → LiveForecastAdapter → NetworkForecastSnapshot
                                              ↓
                                   PROVENANCE GATE            (Stage 7, unchanged)
                                              ↓
                                   MPCController.decide()     (validated, UNMODIFIED)
                                   action space: 6^4 = 1296 joint candidates
                                              ↓  raw proposal (4 gates)
                                   SafetyLayer.validate()     (validated, UNMODIFIED)
                                              ↓  bounds [0,1] + per-step rate limit
                                   DownstreamCapacityGuard.evaluate()   ← NEW (Stage 10)
                                              ↓  FINAL_SAFE_CONTROL_ACTION (4 gates)
                                   ReservoirNetwork.step()
                                              ↓
                                   state → FastAPI → WebSocket → Three.js Twin
```

The three layers report **separate** statuses that are never conflated:

| Layer | Field | Values |
|---|---|---|
| MPC | `controller_status` | `ACTIVE` / `BLOCKED` / `UNAVAILABLE` |
| Existing SafetyLayer | `safety_layer_status` | `SAFE` / `CORRECTED` / `NOT_APPLIED_MPC_BLOCKED` / `NOT_APPLIED_MPC_ERROR` / `NOT_APPLIED_ADAPTER_ERROR` |
| **Downstream capacity** | `downstream_status` | `PROTECTED` / `CORRECTED` / `FAILED_CLOSED` / `NOT_APPLIED_MPC_BLOCKED` / `NOT_APPLIED_MPC_ERROR` / `NOT_APPLIED_ADAPTER_ERROR` / `NOT_APPLIED_NO_CAPACITY` |

The system therefore **cannot** report *"SafetyLayer ACTIVE = downstream capacity
protected"*: the SafetyLayer's own status is unchanged from Stage 8 (and its
docstring still falsely claims a downstream-capacity check, which Stage 10
documents rather than edits), while downstream protection has its own status that
is only `PROTECTED`/`CORRECTED` when the check actually ran and the applied action
was proved to stay within the capacity.

### 2.1 Audit chain exposed to the API / WebSocket / Twin

| Stage | Field |
|---|---|
| MPC raw proposal | `proposed_gate_positions_fraction` (transport-safe: finite or `None`) |
| SafetyLayer output | `safety_layer_gate_positions_fraction` / `_pct` |
| **Final applied action** | `gate_positions_fraction` / `_pct` **and** `final_safe_control_action_fraction` / `_pct` (`source = DOWNSTREAM_CAPACITY_GUARD`) |
| Predicted downstream flow | `downstream_proposed_predicted_flow_mcm_day` → `downstream_predicted_flow_mcm_day` |
| Capacity | `downstream_capacity_mcm_day` (+ `unit`, `tolerance_mcm_day`, `horizon_steps`, `physics`) |
| Whether protection modified the action | `downstream_protection_modified`, `downstream_reason` |

---

## 3. Why the Chosen Integration Boundary Is Correct

Determined by inspecting the validated architecture, not assumed:

| Candidate boundary | Verdict |
|---|---|
| **Inside `ReservoirNetwork`** | ❌ It is the authoritative *physics* model and must remain a pure model. Enforcing control policy there would also make the validated network non-reusable for the offline Phase 15.3 pipeline. **Not touched.** |
| **Inside `SafetyLayer`** | ❌ It is a frozen Phase 15.3 artifact; the Stage 8 finding was precisely that it does *not* implement this check, and the instruction forbids modifying it to add the feature. It also receives no network state, so it *cannot* predict cascade behaviour. |
| **Inside `MPCController`** | ❌ Would change its objective/constraint structure — explicitly forbidden ("do NOT silently change MPC's optimization objective"). The MPC prices downstream excursions as a soft penalty, which is validated behaviour. |
| **Inside the generic optimisation objective** | ❌ Same reason, and it would make the safety property statistical rather than guaranteed. |
| **A separate component inside `LiveMPCOrchestrator.decide()`, after the SafetyLayer** | ✅ **CHOSEN.** `LiveMPCOrchestrator` is the ONE authoritative live controller path and is the only place that holds simultaneously: (a) the live `ReservoirNetwork` (state + routing queues), (b) the action that survived the SafetyLayer, and (c) the authority to decide what is applied. It is new code, so no validated artifact is disturbed, and placing the boundary *after* the SafetyLayer makes the pipeline monotone: each layer can only restrict, never loosen. |

The instruction's suggested placement is therefore confirmed, with the additional
constraint that the boundary may only emit actions that the SafetyLayer would pass
**unchanged** (§4.2) — otherwise it would silently invalidate the guarantee that
precedes it.

---

## 4. Exact Capacity Calculation

### 4.1 What is measured

**Downstream flow ≡ the terminal reservoir's `total_outflow` (MCM/day)**, exactly
as the authoritative model defines it:

* `ReservoirNetwork.step()` computes `terminal_outflow = terminal_node.state.total_outflow`
  and logs `DOWNSTREAM CAPACITY EXCEEDED` when it is `> downstream_capacity`;
* `ObjectiveFunction.evaluate_trajectory()` scores violations against the *same*
  quantity (`terminal_state.total_outflow` vs `downstream_capacity`).

Water spilled from **non-terminal** reservoirs leaves the network and is *not*
part of the downstream flow (established and tested in Stage 9). Because
`total_outflow = controlled_release + spill` and, per step,
`total_outflow = max(release, available − reservoir_capacity)`, the flow cannot be
reduced by opening a gate — only by closing one.

### 4.2 The comparison

```
predicted_flow(t)  =  clone(network).step(inflows, action)[terminal].total_outflow
safe  ⟺  max over t ∈ [0, horizon) of predicted_flow(t)  ≤  capacity + 1e-9 MCM/day
```

* `capacity = network.downstream_capacity` → **50.0 MCM/day** (read from the
  authoritative network, never hardcoded).
* `TOLERANCE_MCM_DAY = 1e-9` — documented numerical tolerance. Chosen far below
  any physically meaningful flow: one gate level changes the flow by ~0.4 MCM/day,
  and the network's own mass-balance residual is ~1e-13 MCM.
* Non-finite or `<= 0` capacity → `NOT_APPLIED_NO_CAPACITY`: the boundary
  explicitly refuses to claim a guarantee it cannot make.

The check is **not** `sum(current releases) <= 50`: it is a rollout of the
authoritative model that includes the routing queues, the spill rule and the
cascade coupling (§6).

### 4.3 The deterministic action policy

| Situation | Action applied | Status |
|---|---|---|
| Proposal already safe | **unchanged** | `PROTECTED` |
| Proposal unsafe, a safe alternative exists | the admissible action **closest** (L1) to the proposal | `CORRECTED` |
| **No** admissible action can satisfy the capacity | the **flow-minimising** admissible action | `FAILED_CLOSED` + `capacity_achieved = False` |

Why the minimum action is a *proof*, not a guess: downstream flow is monotone
non-decreasing in **every** gate (a larger terminal gate means a larger release; a
larger upstream gate means more water arriving at the terminal reservoir). The
flow-minimising action over the admissible box
`[current − max_gate_change, current + max_gate_change] ∩ [0,1]` is therefore the
all-minimum corner. The guard evaluates it **before** searching, which:

1. makes `FAILED_CLOSED` a provable statement ("no admissible action can limit the
   flow; the remainder is forced spill"), rather than "we did not find one"; and
2. bounds the search — if the minimum action is safe, a safe action is guaranteed
   to exist, so the search always terminates with a solution.

The candidate search space is the **validated MPC's own gate lattice** (never a
new action space), augmented **per reservoir** with: `0.0`, the project's
conservative value (`0.1`, i.e. `SafetyLayer.emergency_fallback`), the analytic
gate `capacity / max_release`, and — importantly — **the proposal and the current
gate**, so the guard can always leave a reservoir exactly as the SafetyLayer
produced it instead of gratuitously moving reservoirs that are not part of the
problem. Candidates are ordered by increasing L1 distance from the proposal with
enumeration order as tie-break → deterministic and minimal-intervention.

Every candidate is filtered to be **SafetyLayer-feasible** (bounds `[0,1]` and
`|gate − current| ≤ max_gate_change`) before it is considered, so the final action
is admissible under *both* boundaries and the SafetyLayer's guarantees compose
rather than being overridden.

---

## 5. Prediction Horizon

```
horizon = 1 + sum(routing delays)  =  1 + (2 + 1 + 1)  =  5 steps
```

Derived from the authoritative topology (never hardcoded): one step for the action
about to be applied, plus one step per unit of cumulative routing delay. That
covers the full propagation of the applied action from Reservoir A (top of the
cascade) to the river below Reservoir D — e.g. water released by A at step 0 can
first affect D's outflow at step 4.

Two documented modelling choices, both conservative and both matching the existing
validated controller:

* **Gates are held constant across the horizon** — the same convention
  `MPCController._simulate_trajectory()` uses ("hold gates constant across
  lookahead"). It assumes the action persists, which is the pessimistic
  assumption for a safety check. Only step 0 is ever applied, and the boundary
  re-runs on every decision.
* **Exogenous inflows are held constant** at the values the live controller passes
  (the inflows that will actually be applied in the step) or, if absent, at the
  network's currently observed local inflows. Deliberately **not** taken from the
  forecast: a safety boundary must not depend on the quality of the forecast whose
  validity the Stage 7 provenance gate exists to police. Water already committed to
  the routing queues is real state and **is** included, so arriving flood waves are
  seen.

Pinned by `test_horizon_is_derived_from_the_validated_routing_delays`.

---

## 6. Proof the Authoritative Routing Physics Is Used

| Evidence | Test |
|---|---|
| The guard steps a **clone of the live `ReservoirNetwork`** (`config_dict=deepcopy(network._raw_config)`), copying the REAL storages and the REAL routing queues. It defines no routing, delay, attenuation or spill equation of its own (source-asserted: no `ConnectionState`, no `deque(`). | `test_protected_components_are_unmodified_by_stage10` |
| The guard's prediction equals an **independent** rollout of the authoritative network to 1e-12, step by step. | `test_prediction_uses_the_authoritative_network_physics` |
| The guard's prediction equals the **validated MPC's own** `_simulate_trajectory()` terminal outflow to 1e-12 — the same model *and* the same constant-gate convention. | `test_prediction_matches_the_validated_mpc_rollout` |
| The prediction **never mutates** the live network (storages, queues, timestep, gates compared before/after). | `test_prediction_does_not_mutate_the_live_network` |
| **Cascade delays are respected**: with only A open and D closed the downstream flow is 0 for all 5 steps, and B receives A's water exactly 2 steps later. | `test_cascade_delays_are_respected_by_the_prediction` |
| **Attenuation is respected**: a 100 MCM slug placed in the C→D queue arrives at D as **80 MCM** (0.80), and the predicted flow is 80 — which the boundary then rejects (> 50). | `test_attenuation_is_respected_by_the_prediction` |
| The capacity threshold `capacity / max_release = 0.25` predicts exactly 50.0 MCM/day. | `test_capacity_threshold_is_the_analytic_release_boundary` |
| The horizon is derived from the validated delays 2/1/1. | `test_horizon_is_derived_from_the_validated_routing_delays` |

---

## 7. Proof All Four Reservoirs Participate

| Evidence | Test |
|---|---|
| The boundary evaluates a four-gate action and returns a four-gate action; the horizon/physics metadata is present. | `test_all_four_reservoirs_participate_in_the_safety_check` |
| The applied action, the SafetyLayer's output and the final-safe record all carry exactly the four live node ids. | same |
| **D is not pinned**: the terminal gate the boundary applies is the *computed* `capacity / max_release = 0.25`, never the legacy 100 %, and it is chosen from the state rather than fixed. | `test_reservoir_d_is_not_pinned_by_the_boundary` |
| **Minimal intervention**: when only the terminal reservoir needs reducing, A, B and C keep *exactly* the gates the SafetyLayer produced. | `test_boundary_keeps_other_reservoirs_as_proposed` |
| Nothing protects "only A/B/C": the boundary changes whichever reservoirs are required, and the fail-closed action closes **all four**. | `test_failed_closed_action_is_the_flow_minimiser` |

Concrete measured example (live network, flood proposal `{A:1.0, B:1.0, C:1.0, D:1.0}`):

```
SafetyLayer output : {A: 0.50, B: 0.50, C: 0.50, D: 0.50}
  predicted flow   : 100.0 MCM/day   > 50.0   → UNSAFE
guard output       : {A: 0.50, B: 0.50, C: 0.50, D: 0.25}
  predicted flow   :  50.0 MCM/day   ≤ 50.0   → SAFE   (7 candidates evaluated)
```

---

## 8. Proof Unsafe Flow Cannot Reach `ReservoirNetwork`

### 8.1 The required behavioural test (unsafe case)

`test_downstream_capacity_of_50_is_enforced_on_the_live_network` drives the
**authoritative singleton** end to end:

1. MANUAL mode, all gates 0.0, `sim.step()` → physics gates really are 0.0.
2. AI mode with an injected **validated** bundle; the MPC is monkeypatched to
   return the flood proposal `{1.0, 1.0, 1.0, 1.0}`.
3. `sim.step()`.
4. Assertions:
   * `downstream_status = CORRECTED`, `downstream_protection_modified = True`;
   * the gate positions **actually written to `ReservoirNetwork`** equal the
     reported final action for all four reservoirs;
   * **`ReservoirNetwork`'s terminal outflow ≤ 50.0 + 1e-9 MCM/day** — checked on
     the real network, not on the clone the guard used;
   * the action that the MPC+SafetyLayer produced, applied verbatim, *would* have
     predicted > 50 (counter-proof that the boundary is load-bearing —
     `test_bypassing_the_boundary_would_have_violated_the_capacity`).

Earlier stages' end-to-end tests were extended in the same way, and the Stage 10
suite re-verifies: the final action is what the network received
(`test_final_applied_action_is_what_the_network_received`), the raw proposal cannot
bypass the boundary (`test_raw_mpc_output_cannot_bypass_the_downstream_boundary`),
and every applied action is still SafetyLayer-clean — proven with the **real**
`SafetyLayer` object, which reports zero violations and returns the action
unchanged (`test_every_candidate_the_boundary_applies_is_safety_layer_clean`).

### 8.2 The required behavioural test (safe case)

`test_safe_action_is_not_unnecessarily_modified_even_with_headroom` applies the
known-safe action `{A:0.5, B:0.5, C:0.5, D:0.25}` (predicted flow exactly
50.0 MCM/day) and asserts `PROTECTED`, `modified = False`, and that **every gate is
returned bit-identical**. `test_safe_action_passes_unchanged` does the same through
a real MPC decision.

---

## 9. Safe-Action Pass-Through Proof

| Evidence | Result |
|---|---|
| Real MPC decision on a validated bundle | `PROTECTED`, `modified = False`, applied gates == SafetyLayer output to 1e-12 |
| Known-safe action at exactly the capacity (50.0) | `PROTECTED`, unchanged, flow = 50.0 ≤ 50.0 + 1e-9 |
| The boundary adds no "improvement" when the action is safe | the fast path performs **one** prediction and shares no code with the search |

---

## 10. Provenance Behaviour

The Stage 7 gate is untouched and was **not** weakened:

| Case | Result | Test |
|---|---|---|
| `DEMONSTRATION_ONLY` | MPC `BLOCKED`; SafetyLayer `NOT_APPLIED_MPC_BLOCKED`; **downstream `NOT_APPLIED_MPC_BLOCKED`**; `capacity_achieved = False`; `final_safe_control_action_source = HELD_CURRENT_GATES`; gates = the real current gates | `test_demonstration_forecast_still_blocks_and_the_boundary_does_not_fabricate` |
| Missing Idukki forecast | `BLOCKED`, reason `Virtual Reservoir D:MISSING_FORECAST`, downstream not applied | `test_missing_idukki_forecast_still_blocks` |
| `validated_metrics_apply = false` | `BLOCKED` (Stage 7 gate, still green) | Stage 7 suite |
| Blocked forecast | the downstream boundary is **never invoked** (spy records zero calls) — it cannot manufacture or "rescue" a control decision | `test_demonstration_forecast_never_reaches_the_boundary` |
| `NaN` / `Inf` proposals | still fail-safe: every applied gate is finite and in `[0,1]`; the payload stays JSON-serialisable | `test_nan_proposal_is_still_fail_safe`, `test_infinite_proposal_is_still_fail_safe` |
| Network declares no capacity | `NOT_APPLIED_NO_CAPACITY`, `is_protected = False` | `test_no_downstream_capacity_declared_is_reported_not_claimed` |

**Consequence for today's demo:** because the live forecasts are
`DEMONSTRATION_ONLY`, the live HUD truthfully reads
`DOWNSTREAM CAP · NOT APPLIED MPC BLOCKED` (verified in a live browser session) —
it does **not** claim protection.

---

## 11. Digital Twin Status / Provenance

Three new HUD rows, all fed verbatim from the backend payload:

| Row | Content | Colour |
|---|---|---|
| `DOWNSTREAM CAP` | `downstream_status` (underscores → spaces) | green `PROTECTED`; amber `CORRECTED` / `FAILED_CLOSED`; faint otherwise |
| `PREDICTED FLOW` | `predicted / capacity MCM/d` | green when `downstream_capacity_achieved`, amber otherwise |
| `FINAL ACTION` | `ALL CHECKS PASSED` / `DOWNSTREAM-CORRECTED` / `HELD (NOT APPLIED)` | green / amber / faint |

Existing rows `SAFETY LAYER` and `SAFETY ACTION` are unchanged, so the three
statuses are visible **side by side** and cannot be conflated. JavaScript performs
no decision logic: it displays the backend's strings and numbers, and the status
is only compared against literals for *colouring* (`test_twin_ui_displays_downstream_protection_truthfully`).

The twin payload also carries the full audit chain (§2.1) and the
`downstream_capacity_protection` block (`unit`, `physics`, `horizon_steps`,
`tolerance_mcm_day`, predicted traces' maxima, `min_achievable_flow_mcm_day`,
`candidates_evaluated`, `reason`).

**Live browser verification performed** (uvicorn on port 8124): the page renders
all three rows, reports `NOT APPLIED MPC BLOCKED` for both the SafetyLayer and the
downstream boundary while the gate blocks, `4 RESERVOIRS`, and the DEMONSTRATION
badge — with no page errors.

---

## 12. Tests and Counts

| Suite | Result |
|---|---|
| **Stage 10 tests** (`tests/test_stage10_downstream_capacity.py`) | ✅ **40 passed** |
| **Complete test suite** | ✅ **509 passed, 0 failed** (78.1 s) |
| Stage 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 | ✅ 28 / 41 / 38 / 56 / 40 / 38 / 43 / 40 (all EXIT=0) |
| Stage 3–10 combined | ✅ **324 passed, 0 failed** |

Suite growth: 469 (end of Stage 9) → **509** (Stage 10, +40).

### 12.1 Coverage against the 22 requested test areas

| Req. | Requirement | Stage 10 test(s) |
|---|---|---|
| 1 | Safe action passes unchanged | `test_safe_action_passes_unchanged`, `test_safe_action_is_not_unnecessarily_modified_even_with_headroom` |
| 2 | Unsafe downstream action is detected | `test_unsafe_downstream_action_is_detected_and_corrected` |
| 3 | Capacity 50.0 is enforced | `test_downstream_capacity_of_50_is_enforced_on_the_live_network`, `test_capacity_threshold_is_the_analytic_release_boundary` |
| 4 | Prediction uses authoritative `ReservoirNetwork` physics | `test_prediction_uses_the_authoritative_network_physics`, `test_prediction_matches_the_validated_mpc_rollout` |
| 5 | Cascade delays respected | `test_cascade_delays_are_respected_by_the_prediction`, `test_horizon_is_derived_from_the_validated_routing_delays` |
| 6 | Attenuation respected | `test_attenuation_is_respected_by_the_prediction` |
| 7 | All four reservoirs participate | `test_all_four_reservoirs_participate_in_the_safety_check` |
| 8 | D is not pinned | `test_reservoir_d_is_not_pinned_by_the_boundary` |
| 9 | Existing SafetyLayer remains in the path | `test_safety_layer_remains_in_the_path`, `test_guard_sits_between_the_safety_layer_and_the_network` |
| 10 | Raw MPC proposal cannot bypass safety | `test_raw_mpc_output_cannot_bypass_the_downstream_boundary`, `test_bypassing_the_boundary_would_have_violated_the_capacity` |
| 11 | Demonstration forecasts remain blocked | `test_demonstration_forecast_still_blocks_and_the_boundary_does_not_fabricate` |
| 12 | Missing Idukki forecast remains blocked | `test_missing_idukki_forecast_still_blocks` |
| 13 | NaN/Inf behaviour remains fail-safe | `test_nan_proposal_is_still_fail_safe`, `test_infinite_proposal_is_still_fail_safe` |
| 14 | Safe constrained action is deterministic | `test_safe_constrained_action_is_deterministic`, `test_guard_result_is_deterministic_for_identical_state` |
| 15 | Final applied action IS what was sent to `ReservoirNetwork` | `test_final_applied_action_is_what_the_network_received`, `test_final_safe_control_action_label_exists_for_the_future_hardware_boundary` |
| 16 | WebSocket reports truthful safety provenance | `test_websocket_state_carries_downstream_provenance`, `test_api_reports_the_three_statuses_separately`, `test_twin_payload_never_implies_protection_without_evidence` |
| 17 | Browser cannot bypass downstream protection | `test_browser_cannot_bypass_downstream_protection` |
| 18 | GNN cannot bypass safety | `test_gnn_cannot_bypass_the_downstream_boundary` |
| 19 | Canonical units remain correct | `test_canonical_units_in_the_downstream_boundary` |
| 20 | Stage 3–9 tests remain green | ✅ 284/284 — see §13 |
| 21 | Frozen artifacts unchanged | `test_stage10_does_not_modify_frozen_artifacts`, `test_protected_components_are_unmodified_by_stage10` |
| 22 | Phase 15.3 reproduction identical | §15 |

Plus the fail-closed proof `test_fail_closed_is_reported_when_the_capacity_is_unachievable`
and `test_failed_closed_action_is_the_flow_minimiser`.

### 12.2 Deliberate updates to earlier-stage assertions

Stage 10 changes *which* action is final, so five assertions that pinned "the
SafetyLayer's output is the applied action" were updated to the new contract. Their
**intent is preserved** — each layer's rule is still enforced, and the audit chain
now exposes the SafetyLayer's own output separately:

| File | Test | Change |
|---|---|---|
| `tests/test_stage8_safety_layer_integration.py` | `test_safety_layer_output_equals_a_direct_validate_call` | Asserts `safety_layer_gate_positions_fraction == validate()` to 1e-12, and that the applied action is **at most** that (never looser). |
| same | `test_rate_limit_is_enforced` | Asserts the rate limit on the SafetyLayer's output and `applied ≤ limit` (so the rate limit still holds). |
| same | `test_reservoir_d_is_not_pinned_to_the_legacy_100_percent` | Asserts the SafetyLayer output for D is the rate-limited 0.5 and `applied ≤ 0.5` (D is now also the reservoir the downstream boundary constrains). |
| same | `test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network` | End-to-end now asserts the **final** action reached the network, and separately that the SafetyLayer produced the sanitised action. |
| `tests/test_stage9_four_reservoir_coordination.py` | `test_four_reservoir_proposal_through_safety_reaches_the_physics` | Same split. |

No other Stage 3–9 assertion was weakened, loosened or deleted.

---

## 13. Stage 3–9 Regression

| Suite | Result |
|---|---|
| Stage 3 — live network authority | ✅ 28 passed |
| Stage 4 — single authoritative simulation | ✅ 41 passed |
| Stage 5 — forecast provenance | ✅ 38 passed |
| Stage 6 — forecast adapter | ✅ 56 passed |
| Stage 7 — live MPC integration | ✅ 40 passed |
| Stage 8 — SafetyLayer integration | ✅ 38 passed (4 assertions updated, §12.2) |
| Stage 9 — four-reservoir coordination | ✅ 43 passed (1 test updated, §12.2) |
| **Total** | ✅ **284 passed, 0 failed** |

---

## 14. Frozen-Artifact Status

`scripts/stage3_verify_frozen_artifacts.py` → **exit 0 — `VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`**

| Artifact | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | ✅ MATCH |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | ✅ MATCH |
| `data/processed/scaled/feature_scaler.pkl` | ✅ read-only |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | ✅ MATCH (LF-normalised; raw difference is the known `core.autocrlf` artefact) |

Also hashed around a full boundary-corrected orchestrator run inside the Stage 10
suite, and asserted by source that `safety.py`, `mpc_controller.py` and
`reservoir_network.py` contain no reference to the new boundary.

---

## 15. Phase 15.3 Reproduction Status

`scripts/stage3_phase15_3_reproduction.py` re-run **shielded** → **exit 0**:

* **Protected artifacts: 6/6 UNCHANGED**
* **Reproduction fidelity: 3/3 files byte-for-byte IDENTICAL**

| Metric | Baseline | MPC | Diff |
|---|---|---|---|
| overflow_events | 7.00 | 0.00 | −7.00 BETTER |
| overflow_volume_mcm | 10.75 | 0.00 | −10.75 BETTER |
| ds_violations | 8.00 | 0.00 | −8.00 BETTER |
| peak_ds_flow | 60.00 | 30.00 | −30.00 BETTER |
| total_release_mcm | 2017.25 | 2031.00 | +13.75 WORSE |
| forecast_dates_used | N/A | 74 | — |

Mass-balance residual identical in both runs (−6.82e−13); V3 integrity PASSED.

The offline pipeline does not use the live orchestrator, so the new boundary cannot
affect it — and the byte-identical reproduction confirms that.

---

## 16. Performance / Latency

Measured on the live four-reservoir network (5-step horizon; no optimization was
performed — correctness first).

| Path | Latency (ms) | Notes |
|---|---|---|
| **SAFE** (taken every step in practice) | min 0.273 · **mean 0.372** · max 0.690 | 1 prediction (5 steps); no search |
| **CORRECTED** | min 7.09 · **mean 8.90** · max 11.85 | 7 candidates evaluated (nearest-first exits early) |
| **FAILED_CLOSED** (worst case) | min 0.962 · **mean 1.425** · max 2.723 | 2 predictions only — the monotonicity proof removes the need to search the whole space |
| SafetyLayer (for comparison) | ≈ 0.027 | Stage 8 |
| Whole `decide()` (for comparison) | ≈ 660 | dominated by the MPC's 1296-candidate grid search |

The boundary costs **≈ 0.06 % of a decision on the safe path** and **≈ 1.4 % when
it has to correct an unsafe action**, i.e. it introduces no meaningful per-step
latency. The exhaustive worst case is avoided by design: because downstream flow is
monotone in every gate, an unsafe minimum action *proves* that no search can
succeed, so the boundary short-circuits instead of scanning the candidate grid.

---

## 17. Remaining Limitations

| # | Limitation | Impact |
|---|---|---|
| 17.1 | **The boundary guarantees one step at a time.** It predicts 5 steps ahead but applies only the first; the guarantee is re-established every step. It is not a proof about an arbitrary future policy. | Standard safety-filter semantics; documented in the module. |
| 17.2 | **Fail-closed may release less than the demo's `minimum_environmental_flow_percent = 0.05` assumption.** In an unavoidable-flood state the boundary prioritises flood safety and reports `FAILED_CLOSED` loudly. That assumption is a SIMULATION ASSUMPTION not enforced anywhere in the validated code (verified in Stage 8). | Documented trade-off; unreachable in normal operation (pinned by tests). |
| 17.3 | **The search space is a lattice, not the continuous box.** The guard includes the analytic threshold `capacity / max_release` and the incoming values so it does not spuriously fail, but on a *pathological* lattice it could choose an action slightly more restrictive than strictly necessary. | Conservative direction only; the `FAILED_CLOSED` verdict is still provable via the minimum action. |
| 17.4 | **The prediction assumes constant gates and constant inflows over the horizon** (both matching the validated MPC's own convention). A future decision step that re-opens gates could exceed the capacity *then* — and is caught at that step. | Documented; the boundary re-runs every decision. |
| 17.5 | **`downstream_capacity` is not a hard constraint of the MPC's objective** (still a soft penalty, weight 200). The boundary is therefore load-bearing rather than a safety net for a rare case. | This is why the boundary exists; the MPC could not be modified. |
| 17.6 | **The SafetyLayer still does not check downstream capacity**, and its class docstring still falsely claims it does. Stage 10 documents this (and a test asserts the false claim is still present so it cannot be quietly "fixed" by a future reader) instead of editing a frozen artifact. | Reported; the two boundaries are now explicit and separately reported. |
| 17.7 | **The live AI path is still BLOCKED** (demonstration-only forecasts, no live Idukki forecast), so the boundary's live status is `NOT_APPLIED_MPC_BLOCKED` in today's demo. | Intended; proven with an injected validated bundle. |
| 17.8 | **Reservoir D still has no 3D mesh** in the twin scene (Stage 9 limitation 12.1), unchanged. | Visual only. |
| 17.9 | **`FAILED_CLOSED`'s "no admissible action" proof assumes `ReservoirNetwork`'s spill rule** (`total_outflow = max(release, available − capacity)`). If that physics ever changed, the monotonicity argument would need re-verification. | Pinned by `test_the_physics_source_is_unmodified` (Stage 9) and §14. |
| 17.10 | **The boundary is not exercised over a long closed-loop soak**; the end-to-end tests cover individual corrected steps. | Correctness per step is proven; long-run behaviour is not. |

---

## 18. Hardware Status — Explicit Statement

> **Hardware is NOT connected.** Stage 10 is **software-only**.
>
> * No ESP32 / serial / MQTT / Modbus / actuator adapter was created or modified.
> * `src/hardware/` was **not** touched.
> * No GPIO, no sensor polling, no gate or valve actuation.
> * The twin payload still reports `hardware_status = {esp32, water_level_sensor, flow_sensor, gate_actuator} = NOT_CONNECTED`
>   (verified live against the running API).

The final action is now explicitly labelled for the future hardware boundary:

```
MPC
 ↓
SafetyLayer
 ↓
Downstream Capacity Protection
 ↓
FINAL_SAFE_CONTROL_ACTION          ← final_safe_control_action_fraction / _pct
 ↓
Hardware Adapter                   ← NOT implemented (future)
```

`final_safe_control_action_source` names where the action came from
(`DOWNSTREAM_CAPACITY_GUARD` when the boundary ran, `HELD_CURRENT_GATES` when the
provenance gate blocked), and `final_safe_control_action_fraction` is pinned by
test to equal the gates actually written to `ReservoirNetwork` — so a future
adapter can consume exactly this field and inherit every guarantee above it. **The
hardware adapter was deliberately not implemented.**

---

## 19. Stage 11 Status — Explicit Statement

> **Stage 11 has NOT started.**
>
> * ❌ Stage 11 not started — no work beyond the Stage 10 boundary.
> * ❌ Hardware not implemented.
> * ❌ RL / MARL not implemented.
> * ❌ GNN role not changed (advisory-only; `gnn` does not appear in the new module at all).
> * ❌ MPC not redesigned — objective, candidate grid and optimization logic untouched.
> * ❌ `ReservoirNetwork` not rewritten — the boundary clones and steps it, unmodified.
> * ❌ No real telemetry fabricated; no unrelated optimization added.
> * ❌ Frozen Phase 15.3 artifacts not modified (SHA256-verified).
>
> Work stops here, at the Stage 10 boundary.

---

## Appendix A — Requirement Compliance Summary

| Requirement | Status |
|---|---|
| Establish a real downstream-capacity boundary for the live path | ✅ §2, §8 |
| Do NOT pretend the existing SafetyLayer enforces it | ✅ §2, §17.6 — statuses are separate; the false docstring claim is pinned, not edited |
| Do NOT modify the frozen SafetyLayer to add the feature | ✅ §1 (unmodified) |
| Inspect first and determine the correct integration boundary | ✅ §3 (four alternatives rejected with reasons) |
| Do NOT create a fake safety claim | ✅ `downstream_status` is separate; live HUD truthfully shows `NOT APPLIED MPC BLOCKED` |
| Use the authoritative `ReservoirNetwork`; no second simulator; no duplicated routing; no topology replacement | ✅ §6 |
| Detect whether the action can exceed 50.0 per the authoritative physics; not just `sum(releases)` | ✅ §4, §6 |
| Respect routing/cascade behaviour; document the horizon | ✅ §5, §6 |
| Safe → apply unchanged; unsafe → deterministic documented policy; fail closed if impossible | ✅ §4.3, §8, §9 |
| Do not silently change the MPC's objective | ✅ §1 (unmodified) |
| All four reservoirs as one coordinated action; do not pin D; no special-case D | ✅ §7 |
| Canonical units; no implicit conversions | ✅ §12.1 (req 19) |
| Preserve the Stage 7 provenance gate; never fabricate an action for a blocked forecast | ✅ §10 |
| Digital Twin exposes truthful, separate statuses + proposed/safety/applied/flow/capacity/reason | ✅ §11 |
| No decision logic in JavaScript | ✅ §11 |
| Hardware not implemented; final action clearly labelled `FINAL_SAFE_CONTROL_ACTION` downstream of all software safety | ✅ §18 |
| 22 testing areas + unsafe behavioural test + safe pass-through test | ✅ §12 |
| No scope creep (no hardware, RL/MARL, GNN change, MPC redesign, frozen-artifact changes, `ReservoirNetwork` rewrite, fabricated telemetry, Stage 11) | ✅ §19 |

---

*End of Stage 10 Report. Stopped at the Stage 10 boundary — Stage 11 not started.*
