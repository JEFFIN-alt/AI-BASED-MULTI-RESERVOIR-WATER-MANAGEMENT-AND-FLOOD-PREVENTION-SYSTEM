# STAGE 9 REPORT — COORDINATED FOUR-RESERVOIR CONTROL

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 9 boundary (Stage 10 NOT started)
**Goal:** Ensure the live MPC + SafetyLayer path performs genuinely coordinated
four-reservoir control, with all four reservoirs first-class throughout the
state, the forecast contract, the MPC, the SafetyLayer, the network transition,
the WebSocket state and the Digital Twin.

Validated topology under test:

```
A = Anayirankal  →  B = Ponmudi  →  C = Idamalayar  →  D = Idukki (terminal)
        delay 2/att 0.90   delay 1/att 0.85    delay 1/att 0.80
        downstream capacity 50.0 MCM/day
```

---

## 1. Exact Files Changed

### Created

| File | Purpose |
|---|---|
| `tests/test_stage9_four_reservoir_coordination.py` | **NEW** — 43 Stage 9 tests |
| `results/phase15_stage9_four_reservoir_coordination/PHASE_15_STAGE9_REPORT.md` | **NEW** — this report |

### Modified

| File | Change |
|---|---|
| `src/dashboard/api/routes.py` | `POST /api/gate/{reservoir_id}` now accepts **`reservoir_4` → `Virtual Reservoir D`**. Previously the mapping stopped at `reservoir_3`, so Reservoir D had **no API command at all** and its hardcoded baseline was literally unchangeable. Mapping extracted to the named constant `RESERVOIR_ID_TO_NODE`. |
| `src/dashboard/api/state_manager.py` | (a) the hardcoded `"Virtual Reservoir D": 100.0` manual baseline is **removed** (now 50.0, an ordinary operator baseline); (b) the silent `if v_name == "Virtual Reservoir D": continue` in `_run_ml_pipeline()` is replaced by the named, documented policy constant `LIVE_FORECAST_EXCLUDED`; (c) comment edits only beyond that. |
| `src/dashboard/sim_bridge.py` | `get_state()` emits authoritative reservoir identity per reservoir: `node_id`, `repository_name` (from the validated config), `cascade_position`, `is_terminal`. Additive. |
| `src/dashboard/twin_component/state_adapter.py` | Reservoir mapping extended to **all four** (`reservoir_4`); `TWIN_KEY_ORDER`; identity pass-through per reservoir; a new top-level **`cascade`** inventory block; defaults now cover four reservoirs. |
| `src/controller/live_mpc_orchestrator.py` | New **read-only** `action_space_for(node_ids)` + an `action_space` provenance block on every decision (dimension, node ids, gate levels, candidate-vector count, `coordinated`). **The MPC itself is untouched.** |
| `src/dashboard/web/index.html` | Fourth telemetry card (`R4 Idukki`), fourth topology node (`R1→R2→R3→R4→DS`), fourth gate slider (`G4`), reservoir names driven by the payload's `cascade` block (fallback: the validated names), new `RESERVOIRS` / `MPC ACTION` HUD rows, and a **bug fix** (see §1.1). |
| `tests/test_gate_unit_regression.py` | One assertion list updated: `reservoir_4` is no longer an "unknown id" (it is now valid), replaced by `reservoir_5`. Guard intent unchanged. |
| `tests/test_stage4_single_authoritative_simulation.py` | Same one-line change (`reservoir_4` → `reservoir_5`) in the forged/unknown-id rejection list. |

### NOT modified (hard boundary respected)

`src/controller/mpc_controller.py` · `src/controller/safety.py` · `src/controller/objective.py` ·
`src/network_env/reservoir_network.py` · `src/network_env/topology_config.yaml` ·
`src/network_env/v3_forecast_adapter.py` · `models/lstm_pytorch_v3_logtarget/*` ·
`data/processed/scaled/feature_scaler.pkl` · `results/lstm_pytorch_v3_logtarget/*` ·
`results/phase15_v3_validation/*` · `src/hardware/*` · `src/modeling/gnn_inference.py`

* **No MPC redesign.** The objective, candidate grid, lookahead and optimization logic
  are byte-for-byte unchanged; Stage 9 only *describes* the action space.
* **No second safety layer**, no change to SafetyLayer behaviour.
* **No change to routing delays, attenuation, storage equations, spill behaviour,
  mass balance or downstream-capacity definitions.**

### 1.1 Bug found and fixed during Stage 9

`ReservoirTwinRenderer._setDemoBadge()` read `this.refs`, **which the class never
assigns**. The DEMONSTRATION provenance badge introduced in Stage 6 was therefore
dead code and **never appeared in the browser**, even though the Stage 6 tests
(string-level source assertions) passed. Stage 9 corrected it to read the real
element cache (`this.el`). Verified in a live browser session: the badge now
renders (`#demo-badge.on`) and shows *"DEMONSTRATION — MODEL INPUTS SIMULATED /
validated metrics do not apply"*. Pinned by
`test_demo_badge_wiring_is_not_dead_code`.

---

## 2. Exact Four-Reservoir Control Path

```
Frozen LSTM V3
      ↓
LiveForecastAdapter.for_network(network)          ← bound to the LIVE node ids
      ↓
NetworkForecastSnapshot   (4 nodes; D explicitly FORECAST_UNAVAILABLE)
      ↓
PROVENANCE GATE   (validated_metrics_apply == true for EVERY reservoir)
      ↓
MPCController.decide(network)                     ← validated Phase 15.3, UNMODIFIED
      │   action space = 6 gate levels ^ 4 reservoirs = 1296 JOINT candidates
      │   each candidate = ONE gate for each of A, B, C, D
      ↓  raw proposal (4 gates)
SafetyLayer.validate(proposal, current, [A, B, C, D])   ← validated Phase 15.3, UNMODIFIED
      ↓  validated_gates (4 gates)  — the ONLY thing applied
state_manager._apply_ai_control() → gate_commands
      ↓
ReservoirNetwork.step(inflows, gates)   ← 4 nodes processed in topological order A→B→C→D,
      │                                    releases routed with delay 2/1/1 and attenuation .90/.85/.80
      ↓
state  (4 reservoirs + cascade inventory + control provenance + action_space)
      ↓
FastAPI `/api/state` · WebSocket `/ws/state`
      ↓
Three.js Digital Twin  (R1 Anayirankal → R2 Ponmudi → R3 Idamalayar → R4 Idukki → DS)
```

There is exactly **one** such path. The browser cannot write state
(`POST/PUT/DELETE /api/state` → 405), the GNN is advisory-only (the orchestrator
imports nothing GNN-related), and the legacy rule-based `ForecastAwareController`
is not imported or instantiated by the authoritative state manager (AST-checked).

---

## 3. Proof Reservoir D Is No Longer Pinned

**What was actually there before Stage 9** — two separate legacy surfaces, both found
by inspection, not assumption:

| # | Legacy surface | Why it was a pin |
|---|---|---|
| 1 | `state_manager.manual_gates = {..., "Virtual Reservoir D": 100.0}` | D was hardcoded **fully open** while A/B/C were 40/35/50. |
| 2 | `routes.py` gate mapping = `reservoir_1..3` only | `POST /api/gate/reservoir_4` → **400**, so D's value could not be changed by the UI/API at all. Together with (1) this made D a *permanently fixed* gate decision. |

**After Stage 9:**

| Evidence | Test |
|---|---|
| The 100.0 pin is gone from the code and D has an explicit ordinary baseline (50.0). | `test_reservoir_d_manual_baseline_is_not_the_legacy_100_percent` |
| `reservoir_4` is a valid command id (`RESERVOIR_ID_TO_NODE`) and the command reaches the store. | `test_reservoir_d_is_commandable_through_the_api` |
| **Behavioural non-pinning proof:** two different commands to D (10 %, then 73 %) produce two different authoritative gate positions, in both the payload and the physics (`ReservoirNetwork` node `Virtual Reservoir D`). | `test_reservoir_d_gate_is_not_permanently_fixed` |
| D receives the joint safety-validated action exactly like every other reservoir, including the rate limit (0.9 → 0.5) and clamp paths. | `test_unsafe_mpc_proposal_is_sanitised_before_reaching_reservoir_network` (Stage 8), `test_four_reservoir_proposal_through_safety_reaches_the_physics` (Stage 9) |
| **No special-case D controller**: no `if nid == "Virtual Reservoir D"` branching anywhere in the orchestrator, MPC or SafetyLayer; the only place D is named explicitly is the documented forecast inventory. | `test_no_special_case_controller_for_reservoir_d` |
| The legacy `ForecastAwareController` is not authoritative. | `test_legacy_forecast_aware_controller_is_not_authoritative` |
| D is no longer dropped from the twin payload: `reservoir_4` exists with `repository_name = "Idukki"` and `is_terminal = true`. | `test_every_twin_reservoir_carries_its_authoritative_identity` |

---

## 4. Proof All Four Dimensions Reach the MPC

Read from the validated controller — not assumed:

* `node_ids = network.processing_order` → `['Virtual Reservoir A', 'Virtual Reservoir B',
  'Virtual Reservoir C', 'Virtual Reservoir D']` (the authoritative network, not a list).
* `gate_combos = itertools.product(self.config.gate_levels, repeat=len(node_ids))` with
  `gate_levels = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]` → **6⁴ = 1296 joint candidates**.
* `candidate_gates_step0 = {nid: g for nid, g in zip(node_ids, combo)}` → every candidate
  carries a gate for **all four** reservoirs.

| Evidence | Result | Test |
|---|---|---|
| The MPC is handed the four-node authoritative network, in order A→B→C→D | ✅ | `test_all_four_reservoirs_reach_the_mpc` |
| The final action contains four gate decisions (fractions, percent, raw-proposal audit and `per_node`) | ✅ | `test_final_action_contains_four_gate_decisions` |
| The reported action space is 4-D: `dimension = 4`, `candidate_vectors = 1296`, `coordinated = true` | ✅ | `test_action_space_is_four_dimensional` |
| The MPC **scored** the complete 4-D product (`candidates_evaluated == 1296`) | ✅ | `test_mpc_scored_the_complete_action_vector` |
| **Coordination proof:** the set of candidate vectors the MPC actually simulated is *exactly* the Cartesian product of the gate levels over the four nodes, and every candidate has four keys | ✅ `observed == expected`, all 1296 four-dimensional | `test_candidate_search_covers_the_joint_cartesian_product` |
| Each candidate is rolled out on the authoritative `ReservoirNetwork` and the **live network is never mutated** (storage compared before/after every rollout) | ✅ | `test_mpc_simulates_candidates_on_the_authoritative_network` |
| Deterministic candidate enumeration order | ✅ | `test_candidate_enumeration_order_is_deterministic` |

Because the search enumerates the *joint* product and scores each vector by rolling
the whole cascade forward, the controller cannot be "four independent controllers":
the cost of A's choice is only knowable together with B's, C's and D's.

### 4.1 Upstream/downstream relationships are represented through `ReservoirNetwork`

| Evidence | Result | Test |
|---|---|---|
| A's action arrives at B **exactly 2 days later** (validated A→B delay) and nothing arrives before that | ✅ | `test_upstream_action_changes_downstream_state_according_to_the_physics` |
| A's action propagates through the whole chain: B's release diverges at t=3, C's release at t=4, **D's storage at t=5** — i.e. exactly 2 then 1 then 1 steps | ✅ | `test_upstream_action_propagates_all_the_way_to_the_terminal_reservoir` |
| The magnitude of the first arrival at B equals A's release × 0.90 (validated attenuation) | ✅ | same |
| D is the terminal node and drives the downstream flow | ✅ | `test_terminal_reservoir_is_idukki_and_drives_the_downstream_flow` |

---

## 5. Proof the SafetyLayer Receives All Four

| Evidence | Test |
|---|---|
| `SafetyLayer.validate()` is invoked exactly **once** with `node_ids == [A, B, C, D]`, a proposal containing all four gates and a current-gate map containing all four | `test_safety_layer_receives_the_complete_four_reservoir_proposal` |
| The layer returns a validated gate for every reservoir | `test_safety_layer_returns_a_gate_for_every_reservoir` |
| The layer is the validated `src/controller/safety.py` implementation, configured with the MPC's own `max_gate_change` | `test_the_safety_layer_used_is_the_validated_implementation` (Stage 8) |
| Safety is applied to the **complete coordinated action** — one call carrying all four gates, so a violation on any reservoir corrects the same proposal that carries the others | Stage 8 `test_unsafe_mpc_action_is_constrained_by_the_safety_layer` |
| The layer's guarantees are unchanged and its known gaps are still reported, not silently filled | Stage 8 `test_release_and_downstream_capacity_checks_are_not_implemented` |

**SafetyLayer checks that remain NOT implemented** (reported in Stage 8, still true and
**not** added in Stage 9): release bounds, downstream-capacity constraints, storage
headroom, minimum environmental flow, prior-state validation, impossible-state
detection. The `EMERGENCY` status is still unreachable through `validate()`.

---

## 6. Proof `ReservoirNetwork` Receives All Four Safe Actions

| Evidence | Test |
|---|---|
| After an AI step with an injected validated bundle, **every** reservoir's `state.gate_position` equals the safety-validated decision value | `test_network_receives_the_safety_output_for_all_four_nodes` |
| Commanding each of the four reservoirs through the API moves exactly that reservoir's physical gate | `test_every_reservoir_gate_reaches_the_physics` |
| The required behavioural integration test: an unsafe **four-reservoir** proposal `{A 5.0, B −3.0, C NaN, D 0.9}` → SafetyLayer → applied `{A 0.5, B 0.0, C 0.1, D 0.5}` → `ReservoirNetwork` gate positions equal that vector for all four nodes → each node's `controlled_release == gate × max_release` → mass balance still holds | `test_four_reservoir_proposal_through_safety_reaches_the_physics` |
| The raw proposal did **not** reach the physics (A ≠ 1.0, D ≠ 0.9) | same |
| Coordinated control moves **more than one** reservoir, and `per_node` covers all four | `test_coordinated_control_moves_more_than_one_reservoir` |
| A blocked (demonstration / missing-D) forecast leaves all four gates untouched | Stage 8 `test_demo_forecast_end_to_end_leaves_gates_untouched` |

---

## 7. Topology / Physics Verification

Verified against `src/network_env/topology_config.yaml` (the single authority) **and**
against the live `ReservoirNetwork` built from it. **Nothing was changed.**

| Property | Value | Test |
|---|---|---|
| Processing order | `A → B → C → D` | `test_validated_topology_is_a_b_c_d_with_d_terminal` |
| Connections | exactly A→B, B→C, C→D (no extra, no missing) | same |
| Terminal node | `Virtual Reservoir D` (Idukki) | same, `test_terminal_reservoir_is_idukki_and_drives_the_downstream_flow` |
| Routing delay A→B | 2 days | `test_validated_routing_parameters_are_unchanged` |
| Routing delay B→C | 1 day | same |
| Routing delay C→D | 1 day | same |
| Attenuation A→B / B→C / C→D | 0.90 / 0.85 / 0.80 | same |
| Downstream capacity | 50.0 MCM/day | `test_downstream_capacity_is_unchanged` |
| Mass balance over 10 steps of four-reservoir control | residual < 1e-9 | `test_mass_balance_holds_under_four_reservoir_control` |
| Spill from non-terminal nodes leaves the network (not routed downstream) | ✅ | `test_non_terminal_spill_is_not_routed_downstream` |
| `reservoir_network.py` contains no controller/safety logic | ✅ | `test_the_physics_source_is_unmodified` |

### 7.1 Documented discrepancy (reported, NOT "fixed")

| Observation | Detail |
|---|---|
| **The 3D scene renders three basins, not four.** | The Three.js world was laid out for a three-basin valley (`BASINS` at `z = −290, 0, +290`, valley extent `±450`) with a downstream pool. All four reservoirs are now first-class in the *state, forecast, MPC, SafetyLayer, network transition, WebSocket payload and HUD*, and the HUD topology chain is `R1→R2→R3→R4→DS`, but Reservoir D (Idukki) has **no 3D mesh**. Adding it would require extending the valley and re-deriving every terrain height and camera preset — a rendering redesign, which Stage 9 was explicitly told not to perform. **Reported as a limitation (§12.1), not silently claimed as done.** |
| **The old twin labels did not match the validated mapping.** | The floating labels and telemetry cards read `Anathode / Idamalayar / Idukki` for `reservoir_1/2/3`, i.e. the twin displayed Reservoir A's data under a name that belongs to another reservoir, and Idukki's name was attached to Reservoir C. The names now come from the authoritative payload (`Anayirankal / Ponmudi / Idamalayar / Idukki`), and the fourth card is explicitly `R4 Idukki`. |
| **The live forecast pipeline excludes Reservoir D.** | `LIVE_FORECAST_EXCLUDED = ("Virtual Reservoir D",)`. This is *pre-existing, accepted* Stage 6 behaviour (`test_stage6` pins "the live sim forecasts only A/B/C" and that D is `UNAVAILABLE`), and it is the reason the coordinated MPC is blocked. Stage 9 made it **explicit and named** instead of a bare `continue`, and did **not** start fabricating a D forecast. |

---

## 8. Test Counts

| Suite | Result |
|---|---|
| **Stage 9 tests** (`tests/test_stage9_four_reservoir_coordination.py`) | ✅ **43 passed** |
| **Complete test suite** | ✅ **469 passed, 0 failed** (75.1 s) |
| Stage 3 / 4 / 5 / 6 / 7 / 8 / 9 | ✅ 28 / 41 / 38 / 56 / 40 / 38 / 43 (all EXIT=0) |
| Stage 3+4+5+6+7+8+9 combined | ✅ **284 passed, 0 failed** |

Suite growth: 426 (end of Stage 8) → **469** (Stage 9, +43).

### 8.1 Coverage against the 17 requested test areas

| Req. | Requirement | Stage 9 test(s) |
|---|---|---|
| 1 | All four reservoirs reach MPC | `test_all_four_reservoirs_reach_the_mpc` |
| 2 | D is not pinned at 100 % | `test_reservoir_d_manual_baseline_is_not_the_legacy_100_percent`, `test_reservoir_d_is_commandable_through_the_api`, `test_reservoir_d_gate_is_not_permanently_fixed`, `test_no_special_case_controller_for_reservoir_d` |
| 3 | Final action contains four gate decisions | `test_final_action_contains_four_gate_decisions` |
| 4 | Candidate/action representation has four dimensions | `test_action_space_is_four_dimensional`, `test_mpc_scored_the_complete_action_vector`, `test_candidate_search_covers_the_joint_cartesian_product` |
| 5 | Upstream action can influence downstream state per the authoritative physics | `test_upstream_action_changes_downstream_state_according_to_the_physics`, `test_upstream_action_propagates_all_the_way_to_the_terminal_reservoir` |
| 6 | SafetyLayer receives the complete four-reservoir proposal | `test_safety_layer_receives_the_complete_four_reservoir_proposal`, `test_safety_layer_returns_a_gate_for_every_reservoir` |
| 7 | `ReservoirNetwork` receives the SafetyLayer output for all four nodes | `test_network_receives_the_safety_output_for_all_four_nodes`, `test_every_reservoir_gate_reaches_the_physics`, `test_four_reservoir_proposal_through_safety_reaches_the_physics` |
| 8 | Missing D forecast blocks coordinated MPC | `test_missing_reservoir_d_forecast_blocks_the_coordinated_mpc` |
| 9 | Demonstration forecast remains blocked | `test_demonstration_forecast_blocks_the_coordinated_mpc`, `test_no_zero_fill_average_or_carry_forward_for_reservoir_d` |
| 10 | No legacy `ForecastAwareController` is authoritative | `test_legacy_forecast_aware_controller_is_not_authoritative` |
| 11 | Browser cannot bypass the path | `test_browser_cannot_bypass_the_coordinated_path`, `test_only_the_four_known_reservoirs_are_commandable` |
| 12 | WebSocket state has all four reservoirs + controller provenance | `test_websocket_state_has_four_reservoirs_and_provenance`, `test_twin_payload_declares_the_four_reservoir_cascade`, `test_every_twin_reservoir_carries_its_authoritative_identity`, `test_twin_ui_represents_all_four_reservoirs` |
| 13 | Canonical units remain correct | `test_canonical_units_hold_for_all_four_reservoirs`, `test_no_gate_value_crosses_the_boundary_implicitly` |
| 14 | Deterministic behaviour remains deterministic | `test_four_reservoir_decision_is_deterministic`, `test_candidate_enumeration_order_is_deterministic` |
| 15 | Stage 3–8 tests remain green | ✅ 241/241 (28+41+38+56+40+38) — see §9 |
| 16 | Frozen artifacts unchanged | `test_stage9_does_not_modify_frozen_artifacts`, `test_validated_mpc_and_safety_are_unmodified` |
| 17 | Phase 15.3 reproduction identical | §11 |

Plus the required behavioural integration test:
`test_four_reservoir_proposal_through_safety_reaches_the_physics` (§6) and
`test_coordinated_control_moves_more_than_one_reservoir`.

### 8.2 Deliberate test updates carried with this stage

| File | Change | Justification |
|---|---|---|
| `tests/test_gate_unit_regression.py` | `reservoir_4` → `reservoir_5` in the "invalid reservoir id" parametrization | `reservoir_4` is now a **valid** command target. The guard's intent (an unknown id returns 400 and mutates nothing) is preserved and still exercised. |
| `tests/test_stage4_single_authoritative_simulation.py` | `reservoir_4` → `reservoir_5` in `test_unknown_reservoir_id_is_rejected_without_state_change` | Same reason. `reservoir_99`, `Virtual Reservoir D` and the SQL string are still asserted to be rejected. |

No other Stage 3–8 assertion was weakened, loosened or deleted; both edits were forced
by the deliberate Stage 9 change that makes Reservoir D commandable, and both are
called out here rather than made silently.

---

## 9. Stage 3–8 Regression

Every earlier stage suite was re-run unchanged (except the two documented id-list
edits above) and is green:

| Suite | Result |
|---|---|
| Stage 3 — live network authority | ✅ 28 passed |
| Stage 4 — single authoritative simulation | ✅ 41 passed |
| Stage 5 — forecast provenance | ✅ 38 passed |
| Stage 6 — forecast adapter | ✅ 56 passed |
| Stage 7 — live MPC integration | ✅ 40 passed |
| Stage 8 — SafetyLayer integration | ✅ 38 passed |
| **Total** | ✅ **241 passed, 0 failed** |

The Stage 8 invariants the user accepted remain intact: the SafetyLayer is still
between the MPC and `ReservoirNetwork`, raw MPC proposals still cannot bypass it,
and demonstration / missing-Idukki forecasts still block.

---

## 10. Frozen-Artifact Status

`scripts/stage3_verify_frozen_artifacts.py` → **exit 0 — `VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`**

| Artifact | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | ✅ MATCH |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | ✅ MATCH |
| `data/processed/scaled/feature_scaler.pkl` | ✅ read-only |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | ✅ MATCH (LF-normalised; the raw-hash difference is the known `core.autocrlf` working-tree artefact) |

Also hashed around a full four-reservoir orchestrator run inside the Stage 9 suite
(`test_stage9_does_not_modify_frozen_artifacts`).

---

## 11. Phase 15.3 Reproduction Status

`scripts/stage3_phase15_3_reproduction.py` re-run **shielded** (its `OUTPUT_DIR`
redirected, protected directory hash-guarded) → **exit 0**:

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

**The validated offline pipeline is bit-identical to before Stage 9.**

---

## 12. Remaining Limitations

| # | Limitation | Impact |
|---|---|---|
| 12.1 | **Reservoir D has no 3D mesh.** The Three.js scene renders three basins; D (Idukki) is first-class in the state, forecast contract, MPC, SafetyLayer, network, WebSocket payload, HUD cards and the `R1→R2→R3→R4→DS` topology chain, but is not drawn as a fourth basin. Extending the 3D world would change every terrain height and camera preset (a rendering redesign, out of Stage 9 scope). | Visual only; the data and control path are complete and tested. |
| 12.2 | **The live AI path is still BLOCKED.** All live forecasts remain `DEMONSTRATION_ONLY`, and Reservoir D still has no live forecast at all (`LIVE_FORECAST_EXCLUDED`), so the Stage 7 provenance gate refuses and the coordinated MPC is not invoked. | Intended, accepted behaviour ("if this means the live controller remains blocked under today's demo inputs, that is acceptable"). The coordination behaviour is proven with an injected validated bundle. |
| 12.3 | **The MPC's candidate space is enumerated by full Cartesian product**, so cost grows as `levels^4`. 1296 candidates × 3 rollout steps ≈ 0.66 s per decision. | Fine for a daily-timestep demo; not a real-time controller. Not optimized (Stage 9 was told not to optimize prematurely). |
| 12.4 | **The SafetyLayer still does not check release bounds, downstream capacity, storage headroom or minimum flow**, and `EMERGENCY` is unreachable through `validate()`. | Unchanged from Stage 8; reported there, **not** silently expanded here. |
| 12.5 | **The SafetyLayer is applied twice per step** (once inside the validated MPC, once at the live boundary). | Idempotent on the current validated path; the boundary pass is the authoritative one. |
| 12.6 | **D's live forecast absence is a policy, not a physics limit.** The live pipeline *could* attempt a D forecast (it would be labelled `DEMONSTRATION_ONLY` and would still block). Stage 9 deliberately did not change which reservoirs are forecast, because that is accepted Stage 6 behaviour and changing it would alter the verified "missing Idukki blocks" semantics. | Documented; changing it is a future-stage decision, not a Stage 9 one. |
| 12.7 | **`get_adapted_state()` still recomputes the ML pipeline on every read**, including every WebSocket broadcast. | Pre-existing since Stage 4; performance only. |
| 12.8 | **The behavioural integration tests cover a single step** (plus a 10-step physics-only mass-balance test). There is no multi-step closed-loop soak of the live AI path. | Correctness for a step is proven; long-run behaviour is not. |
| 12.9 | **`manual_gates` is still a per-reservoir hand-set baseline** (40/35/50/50). It is no longer *pinned*, and all four are commandable, but the values themselves remain `SIMULATION ASSUMPTION`s, not measured targets. | Documented in code. |
| 12.10 | **Two legacy three-reservoir assertions had to be updated** (§8.2). | Deliberate and reported; they are the only Stage 3–8 assertions touched. |

---

## 13. Hardware Status — Explicit Statement

> **Hardware is NOT connected.** Stage 9 is **software-only**.
>
> * No ESP32 / serial / MQTT / Modbus / actuator adapter was created or modified.
> * `src/hardware/` was **not** touched.
> * No GPIO, no sensor polling, no gate or valve actuation.
> * The twin payload still reports
>   `hardware_status = { esp32: NOT_CONNECTED, water_level_sensor: NOT_CONNECTED, flow_sensor: NOT_CONNECTED, gate_actuator: NOT_CONNECTED }`
>   (verified live against the running API in §14).

The design keeps the eventual hardware boundary **strictly downstream of the safety
decision**, and Stage 9 did not weaken that:

```
MPC  →  SafetyLayer  →  Hardware / Simulation command boundary  →  ReservoirNetwork (today) | Actuator (future)
```

Because in AI mode the four gate commands are written to `ReservoirNetwork` by exactly
one place (`state_manager._apply_ai_control`, from `decision.gate_positions_pct` =
SafetyLayer output), a future hardware adapter placed at that same boundary inherits
the same four-reservoir safety guarantee. **The hardware adapter was deliberately not
implemented.**

---

## 14. Browser / Live-Server Validation Performed

In addition to pytest, the twin was exercised against a live server
(`uvicorn src.dashboard.api.app:app`, port 8123) in the integrated browser:

| Check | Result |
|---|---|
| Page loads with **four** telemetry cards | ✅ `R1 Anayirankal`, `R2 Ponmudi`, `R3 Idamalayar`, `R4 Idukki` (+ `DS Downstream`) |
| Topology chain shows four reservoirs | ✅ `R1 → R2 → R3 → R4 → DS` |
| Fourth gate slider exists | ✅ `G4 = 50` |
| Reservoir names come from the payload | ✅ floating labels read `Anayirankal / Ponmudi / Idamalayar` |
| DEMONSTRATION badge actually appears (Stage 9 fix) | ✅ `#demo-badge.on` visible |
| `/api/state` returns four reservoirs with authoritative names | ✅ `["Anayirankal","Ponmudi","Idamalayar","Idukki"]` |
| `cascade` block present | ✅ `count = 4`, `terminal = Idukki` |
| `action_space` present in the control block | ✅ |
| Hardware still disconnected | ✅ all four `NOT_CONNECTED` |
| **The page issues no write to the simulation on load/reload** (display-only preserved) | ✅ gate store byte-identical before and after a full reload |

---

## Appendix A — Requirement Compliance Summary

| Requirement | Status |
|---|---|
| All four reservoirs first-class in state / forecast / MPC / SafetyLayer / network transition / WebSocket state / Digital Twin | ✅ §2, §4, §5, §6, §12.1 (3D mesh gap reported) |
| Eliminate the legacy "D permanently pinned at 100 %" behaviour | ✅ §3 — two legacy surfaces removed |
| D must not be hardcoded as a fixed gate decision | ✅ §3 |
| Do not reintroduce `ForecastAwareController`; no special-case D controller | ✅ §3 |
| Prove the MPC optimizes the complete four-reservoir decision | ✅ §4, §4.1 — joint Cartesian product, 1296 four-dimensional candidates, all scored |
| Do NOT redesign / modify the validated MPC algorithm | ✅ §1 (objective, grid, optimization untouched; source pinned) |
| Keep the Stage 7 provenance gate; all four forecasts must be eligible; demonstration stays blocked; missing D stays blocked; no zero-fill / averaging / carry-forward / fabrication | ✅ §5, §8.1 (reqs 8–9); `test_no_zero_fill_average_or_carry_forward_for_reservoir_d` |
| Live controller may remain blocked — do not weaken the gate to make the twin move | ✅ §12.2 — the gate was not weakened |
| Keep `MPC → SafetyLayer → ReservoirNetwork`; do not bypass; no second safety layer; do not modify the SafetyLayer | ✅ §5; `safety.py` and `mpc_controller.py` unmodified |
| Report missing safety guarantees rather than expanding scope | ✅ §5 (release bounds, downstream capacity, storage, min flow, unreachable `EMERGENCY` still reported) |
| Use the authoritative `ReservoirNetwork`; do not silently change delays, attenuation, storage equations, spill, mass balance or downstream capacity; report discrepancies | ✅ §7, §7.1 — nothing changed; the 3D-scene and label discrepancies are reported |
| Do not implement hardware, but keep the hardware boundary downstream of the safety decision | ✅ §13 |
| Do not implement RL/MARL, do not change the GNN role | ✅ GNN still advisory-only (`test_gnn_remains_advisory_only`) |
| Do not modify frozen Phase 15.3 artifacts | ✅ §10, §11 |
| Do not perform Stage 10 work; stop after Stage 9 | ✅ §15 |

---

## 15. Stage 10 Status — Explicit Statement

> **Stage 10 has NOT started.**
>
> * ❌ Stage 10 not started — no work beyond the Stage 9 boundary.
> * ❌ Hardware not implemented (no actuator adapter, `src/hardware/` untouched).
> * ❌ RL / MARL not implemented.
> * ❌ GNN role not changed (advisory-only; still cannot influence control).
> * ❌ MPC not redesigned — objective, candidate grid and optimization logic untouched.
> * ❌ Frozen Phase 15.3 artifacts not modified (SHA256-verified).
> * ❌ No FutureWarning/DeprecationWarning cleanup, no refactor, no unrelated feature work.
>
> Work stops here, at the Stage 9 boundary.

---

*End of Stage 9 Report. Stopped at the Stage 9 boundary — Stage 10 not started.*
