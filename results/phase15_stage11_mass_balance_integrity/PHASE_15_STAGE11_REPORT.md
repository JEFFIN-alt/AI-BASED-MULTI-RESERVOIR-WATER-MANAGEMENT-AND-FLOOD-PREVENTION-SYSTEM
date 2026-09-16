# PHASE 15 — STAGE 11 REPORT
## Authoritative LIVE Mass-Balance Integrity

**Status:** COMPLETE — Stage 11 only. **Stage 12 has NOT been started.**
**Date:** 2026-09-15
**Scope:** establish that every authoritative LIVE simulation step satisfies the
conservation principles already implemented by the validated `ReservoirNetwork`,
that the check is performed on the state produced by the action that was
**actually applied**, and that the truthful result reaches the API, the
WebSocket and the Digital Twin.

---

## 1. Exact files changed

### Created

| File | Purpose |
|---|---|
| `src/network_env/mass_balance.py` | `MassBalanceMonitor` — the read-only per-step auditor of the validated physics |
| `tests/test_stage11_mass_balance_integrity.py` | 47 Stage 11 tests |
| `scripts/run_stage11_mass_balance_integrity.py` | Evidence + overhead measurement |
| `results/phase15_stage11_mass_balance_integrity/stage11_mass_balance_evidence.json` | Machine-checkable evidence |
| `results/phase15_stage11_mass_balance_integrity/PHASE_15_STAGE11_REPORT.md` | This report |

### Modified (live-path integration only — no physics, no controller)

| File | Change |
|---|---|
| `src/network_env/live_cascade_adapter.py` | `step()` now performs the authoritative step **through** `MassBalanceMonitor.step_and_check(...)`; added `mass_balance_diagnostic()`; `reset()` clears the monitor |
| `src/dashboard/sim_bridge.py` | `step(..., action_source=...)` passthrough; `mass_balance_diagnostic()`; `get_state()` now carries a `mass_balance` block |
| `src/dashboard/api/state_manager.py` | names the applied action's provenance, records `_verify_applied_action()`, and publishes the `mass_balance` block on every state payload |
| `src/dashboard/twin_component/state_adapter.py` | `_mass_balance_block()` (never invents a PASS) + `mass_balance` key in the twin payload |
| `src/dashboard/web/index.html` | four truthful HUD rows: `MASS BALANCE`, `MB RESIDUAL`, `MB CHECKED`, `MB RESERVOIRS` |

### Explicitly NOT touched (verified by test)

`src/network_env/reservoir_network.py`, `src/controller/safety.py`,
`src/controller/mpc_controller.py`,
`src/controller/downstream_capacity_guard.py`,
`src/controller/live_mpc_orchestrator.py`,
`models/lstm_pytorch_v3_logtarget/*`, `results/phase15_v3_validation/*`,
`results/lstm_pytorch_v3_logtarget/*`.

`test_frozen_physics_implementation_is_untouched` asserts that none of the
frozen/validated components contains a Stage 11 hook.

---

## 2. Exact mass-balance equation / semantics from the authoritative model

Read out of `reservoir_network.py` (not assumed).

### Per reservoir, per step (`ReservoirNode.step`)

```text
gate_clamped       = clamp(gate_position, 0, 1)
requested_release  = gate_clamped * max_release
total_inflow       = local_inflow + routed_inflow
available          = storage + total_inflow
controlled_release = min(requested_release, available)
preliminary        = available - controlled_release
spill              = max(0, preliminary - capacity);  preliminary -= spill
storage            = min(preliminary, capacity)
total_outflow      = controlled_release + spill
```

Audited invariant (identical to the model's own documented law):

```text
new_storage - old_storage
    = inflow_local + inflow_routed - controlled_release - spill
```

### Per connection, per step (`ReservoirNetwork.step`)

```text
raw_arriving       = queue.popleft()          # the OLDEST queued volume
attenuated_arrival = raw_arriving * attenuation
transmission_loss  = raw_arriving * (1 - attenuation)
queue.append(today's controlled release of the source)
```

Audited invariants:

```text
raw_arriving = attenuated_arrival + transmission_loss
queue_after  = queue_before[1:] + [released_this_step]          # exact FIFO identity
raw_arriving = controlled_release_of_source(step - delay)       # delayed, never instantaneous
routed_inflow(destination) = Σ attenuated_arrival of its incoming connections
```

### Network level, per step

```text
total_external_inflow = storage_change + terminal_outflow + nonterminal_spill
                      + routing_loss + change_in_transit
```

This is the **incremental form of the network's own cumulative
`mass_balance_check()`**, pinned by
`test_incremental_equation_agrees_with_the_networks_own_cumulative_check`.

> **Double-count guard.** The terminal node's spill is already inside
> `terminal_outflow`, so it is reported separately (`terminal_spill_mcm`) and is
> **not** added again. Adding it again would produce a phantom residual equal to
> the terminal overflow — pinned by
> `test_spill_case_nonterminal_and_terminal`.

---

## 3. Time-step convention

`ReservoirNetwork.step()` increments `timestep` by **1 day** and the model is
documented as a *discrete-time daily mass balance*. There is **no `dt` scaling
anywhere** in `reservoir_network.py` (no `* dt`, no `/ dt`), so a flow of
`X MCM/day` moves exactly `X MCM` during one step.

The audit therefore performs each per-step check in **MCM per step** and reports
`timestep_days: 1.0` next to every residual, so the daily-flow ↔ per-step-volume
relationship is explicit rather than implicit. No unit is converted inside the
monitor; it compares the model's own numbers in the model's own units.

---

## 4. Spill / routing treatment

* **Spill** is booked separately from the controlled release (never lumped in).
* **Non-terminal spill** leaves the network and is never routed downstream →
  its own term `nonterminal_spill`.
* **Terminal spill** leaves the network and is already inside `terminal_outflow`
  → reported separately, never added twice (see §2).
* **Attenuation loss** `(1-α)·raw_arriving` is booked in full as `routing_loss`
  and reported per connection — the model does not silently destroy water, and
  neither does the audit.
* **Water in transit** (routing queues) is real state and appears as
  `change_in_transit`, so nothing is double-counted and nothing is lost between
  steps.
* **Delayed routing is never treated as instantaneous.** `inflow_routed` at a
  node is the *delayed, attenuated arrival*; the audit proves it three ways: the
  exact FIFO queue identity, the monotone `delay_history_ok` check against the
  monitor's own release history, and the equality between a node's reported
  `inflow_routed` and the sum of its incoming connections' attenuated arrivals.

Routing parameters are read from the live connections (never hardcoded) and
pinned by `test_routing_parameters_are_the_validated_ones`:
delays **2 / 1 / 1**, attenuation **0.90 / 0.85 / 0.80** — unchanged.

---

## 5. Exact residual / tolerance

| Item | Value |
|---|---|
| Tolerance | `MASS_BALANCE_TOLERANCE_MCM = 1e-9` **MCM** (single explicit constant) |
| Reported residual | worst absolute residual over **all** invariants of the step (per-reservoir, routed-inflow, routing, network) |
| Measured noise floor | `2.84e-14` MCM worst case over a 12-step live run |
| Gate-application tolerance | `GATE_TOLERANCE = 1e-9` (fraction) |
| Applied-vs-final-action tolerance | `ACTION_MATCH_TOLERANCE_PERCENT = 1e-9` (percent) |

`PASS` requires: a check actually ran, all four reservoirs were diagnosable,
every residual within tolerance, and no non-finite value.
`test_residual_tolerance_is_explicit_and_meaningful` pins that a `1e-12` MCM
perturbation passes while a `1e-6` MCM error fails.

---

## 6. Proof that all four reservoirs are checked

* `reservoirs_expected == reservoirs_checked == 4`, and the audited rows equal
  `network.processing_order` — `test_all_four_reservoirs_are_checked`.
* Every row carries the **same** key set (no terminal-specific / D-specific
  branch) and each row contains `storage_before/after`, `storage_change`,
  `inflow_local`, `inflow_routed`, `controlled_release`, `spill`,
  `total_outflow`, `commanded_gate_fraction`, `applied_gate_position`,
  `residual_mcm`, `within_tolerance`.
* `test_reservoir_d_is_not_pinned_or_special_cased`: D is audited like the
  others and its gate is reported **as applied** (0.37 → 0.37); the audit never
  writes it. `test_the_audit_never_mutates_the_network` proves the audit leaves
  storages, queues and gates untouched.

Evidence (12-step live run, `stage11_mass_balance_evidence.json`), per-reservoir
residuals — A = Anayirankal, B = Ponmudi, C = Idamalayar, D = Idukki:

| step | A residual | B residual | C residual | D residual | network residual | status |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | -2.84e-14 | 0.0 | 2.84e-14 | PASS |
| 2 | 0.0 | 0.0 | 2.84e-14 | 0.0 | -1.42e-14 | PASS |
| 3 | 0.0 | -8.88e-16 | 2.84e-14 | 0.0 | -1.42e-14 | PASS |
| … | … | … | … | … | … | PASS |
| 12 | 0.0 | 0.0 | 2.84e-14 | 0.0 | -1.42e-14 | PASS |

`max |residual| = 5.684e-14 MCM` — 4.5 orders of magnitude inside tolerance.

---

## 7. Proof that the actual final applied action is used

The audit cannot be performed on anything other than the applied action **by
construction**: `LiveCascadeAdapter.step()` no longer calls `network.step()`
directly — the only call site is inside `MassBalanceMonitor.step_and_check()`,
which hands the *same two dicts* to `ReservoirNetwork.step()` that it then audits.
`test_the_live_step_cannot_bypass_the_audit` pins this
(`"self.network.step("` is absent from the adapter).

Three independent confirmations on the real live path
(`test_e2e_validated_forecast_to_mass_balance`), with the raw MPC proposal
deliberately set to a flood (`1.0` on every gate):

| Quantity (Reservoir D) | Value |
|---|---|
| raw MPC proposal | `1.0` |
| SafetyLayer output (rate-limited) | `0.5` |
| **FINAL_SAFE_CONTROL_ACTION** (downstream-corrected) | **`0.25`** |
| gates passed to `ReservoirNetwork.step()` (spied) | `0.25` |
| `network.nodes[D].state.gate_position` | `0.25` |
| `audit["applied_action_fraction"][D]` | `0.25` |
| audited `controlled_release_mcm_day` | `0.25 × 200 = 50.0` MCM/day |

`test_actual_final_applied_action_is_used_not_the_raw_proposal` asserts all three
actions differ and that the audited physics belongs to the **applied** one.
`test_downstream_guard_modification_reaches_the_mass_balance_record` asserts the
audit names `DOWNSTREAM_CAPACITY_GUARD` and that D's audited terminal outflow is
exactly the 50.0 MCM/day capacity the guard enforced.

`state_manager._verify_applied_action()` additionally compares the gates written
to the network with `final_safe_control_action_pct` and publishes
`matches_final_safe_control_action` (`True` in AI mode, `None` —
"not applicable" — in MANUAL mode, never a hopeful `True`).

**The E2E chain covered:** validated forecast → provenance gate → MPC →
SafetyLayer → DownstreamCapacityGuard → FINAL_SAFE_CONTROL_ACTION →
`ReservoirNetwork.step()` → mass-balance check.

---

## 8. Deliberate corruption detection result

The diagnostic is **not a constant PASS**. Corruption probes
(`corruption_probe` in the evidence JSON):

| Probe | Result |
|---|---|
| clean step (baseline) | `PASS`, residual `0.0` |
| storage corrupted by `+0.5` MCM after the step | `VIOLATION`, residual `+0.5000000000000142` MCM, `state_valid = False` |
| reported release corrupted by `+1.25` MCM/day | `VIOLATION`, reservoir residual `+1.25`, network residual `-1.25` |
| routing queue corrupted by `+2.0` MCM | `VIOLATION` (`ROUTING_MASS_BALANCE_VIOLATION`) |

Corruption is **never repaired**: the state read by the audit is left exactly as
it was (`test_corruption_is_never_repaired_by_the_audit`), and
`test_fail_safe_gap_is_reported_rather_than_invented` asserts the audit region
contains no write to the live network, no `queue.append`/`popleft`, no
`network.reset()` and no second `network.step()`.

**Failure handling.** On violation the audit: reports the offending residuals,
marks `state_valid = False`, records `ever_violated = True` /
`violation_count` (sticky — a later passing step cannot hide it), and does not
alter storage, inflow, outflow or spill. The **fail-safe gap is reported, not
invented**:

```text
fail_safe     = "NONE_DEFINED_IN_EXISTING_ARCHITECTURE"
fail_safe_gap = "The validated architecture (ReservoirNetwork / SafetyLayer /
                 MPC / DownstreamCapacityGuard) defines no response to a
                 mass-balance violation … Stage 11 therefore REPORTS the
                 violation, marks the state invalid and never repairs it. It
                 does not invent a fail-safe, does not halt the loop and does
                 not suppress the diagnostic."
```

The monitor deliberately does **not** raise into the control loop: raising would
let a numerical fault silently stop the twin instead of being displayed by it.

**NaN / Inf behaviour.** Non-finite state or flows produce
`VIOLATION` with a `NON_FINITE_STATE_DETECTED` reason and a populated
`non_finite` list; nothing is clamped or "fixed"
(`test_nan_state_is_detected_safely`, `test_infinite_flow_is_detected_safely`).
The transport form of the diagnostic (`to_dict()`) renders non-finite floats as
strings, so a NaN state can still be **reported** over the WebSocket instead of
taking the broadcast down with it.

---

## 9. API / WebSocket / Digital Twin diagnostics

### Structured diagnostic (exposed everywhere)

```text
mass_balance:
    status              PASS | VIOLATION | NOT_CHECKED
    checked             true only after a check actually ran
    residual            worst absolute residual (MCM)
    tolerance           1e-09
    unit / flow_unit    MCM / MCM/day
    timestep_days       1.0
    timestep, step_index, timestamp
    reservoirs_checked / reservoirs_expected
    per_reservoir[]     node_id, storage_before/after_mcm, storage_change_mcm,
                        inflow_local/inflow_routed_mcm_day, controlled_release_mcm_day,
                        spill_mcm, total_outflow_mcm_day, commanded_gate_fraction,
                        applied_gate_position, residual_mcm, within_tolerance, terminal
    routing[]           connection, delay_days, attenuation, released_this_step,
                        raw_arriving, attenuated_arrival, transmission_loss,
                        residual, queue_evolution_ok, delay_history_ok
    network             total_external_inflow, storage_change, terminal_outflow,
                        terminal_controlled_release, terminal_spill, nonterminal_spill,
                        routing_loss, water_in_transit before/after, change_in_transit,
                        residual_mcm, within_tolerance, routing_delays_days,
                        attenuation_factors, cumulative_residual_mcm
    applied_action_fraction / applied_action_percent / applied_action_source
    applied_inflows_mcm_day, action_fully_applied, gate_clamped[]
    non_finite[], violations[], reason
    state_valid, ever_violated, violation_count, checks
    fail_safe, fail_safe_gap
    equation, routing_equation, network_equation, physics, latency_ms, telemetry
```

* **API** — `GET /api/state` (and therefore `POST /api/simulation/step`, play,
  pause, reset) carries the block; verified by
  `test_api_exposes_mass_balance_diagnostics`.
* **WebSocket** — `/ws/state` sends the same payload
  (`test_websocket_state_carries_mass_balance`).
* **Digital Twin** — `mass_balance` is a top-level twin key. `PASS` is shown only
  when `checked === true`; the status string is rendered **verbatim** (no
  calculation in JavaScript — asserted by
  `test_twin_js_contains_no_mass_balance_calculation`); the page parses
  (`test_twin_js_parses`).
* **No audit yet** → `NOT_CHECKED`, `checked: false`, `residual: null` — never
  `PASS` (`test_twin_payload_never_implies_conservation_without_an_audit`,
  `test_never_reports_pass_before_a_check_has_run`).
* **No browser state injection**: `POST`/`PUT /api/state` → `405`; forged
  `mass_balance` payloads on the command endpoints are ignored
  (`test_browser_cannot_inject_mass_balance_state`).
* **Demonstration forecast still blocked** (and still audited on the held action,
  `applied_action_source = HELD_CURRENT_GATES`); **missing Idukki forecast still
  blocked**.

Twin HUD rows added: `MASS BALANCE`, `MB RESIDUAL` (residual / tolerance, MCM),
`MB CHECKED` (YES/NO), `MB RESERVOIRS` (**4 CHECKED**).

---

## 10. Tests and counts

| Suite | Result |
|---|---|
| `tests/test_stage11_mass_balance_integrity.py` | **47 passed** |
| Full suite | **556 passed** (509 baseline + 47) |
| Stage 3–10 regression | **324 passed** (identical to the accepted Stage 10 count) |

Stage 11 test coverage maps to the required list: one-step (1), multi-step (2),
all four reservoirs (3), zero-flow steady state (4), normal inflow/release (5),
spill incl. the terminal double-count guard (6), delayed routing (7, 8),
terminal downstream flow (9), applied FINAL_SAFE_CONTROL_ACTION (10, 15),
raw proposal not used (11), downstream-guard modification reflected (12),
NaN/Inf (13), corrupted state (14), corrupted flow (26 + routing queue),
tolerance (16), API/WebSocket (17), truthful twin + UI (18, 21, 22),
demonstration forecast blocked (19), missing Idukki blocked (20), no browser
injection, GNN advisory-only, frozen artifacts (24), Phase 15.3 manifest (25).

Anti-tautology: `test_the_diagnostic_is_not_a_constant_pass` and the
`corruption_probe` evidence both require the audit to FAIL when it should.

---

## 11. Stage 3–10 regression

`324 passed` across
`test_stage3_live_network_authority.py`, `test_stage4_single_authoritative_simulation.py`,
`test_stage5_forecast_provenance.py`, `test_stage6_forecast_adapter.py`,
`test_stage7_live_mpc_integration.py`, `test_stage8_safety_layer_integration.py`,
`test_stage9_four_reservoir_coordination.py`, `test_stage10_downstream_capacity.py`
— unchanged. Notably the Stage 3 parity harness and the Stage 4 single-authoritative-
simulation invariants still hold even though every live step now runs through the
auditor.

## 12. Frozen-artifact status

`py scripts/stage3_verify_frozen_artifacts.py` →
**`VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`**

| artifact | raw | LF-normalised |
|---|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | MATCH | n/a (binary) |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | MATCH | MATCH |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | CRLF-diff only | MATCH |

## 13. Phase 15.3 reproduction status

`py scripts/stage3_phase15_3_reproduction.py` →
**`VERDICT: protected artifacts untouched; reproduction complete.`**

* 6/6 protected files `UNCHANGED`; 3/3 compared files **byte-IDENTICAL**.
* Baseline and MPC mass-balance residuals: `-6.82e-13` (unchanged).
* `V3 integrity: PASSED`.

---

## 14. Performance overhead

Measured with `MassBalanceMonitor.performance_overhead()` (300 iterations,
cloned network — the live network is never advanced by a measurement):

| | ms/step |
|---|---|
| bare `ReservoirNetwork.step()` | 0.0153 |
| step **with** the mass-balance audit | 0.0988 |
| **overhead** | **0.0836** |

Per-step audit latency on the live path: `~0.07–0.14 ms`.

Context: `MPCController.decide()` is ~660 ms and the Stage 10 downstream
boundary is 0.37–8.9 ms. The audit therefore adds **~0.013 %** of a `decide()`
cycle and cannot materially interfere with the control loop; it is the same
order as the safety layer's own validation. The test asserts `< 2 ms/step` and
`< 1 %` of a `decide()` cycle.

---

## 15. Remaining limitations

1. **No fail-safe exists for a conservation violation** (reported, not invented).
   The validated architecture defines none; Stage 11 marks the state invalid and
   reports the gap. A future stage must decide the response.
2. **The audit observes, it does not gate.** It runs *after*
   `ReservoirNetwork.step()`, so a violation is detected on the state that was
   produced, not prevented. Preventing it would require changing the frozen
   physics or the controller chain — out of scope here.
3. **`ReservoirState.transmission_loss` is never populated by the model** (it is
   always `0.0`; the routing loss lives on the connection). The audit books the
   per-connection loss itself and reports it; the unused field is documented as
   an observation, not "fixed" (that would mean editing frozen physics).
4. **A zero-delay connection has no dedicated delay proof.** The validated
   topology has delays 2/1/1, so this never arises; `expected_raw` is defined as
   `queue_before[0] if queue_before else 0.0`, which matches the implementation
   exactly, and `delay_history_ok` would report `None` rather than a false PASS.
5. **The Streamlit twin component (`src/dashboard/twin_component/reservoir_twin.html`)
   does not render the mass-balance block.** The live Digital Twin served by the
   FastAPI app (`src/dashboard/web/index.html`) does. Noted rather than silently
   half-wired.
6. **Corruption can only be detected, not attributed.** The audit reports *that*
   mass did not balance and *where* the residual appears; it does not attempt
   root-cause classification (e.g. distinguishing a sensor fault from a code
   defect). Deliberate: attribution logic would be new behaviour.
7. **`cumulative_residual_mcm` is informational only** and is not part of the
   verdict (it is only meaningful when the monitor has audited every step since
   the network was created).

---

## 16. Hardware — explicit statement

**No hardware is implemented and no hardware is connected.**

`telemetry: {source: "SIMULATION", hardware_connected: false, sensor_quality: null}`
is present in every diagnostic, exactly as the pre-existing
`hardware_status` block already reported `ESP32 / WATER LEVEL SENSOR / FLOW
SENSOR / GATE ACTUATOR = NOT_CONNECTED`. No ESP32, serial, sensor-driver or
actuator code exists in Stage 11.

The framework is nevertheless **suitable for later hardware integration**:
`MassBalanceMonitor.verify(network, snapshot, applied_inflows=…,
applied_gates_fraction=…)` takes the applied action and the applied inflows as
explicit inputs, so a future telemetry record can supply the same inputs and the
same integrity framework can compare **commanded vs actual plant behaviour**
(commanded vs applied gate, expected vs measured inflow/outflow/storage). Every
diagnostic already carries `timestamp`, per-reservoir storage/level, commanded
gate, applied gate, measured inflow, measured outflow, spill and a
`sensor_quality` slot (`None` today). Documented in `FUTURE_TELEMETRY_FIELDS`;
**not implemented**.

---

## 17. Scope statement

Stage 11 implemented mass-balance integrity **only**.

* No hardware was implemented.
* No RL / MARL was implemented.
* The GNN's role is unchanged (advisory only — asserted by test).
* The MPC, SafetyLayer and DownstreamCapacityGuard were **not** redesigned or
  edited.
* Frozen Phase 15.3 artifacts were not modified (verified by SHA256 and by
  byte-identical reproduction).
* **Stage 12 has NOT been started.**
