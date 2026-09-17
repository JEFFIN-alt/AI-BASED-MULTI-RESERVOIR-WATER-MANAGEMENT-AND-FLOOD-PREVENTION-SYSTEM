# STAGE 15 — END-TO-END ARCHITECTURE / INTEGRATION VERIFICATION — COMPLETE

**Status:** COMPLETE — Stage 15 only.
**Date:** 2026-09-16
**Goal:** prove the complete authoritative architecture works as one system, from
input → forecast → GNN advisory → MPC → SafetyLayer → downstream protection →
physics → mass balance → API/WebSocket → Digital Twin/Streamlit.

Stages 0–14 were treated as the protected baseline. Verification, not
optimisation: no component was redesigned, no model retrained, no threshold
invented.

---

## 1. Executive summary

The architecture **is** what the previous stages documented, with three
important qualifications that this stage measured rather than assumed:

1. **The live control path is intact and correctly ordered.** Instrumenting the
   real boundary methods recorded the call order `mpc.decide → safety.validate →
   safety.validate → downstream_guard.evaluate`. The SafetyLayer is invoked
   **twice** per decision (once inside `MPCController.decide()` on the gates that
   controller chose, once by the orchestrator on the proposal it returns). Both
   precede the capacity boundary, so the required ordering
   `MPC → SafetyLayer → DownstreamCapacityGuard → FINAL_SAFE_CONTROL_ACTION`
   holds — but the orchestrator's comment claiming the layer runs *"exactly ONCE
   per decision"* was **inaccurate**. That comment was corrected (minimum fix).

2. **In the live four-reservoir cascade the coordinated MPC never becomes
   eligible — by design.** The live forecasts are `DEMONSTRATION_ONLY`
   (simulation-derived inputs ⇒ `validated_metrics_apply = False`), **and**
   Reservoir D has no live forecast at all (`MISSING_FORECAST`, Stage 9 policy).
   The provenance gate therefore blocks the MPC and the system **holds the
   current gates**. That is the correct fail-safe behaviour, and it means the
   live AI-mode control authority is "hold", not "optimise" — an honest and
   important property that no earlier report stated this bluntly.

3. **The full chain does produce a real, protected action** when the validated
   contract is satisfied (measured with a validated bundle):
   raw MPC proposal `A 0.30 / B 0.15 / C 0.0 / D 0.0` → SafetyLayer `SAFE`
   (unmodified) → capacity guard `PROTECTED` → **FINAL action
   `A 30 % / B 15 % / C 0 % / D 0 %`, source `DOWNSTREAM_CAPACITY_GUARD`** →
   physics received **exactly** that.

Everything else that Stage 15 was asked to prove was verified with runtime
evidence: one simulation owner, one authoritative state, forecast integrity,
GNN advisory-only invariance, 1296-candidate 4-reservoir MPC, safety correction
of a deliberately unsafe proposal, mass-balance detection of four corruption
kinds, REST/WebSocket agreement, display-only frontends, end-to-end commands,
fourteen failure paths, and the hardware boundary.

**STAGE 16 HAS NOT BEEN STARTED.**

---

## 2. Complete live architecture trace

```text
authoritative inflows  (_update_inflows: baseline × (1 + 3·storm))
        ↓
frozen LSTM V3  →  LiveForecastAdapter  →  NetworkForecastSnapshot
        ↓                                        (live bundle: 3 of 4 nodes available)
GNN advisory  (spatial-dependency representation; advisory_only=true, affects_control=false)
        ↓                                        (never read by anything below)
forecast-provenance gate  →  eligible?  ── no ──→  BLOCKED / NOT_INVOKED
        ↓ yes                                      final source: HELD_CURRENT_GATES
MPCController.decide()          (1296 candidate vectors, 4 reservoirs)
        ↓  raw proposal
SafetyLayer.validate()          (bounds + per-step rate limit)
        ↓
DownstreamCapacityGuard.evaluate()   (authoritative capacity 50.0 MCM/day)
        ↓
FINAL_SAFE_CONTROL_ACTION
        ↓
SimBridge.step(inflows, gates, action_source)  →  LiveCascadeAdapter.step()
        ↓                                              →  ReservoirNetwork.step()
        │                                                  A→B→C→D, delays 2/1/1,
        │                                                  attenuation .90/.85/.80
        └── MassBalanceMonitor.step_and_check()  (audit of the applied step)
        ↓
GlobalSimulationState.get_adapted_state()  →  adapt_state_for_twin()
        ↓
FastAPI  GET /api/state  +  /ws/state
        ↓
Three.js Digital Twin (display)        Streamlit (viewer + command proxy)
```

`step()` itself, verbatim:

```python
self._update_inflows();  self._record_history()
ctrl_forecasts = self._run_ml_pipeline()            # LSTM V3 + GNN advisory + guard
gate_commands = self.manual_gates.copy(); action_source = "MANUAL_OPERATOR_GATES"
if self.mode == "AI":
    gate_commands = self._apply_ai_control(...)     # MPC → Safety → Guard
    action_source = decision.final_safe_control_action_source
self.bridge.step(self.manual_inflows, gate_commands, action_source=action_source)
self._applied_action_verification = self._verify_applied_action(gate_commands)
return self.get_adapted_state()
```

---

## 3. Single-authority proof

| Check | Result |
|---|---|
| Live `GlobalSimulationState` instances | **1** (`authoritative_instance_count() == 1`) |
| Modules constructing the live bridge | exactly `src/dashboard/api/state_manager.py` |
| Modules constructing the live state | exactly `src/dashboard/api/state_manager.py` |
| Authoritative object types | `LiveCascadeAdapter` → `ReservoirNetwork` |
| Physics engines imported by the live path | **none** (AST import audit over state_manager, routes, api/app, Streamlit app, sim_bridge, live_cascade_adapter, orchestrator) |
| Offline engines | `SimulationEngine` in `src/simulator/*` + `simulation_page.py`; `VirtualCascade` in `src/simulator/environment.py` — **all unreachable from the live path** |
| Dormant legacy controller | `SimBridge.compute_ai_recommendation` (imports `src.simulator.controllers`) — documented NON-AUTHORITATIVE (Stage 7), **zero callers** in `src/` and `tests/` |

**LIVE AUTHORITATIVE:** `state_manager.sim_state` → `SimBridge` →
`LiveCascadeAdapter` → `ReservoirNetwork` + `MassBalanceMonitor`.
**OFFLINE / RESEARCH:** `src/simulator/*`, `simulation_page.py`,
`SimBridge.compute_ai_recommendation`, the 16-dam analytics path in
`data_bridge.py`. None can advance authoritative physical state.

---

## 4. Forecast integrity

The live bundle: `LiveForecastAdapter.build_bundle(...)` →
`NetworkForecastSnapshot` with **3 of 4 nodes** available (D absent, by policy).

| Case | Eligible | MPC | Final action source | Control applied |
|---|---|---|---|---|
| **VALIDATED (all four)** | **True** | **OPTIMAL** | `DOWNSTREAM_CAPACITY_GUARD` | **True** |
| DEMONSTRATION_ONLY | False | `NOT_INVOKED` | `HELD_CURRENT_GATES` | False |
| missing Reservoir D | False | `NOT_INVOKED` | `HELD_CURRENT_GATES` | False |
| NaN forecast | False | `NOT_INVOKED` | `HELD_CURRENT_GATES` | False |
| infinite forecast | False | `NOT_INVOKED` | `HELD_CURRENT_GATES` | False |
| negative forecast | False | `NOT_INVOKED` | `HELD_CURRENT_GATES` | False |

For the validated case the contract is complete: all four reservoirs present,
1d/3d/7d horizons present, MCM/day units, finite and non-negative values, and
`validated_metrics_apply == true`. For every invalid case the gates handed on are
the **current authoritative gates** (verified numerically), never a fabricated
value, and the payload states the reason per reservoir, e.g. for the live cascade:

```text
Virtual Reservoir D: MISSING_FORECAST, NOT_VALIDATED_METRICS,
                     HORIZON_UNAVAILABLE:1d/3d/7d
Virtual Reservoir A: NOT_VALIDATED_METRICS,
                     STATUS_NOT_VALIDATED:DEMONSTRATION_ONLY
```

---

## 5. GNN advisory proof

| Requirement | Evidence |
|---|---|
| executes when valid inputs exist | live advisory `status = AVAILABLE` after the history window fills; `graph_nodes 16`, `graph_undirected_edges 41`, `embedding_dimensions 64` |
| produces advisory information | 16 node embeddings + a 16×16 embedding-similarity matrix, all from real inference |
| preserves provenance | `graph_provenance` (statistical correlation, r ≥ 0.55, ≥ 365 training dates, leakage-safe, `is_physical_topology: false`), `validation_metrics` copied from the published artifact, `validation_metrics_note` |
| `advisory_only` | **`true`** |
| `affects_control` | **`false`** |

---

## 6. A/B control-invariance result

Re-checked at the integration level (Stage 14 baseline reproduced):

| | Normal advisory | **Modified** advisory (constant 12345.0 embeddings, gate 0.999, `graph_nodes` 999) | **Disabled** advisory (`MODEL_NOT_LOADED`) |
|---|---|---|---|
| MPC proposal | A .30 B .15 C 0 D 0 | identical | identical |
| SafetyLayer output / status | A 30 B 15 C 0 D 0 · `SAFE` | identical | identical |
| Downstream guard output / status | A 30 B 15 C 0 D 0 · `PROTECTED` | identical | identical |
| FINAL_SAFE_CONTROL_ACTION / source | A 30 B 15 C 0 D 0 · `DOWNSTREAM_CAPACITY_GUARD` | identical | identical |
| ReservoirNetwork state | — | identical | identical |

End to end through the authoritative payload, with everything else held
identical: only `gnn_advisory` and the per-payload `simulation_time` differ;
**every other top-level block is byte-identical** (`reservoirs`, `cascade`,
`control`, `downstream`, `mass_balance`, `forecast_summary`,
`forecast_provenance`, `state_identity`, `simulation`, `storm`,
`hardware_status`).

---

## 7. MPC proof

| Check | Result |
|---|---|
| Action space dimension | **4** — one gate per reservoir, coordinated |
| Candidate vectors | **1296** = 6⁴ (`gate_levels [0, 0.15, 0.3, 0.5, 0.7, 1.0]`) |
| Reservoirs participating | `Virtual Reservoir A/B/C/D` → Anayirankal / Ponmudi / Idamalayar / Idukki |
| D pinned or omitted? | **No** — D's gate tracked `0.5 → 0.05` exactly across two consecutive real steps |
| Inputs | the live `ReservoirNetwork` (read, never mutated), the validated `NetworkForecastSnapshot`, constraints via `mpc.config.max_gate_change` |

---

## 8. SafetyLayer proof

Call order **measured** by instrumenting the real methods during one decision:

```text
mpc.decide → safety.validate → safety.validate → downstream_guard.evaluate
```

The `post-check` (`safety_layer_feasible`) recomputes the rule and adds **no**
layer invocation of its own. Deliberately unsafe proposal, recorded values:

| Stage | Value |
|---|---|
| raw MPC-style proposal | A **5.0**, B **−3.0**, C **NaN**, D **0.95** |
| SafetyLayer status | **CORRECTED** |
| violations | `A: gate 5.0000 clamped to 1.0000`, `A: gate movement 1.0000 exceeds limit 0.5000 — rate-limited to 0.5000`, `B: gate −3.0000 clamped to 0.0000` |
| SafetyLayer output | A **0.5**, B **0.0**, C **0.1**, D **0.5** |
| Downstream guard status | **CORRECTED** |
| guard output | A **0.5**, B **0.0**, C **0.1**, D **0.25** |
| NaN reaching physics | **no** · bounds respected **yes** · rate limit respected **yes** |

And for the validated chain: the gates handed to `ReservoirNetwork.step()` are
reported by physics as **exactly** the FINAL_SAFE_CONTROL_ACTION
(max abs difference 0).

---

## 9. Downstream-capacity proof

* Capacity read from the authoritative network: **50.0 MCM/day** — no threshold
  was invented and none was changed.
* The guard's output respects the SafetyLayer's own admissibility rules
  (bounds + per-step rate limit) — the post-check confirms it independently.
* With a safety-corrected action it returned `CORRECTED` and reduced D from
  0.50 → 0.25; with the validated action it returned `PROTECTED` and left the
  action unchanged. `FAILED_CLOSED` (its fail-closed mode) was not triggered in
  any scenario exercised here; the three statuses are all present in the
  implementation and covered by the Stage 10 suite (**40 passed**).

---

## 10. Final-action-to-physics proof

```text
FINAL_SAFE_CONTROL_ACTION  A 30 % B 15 % C 0 % D 0 %
gate_position reported by physics  A 0.30 B 0.15 C 0.00 D 0.00
max abs difference                 0.0
```

`SimBridge.step(inflows, gates, action_source)` is the only writer, and the
Stage 11 `_verify_applied_action` re-checks the applied gates against the
orchestrator's FINAL action on every step.

---

## 11. Mass-balance proof

Correct protocol (pre-step snapshot → real physics step → corruption → verify):

| Scenario | Result |
|---|---|
| clean step (`step_and_check`) | **PASS**, residual `0.000e+00`, tolerance `1e-9` |
| clean control, same protocol, no corruption | **PASS**, residual `0.000e+00` |
| corrupted **storage** (`C += 7.0`) | **VIOLATION** — detected |
| corrupted **outflow** (`D += 3.0`) | **VIOLATION** — detected |
| corrupted **routing queue** (`queue[-1] += 2.0`) | **VIOLATION** — detected |
| **non-finite** storage (`D = NaN`) | **VIOLATION** — `NON_FINITE_STATE_DETECTED` |

Nothing was repaired: the NaN was still NaN afterwards, and the inflated values
were untouched. On the live path the audit runs **after** the applied physics
action (`bridge.step` → `MassBalanceMonitor.step_and_check`), reporting
residual `2.84e-14 MCM` against the preserved `1e-9 MCM` tolerance.

---

## 12. REST / WebSocket state-authority proof

| Check | Result |
|---|---|
| Same-timestep agreement (11 blocks: reservoirs, cascade, control, mass_balance, downstream, forecast_summary, storm, simulation, state_identity, hardware_status, forecast_provenance) | **all equal** |
| Advisory content at the same timestep | equal (excluding the two per-payload volatile fields `inference_timestamp` / `inference_latency_ms`) |
| Pushed payload vs REST after the corresponding STEP | **all equal** |
| State identity advances exactly once per STEP | `network_timestep` delta = **1**; `state_id` `step10-t0` → `step11-t1` |
| State-write endpoint | `POST/PUT /api/state` → **405** |

---

## 13. Digital Twin proof

The twin (`src/dashboard/web/index.html` + `api.js`) contains **none** of:
`torch`, `tensorflow`, `onnx`, `GatedGCNLSTM`, `LiveGNNForecaster`,
`gnn_inference`, `SimBridge`, `mpc_controller`, `reservoir_network`,
`SimulationEngine`, `VirtualCascade`, `state_dict`. It **does** read
`state.gnn_advisory` and `embedding_similarity` — i.e. it renders payload values
only. Its inline JavaScript was parse-checked with `node --check`.

It renders reservoir state, gates, forecasts, the GNN advisory,
controller/safety status, downstream and mass balance — all as strings from the
payload. It runs no LSTM, no GNN, no MPC, no SafetyLayer, no guard, and computes
no physics or mass balance.

---

## 14. Streamlit proof

* **Reads** authoritative state over `GET /api/state`; **posts** only bounded
  commands.
* Imports **no** model, engine or controller module (AST audit of direct
  imports: `gnn*`, `sim_bridge`, `simulator`, `state_manager`, `network_env`,
  `controller`, `src.dashboard.api` — none present).
* Runs no GNN (`node_representations` / `cosine` absent) and makes no advancing
  call (AST call audit: no `step`, `init_cascade`, `simulation_loop`).
* Its gate command for **D/Idukki** reached the authoritative singleton:
  `manual_gates["Virtual Reservoir D"] = 20.0`, payload gate `0.20`.

---

## 15. End-to-end command tests

| Command | Result |
|---|---|
| **RESET** | backend reset executed; identity became `step10-t0` (physics clock restarted while the operator counter continued) |
| **STEP** | `network_timestep 0 → 1`, pushed state id `step11-t1`, mass balance after the step **PASS** |
| **GATE (D/Idukki)** | HTTP `200`; `manual_gates["Virtual Reservoir D"] = 20.0`; payload `reservoir_4.gate = 0.20`; cascade name `Idukki` |
| **PLAY / PAUSE** | `running` True then False, owned by the backend; the page merely requests |
| **WebSocket** | every one of the above arrived pushed on `/ws/state` |

---

## 16. Failure-path results

**14 of 14 fail safe.**

| # | Failure | Behaviour |
|---|---|---|
| 1 | missing forecast (D) | not eligible → `NOT_INVOKED`, gates held, `MISSING_FORECAST` reported |
| 2 | invalid forecast (negative) | not eligible → `NOT_INVOKED`, gates held |
| 3 | demonstration-only forecast | not eligible → `NOT_INVOKED`, `DEMONSTRATION_ONLY` reported |
| 4 | NaN forecast | not eligible → blocked; no NaN in any payload |
| 5 | infinite forecast | not eligible → blocked |
| 6 | missing reservoir | not eligible → blocked |
| 7 | unsafe MPC proposal | SafetyLayer `CORRECTED` (clamp + rate limit); NaN → 0.1 |
| 8 | downstream capacity violation | guard `CORRECTED` (D 0.50 → 0.25); `FAILED_CLOSED` available |
| 9 | mass-balance corruption (×4) | `VIOLATION` detected, nothing repaired |
| 10 | unavailable GNN | advisory `UNAVAILABLE`; simulation still steps exactly one timestep |
| 11 | modified GNN advisory | every control output unchanged (A/B) |
| 12 | malformed frontend command | `422` |
| 13 | invalid gate (`"wide open"`, NaN, unknown id) | `422` / `422` / `400`; out-of-range finite values clamp to 0 / 100 |
| 14 | invalid speed / storm / mode | `422`; zero speed clamps to ≥ 0.05 (never zero — a zero speed would divide by zero in the loop) |

No new safety behaviour was invented for any of these.

---

## 17. Hardware boundary

* `hardware_status` = `esp32` / `water_level_sensor` / `flow_sensor` /
  `gate_actuator` all **`NOT_CONNECTED`**; `mass_balance.hardware_connected`
  is **false**.
* No actuator command path exists: the only route that reaches the physics is
  `ReservoirNetwork.step()`, driven by a gate fraction in [0, 1].
* Where a future hardware adapter would connect: at the **command boundary** —
  `FINAL_SAFE_CONTROL_ACTION` is the single, bounded, provenance-tagged action
  (`final_safe_control_action_fraction/_pct/_source`) that a hardware layer would
  need to consume, and the FastAPI command models are the single ingress point
  for inbound telemetry/acknowledgements.

No hardware, PLC, Modbus, OPC-UA or MQTT was implemented.

---

## 18. Stage 15 test count

`tests/test_stage15_end_to_end_integration.py` → **56 passed** (0 failed).

Coverage: A full control cycle (measured boundary order + final action) ·
B forecast→MPC contract (validated eligible; 5 invalid variants fail closed;
snapshot contract; the live blocked path) · C GNN advisory invariance (modified,
unavailable, flags) · D safety ordering + correction · E final action→physics ·
F physics→mass balance (PASS + 4 corruption kinds + clean control) ·
G state→REST/WebSocket · H API→twin · I Streamlit→API→backend ·
J all four reservoirs · K failure paths (10 parametrised HTTP rejections + clamp,
corruption, GNN, hardware) · L single simulation authority.

The module carries an autouse finalizer that restores the shared simulation to a
paused 50 % state, so driving the singleton here cannot poison other suites.

---

## 19. Full test-suite count

`py -m pytest -q --no-header -p no:cacheprovider` → **727 passed** in 173.07 s
(671 Stage-14 baseline + 56 Stage 15). Zero failures, zero errors.

---

## 20. Stage 3–14 regression count

All `tests/test_stage{3..14}_*.py` → **486 passed** in 84.51 s — no regressions.

| Suite | Result |
|---|---|
| Stage 3–14 (combined) | **486 passed** |
| Stage 10 downstream capacity | **40 passed** |
| Stage 11 mass balance | **47 passed** |
| Stage 12 state authority | **37 passed** |
| Stage 13 Streamlit | **48 passed** |
| Stage 14 GNN advisory | **30 passed** |

---

## 21. Frozen artifact integrity

`py scripts/stage3_verify_frozen_artifacts.py` → **exit 0**,
`VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`.

The Stage 15 evidence script additionally hashes five artifacts before and after
its own run — including the LSTM V3 model, its scaler, the frozen predictions,
the **GNN checkpoint** and the **graph edges** — with
`frozen_artifacts_unchanged = True`. No retraining, no regeneration and no line
ending normalisation occurred.

---

## 22. Phase 15.3 reproduction

`py scripts/stage3_phase15_3_reproduction.py` → **exit 0**:

* protected artifacts re-hashed: **6/6 UNCHANGED**;
* re-derived outputs vs the frozen run: **3/3 IDENTICAL**;
* V3 integrity: **PASSED**.

`VERDICT: protected artifacts untouched; reproduction complete.`

---

## 23. End-to-end performance

Phases measured individually (CPU), from the audit run:

| Phase | Latency |
|---|---|
| Forecast + GNN advisory (`_run_ml_pipeline`) | **78.8 ms** |
| — of which GNN advisory inference | 7.6 ms |
| MPC → SafetyLayer → DownstreamCapacityGuard (`decide`) | **657.4 ms** |
| — SafetyLayer | 0.009 ms |
| — DownstreamCapacityGuard | 0.901 ms |
| Physics step (`bridge.step`) | **0.446 ms** |
| Mass-balance audit latency | **0.370 ms** |
| HTTP `POST /api/simulation/step` round trip | 143.3 ms |
| **Total live AI-mode cycle (as the system actually ran)** | **173.5 ms** |

**Critical reading of these numbers:** the live AI cycle is *cheap* (173 ms)
precisely **because** the provenance gate blocks the MPC — no 1296-candidate
search runs. A cycle in which the coordinated MPC actually executes costs
≈ 78.8 + 657.4 + 0.45 ≈ **737 ms**. Both figures are reported so neither can be
quoted out of context. The physics step is sub-millisecond and the mass-balance
audit is ~0.37 ms, so the physics and audit layers are effectively free next to
the control search.

No optimisation was performed.

---

## 24. Scientific claim audit

A scan of the live surface (`src/**/*.html`, `src/**/*.js` excluding `vendor/`,
and the Streamlit page) for `guarantee flood`, `flood-proof`, `prevents
flooding`, `causal discovery`, `discovers causal`, `proves hydraulic`, `controls
real dams`, `gnn directly controls`, `hardware connected`:

```text
hits: {}    →  NO UNSUPPORTED CLAIMS FOUND
```

Required terminology is present and payload-driven: the **ADVISORY ONLY**
disclaimer, the **embedding similarity** label, and the **DEMONSTRATION** banner
(`live_forecasts_are_validated = false`). Component *labels* such as "SAFETY
LAYER" are display text, not claims, and were left alone. No unrelated
documentation was rewritten.

---

## 25. Remaining limitations

1. **The live cascade never reaches the coordinated MPC.** Two independent
   reasons, both by design: the live forecasts are `DEMONSTRATION_ONLY`
   (simulation-derived inputs ⇒ `validated_metrics_apply = False`) and Reservoir D
   has no live forecast (`MISSING_FORECAST`). Consequence: in AI mode the live
   system's control authority is **"hold the current gates"**, and the
   MPC→Safety→Guard chain is exercised only when the validated contract is met
   (as it is in this stage's evidence and in the Stage 7–10 suites). Closing this
   gap requires a validated live forecast source for D — a data/model matter, not
   an architecture defect, and out of scope here.
2. **The SafetyLayer runs twice per decision** (inside the MPC and in the
   orchestrator). Functionally safe and idempotent — both evaluations precede the
   capacity boundary and the orchestrator's output is authoritative — but it is
   redundant work and the earlier "exactly ONCE" documentation was wrong
   (corrected in code).
3. **`mpc_status` is not published** in the payload's `control` block, although
   the orchestrator's decision object carries it (`NOT_INVOKED`, `OPTIMAL`, …).
   The UI therefore infers the MPC row from `forecast_control_eligible`. Adding
   the field would be a payload-schema change; it is reported here rather than
   made, since Stage 15 is verification.
4. **Before the first AI step after a reset, the published `control` block is the
   "never decided" record** (`NOT_APPLIED` / `NO_DECISION_YET`). That is honest
   (no decision has been made) but can read as a contradiction next to a
   `HELD_CURRENT_GATES` decision from `last_control_decision`; the distinction is
   "no decision yet" vs "decided to hold".
5. **A dormant legacy rule-based controller instance remains on `SimBridge`**
   (`compute_ai_recommendation` → `ForecastAwareController`), documented
   NON-AUTHORITATIVE in Stage 7 with **zero callers**, verified here. It costs a
   little memory and conceptual coupling; removing it was out of scope and would
   touch Stage 7/8/9-pinned surfaces.
6. **The "duplicate physics" scan is heuristic.** A token scan over `src/`
   flags `gnn_advisory.py`, `live_cascade_adapter.py`, `mass_balance.py` and
   `src/simulator/environment.py` as *mentioning* physics-like tokens. The
   authoritative proof is stronger and separate: exactly one live
   `ReservoirNetwork`, reached only through `LiveCascadeAdapter`, with no engine
   importable from the live path.
7. **The live AI cycle latency is measured through `sim.step()`**, which in the
   current blocked configuration does not include an MPC search. The 737 ms
   full-chain figure is a composed estimate from the phases, not a single
   end-to-end measurement of an applied MPC cycle.
8. **Pre-existing repository junk remains untouched** (`check_index.py`,
   `debug_browser.py`, `patch*.py`, `test.js`, `tmp_*.txt` at the repository
   root) — not created by this stage, not on any runtime path.

---

## 26. Exact git status

```text
 M results/phase15_stage12_authoritative_twin_state/PHASE_15_STAGE12_REPORT.md
 M results/phase15_stage3_reproduction/REPRODUCTION_CHECK.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/PHASE_15_3_V3_VALIDATION_REPORT.md
 M results/phase15_stage3_reproduction/phase15_3_reproduction/provenance_audit.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/v3_integrity_check.json
 M results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json
 M scripts/run_stage12_state_authority_audit.py
 M src/controller/live_mpc_orchestrator.py
 M src/dashboard/api/state_manager.py
 M src/dashboard/app.py
 M src/dashboard/twin_component/state_adapter.py
 M src/dashboard/web/index.html
 M src/modeling/gnn_inference.py
 M src/network_env/gnn_forecast_adapter.py
?? results/phase15_stage12_authoritative_twin_state/stage12_state_authority_evidence.json
?? results/phase15_stage13_streamlit_command_proxy/
?? results/phase15_stage14_gnn_advisory/
?? results/phase15_stage15_end_to_end/
?? scripts/run_stage13_streamlit_runtime_audit.py
?? scripts/run_stage14_gnn_advisory_audit.py
?? scripts/run_stage15_end_to_end_audit.py
?? src/modeling/gnn_advisory.py
?? tests/test_stage13_streamlit_command_proxy.py
?? tests/test_stage14_gnn_advisory.py
?? tests/test_stage15_end_to_end_integration.py
```

### Stage 15 changes only

| File | Change |
|---|---|
| `src/controller/live_mpc_orchestrator.py` | **comment correction only** — the "SafetyLayer called exactly ONCE per decision" claim replaced by the measured truth (two invocations, ordering still correct). No behaviour change. |
| `tests/test_stage15_end_to_end_integration.py` | **new** — 56 integration tests |
| `scripts/run_stage15_end_to_end_audit.py` | **new** — instrumented trace + failure matrix + claim audit (33/33 checks PASS) |
| `results/phase15_stage15_end_to_end/` | **new** — evidence JSON + this report |

Everything else listed is the previously completed Stage 12/13/14 work, plus the
Stage 3 verification JSONs whose **timestamps** changed because those scripts were
re-run as regression checks (verdicts and hashes identical).

**File hygiene:** every temporary file created during Stage 15 was removed (the
`tmp_*.txt` scratch files used to capture long command output). The three
`tmp_*.txt` files still present in the repository root are **pre-existing** and
untouched. No unrelated file was modified.

---

## 27. Scope statement

# STAGE 16 HAS NOT BEEN STARTED.

* No performance optimisation.
* No hardware, PLC/Modbus/OPC-UA/MQTT.
* No RL / MARL.
* No GNN redesign and no retraining; no LSTM replacement or retraining.
* No redesign of the MPC, SafetyLayer, DownstreamCapacityGuard,
  ReservoirNetwork, MassBalanceMonitor or the Digital Twin.
* No second simulation runtime.
* No frozen artifact modified.
* Exactly one production-behaviour-neutral correction was made: the
  SafetyLayer invocation-count comment in `live_mpc_orchestrator.py`, corrected
  to match the measured call order.
