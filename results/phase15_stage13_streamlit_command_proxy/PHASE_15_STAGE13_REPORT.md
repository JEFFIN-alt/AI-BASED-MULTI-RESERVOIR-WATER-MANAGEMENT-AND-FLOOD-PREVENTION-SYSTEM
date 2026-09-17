# STAGE 13 — STREAMLIT READ-ONLY / COMMAND-PROXY CLEANUP — COMPLETE

**Status:** COMPLETE — Stage 13 only.
**Date:** 2026-09-16
**Goal:** make the Streamlit dashboard a pure viewer + command proxy of the ONE
authoritative backend simulation, with zero independent simulation owners, zero
independent simulation clocks and zero direct physical-state mutation.

Stage 12 was treated as the baseline and was not weakened: the Three.js Digital
Twin remains a display-only consumer of the same authoritative state.

---

## 1. Executive summary

Streamlit was **already read-only** in the parts Stage 4 fixed (it imported no
`SimBridge`, held no `sim_tick`/`sim_running`/`manual_gates`, and read
`GET /api/state`). Stage 13 therefore did **not** rewrite that work. What was
actually missing was the other half of the mandate:

* **The dashboard had no operator controls at all.** Stage 4 deleted them
  (correctly — they ran a second simulation) and recorded the follow-up as a
  known limitation: *"a future stage could add command-proxy buttons that POST to
  the FastAPI API instead of running local physics."* Stage 13 is that stage.
  Without it, requirements 5–11 of the Phase 12 behavioural test cannot even be
  attempted, because nothing can be "sent through Streamlit".
* **Most of the required authoritative state was not displayed**: state identity,
  simulation status, all four reservoir cards, mass balance, downstream-capacity
  status, forecast summary, hardware status.

Both are now implemented, and the whole thing is proven by **48 focused tests**
plus a **real runtime audit** (real uvicorn backend + real Streamlit script
executed by Streamlit's own `AppTest` + real widget clicks).

Three findings worth highlighting:

1. **The Streamlit dashboard could not run at all in this environment**:
   `streamlit` *and* `plotly` were both absent from the interpreter although
   both are listed in `requirements.txt`. They were installed to perform the
   runtime test (an environment change — no repository file was involved).
2. **A genuine command-proxy layer now exists** (`post_command` /
   `dispatch_command` / `render_command_proxy`), with a fixed, audited route
   allow-list. It cannot reach anything except the backend's bounded command
   endpoints.
3. **The dormant Phase 14.4 research page** (`simulation_page.py`, which
   constructs a `SimulationEngine`) is **still retained and still dormant** — as
   Stage 4 required — and Stage 13 proves it remains unreachable from the
   Streamlit entrypoint (no import, no `pages/` auto-discovery directory).

---

## 2. Exact files changed

### Created

| File | Purpose |
|---|---|
| `tests/test_stage13_streamlit_command_proxy.py` | 48 Stage 13 tests |
| `scripts/run_stage13_streamlit_runtime_audit.py` | Phase 12 runtime evidence (live backend + real Streamlit run) |
| `results/phase15_stage13_streamlit_command_proxy/stage13_streamlit_runtime_evidence.json` | Machine-checkable evidence |
| `results/phase15_stage13_streamlit_command_proxy/PHASE_15_STAGE13_REPORT.md` | This report |

### Modified

| File | Change |
|---|---|
| `src/dashboard/app.py` | added the command proxy (`COMMAND_PROXY_ENDPOINTS`, `post_command`, `dispatch_command`, `render_command_proxy`), the payload-driven four-reservoir section (`render_authoritative_reservoirs`), the missing authoritative status blocks (state identity, simulation, storm level, mass balance, downstream capacity, forecast summary, hardware), the display-only formatters `flow_text` / `residual_text`, and an env-var override for the backend host/port (defaults unchanged) |

### Environment (not repository) changes

`pip install streamlit` (1.64.0) and `pip install plotly` (7.1.0) — both are
declared in `requirements.txt` but were missing, so the dashboard could not have
been run or tested at all.

### Explicitly NOT touched

`src/network_env/reservoir_network.py`, `src/network_env/mass_balance.py`,
`src/controller/mpc_controller.py`, `src/controller/safety.py`,
`src/controller/downstream_capacity_guard.py`,
`src/controller/live_mpc_orchestrator.py`, `src/dashboard/sim_bridge.py`,
`src/dashboard/simulation_page.py`, `src/dashboard/data_bridge.py`,
`src/dashboard/twin_component/*`, `src/dashboard/web/*`, `src/simulator/*`,
the frozen LSTM V3 artifacts and `results/phase15_v3_validation/*`.

No physics, controller, model or frontend-rendering file was modified. No new
simulation instance was created anywhere.

---

## 3. Streamlit architecture BEFORE (as found)

```text
src/dashboard/app.py  (Streamlit entrypoint)
   ├─ get_reservoir_data() ──► data_bridge ──► src/management (16 real dams,
   │                             raw telemetry JSON + risk engine + own
   │                             LiveForecaster instance)          [analytics]
   ├─ fetch_authoritative_state() ──► GET /api/state               [read-only]
   ├─ components.html(twin_component/reservoir_twin.html)          [viewer]
   ├─ postMessage mirror of the payload into the embedded viewer   [read-only]
   └─ time.sleep(1.0) + st.rerun()   ("Live follow", read-only)    [refresh]

   NO CONTROLS AT ALL.
   No operator could PLAY / PAUSE / STEP / RESET / SET_SPEED, change a gate,
   set storm intensity or switch controller mode from Streamlit — those had been
   removed in Stage 4 and never replaced with a proxy.
```

Also present, unchanged and unreachable:

```text
src/dashboard/simulation_page.py
   render_simulation_page()  ──► SimulationEngine + controllers + metrics
                                 *** a complete local simulation runtime ***
   zero callers; not imported by app.py; no Streamlit `pages/` directory
   exists, so multipage auto-discovery cannot surface it either.
```

---

## 4. Streamlit architecture AFTER

```text
                    ┌───────────────────────────────┐
                    │  AUTHORITATIVE FASTAPI        │
                    │  GlobalSimulationState (ONE)  │
                    │  ReservoirNetwork · MPC ·      │
                    │  SafetyLayer · CapacityGuard · │
                    │  MassBalanceMonitor           │
                    └──────────────┬────────────────┘
                        GET /api/state │ POST /api/<command>
                    ┌───────────────┴────────────────┐
                    │                                │
          ┌─────────▼─────────┐          ┌───────────▼──────────┐
          │ Three.js Twin     │          │ Streamlit            │
          │ /ws/state         │          │ viewer + COMMAND     │
          │ DISPLAY ONLY      │          │ PROXY (no physics)   │
          └───────────────────┘          └──────────────────────┘
```

```text
src/dashboard/app.py  (Streamlit entrypoint — viewer + command proxy)
   ├─ GET /api/state ──► renders the backend's verdicts              [read-only]
   ├─ render_authoritative_reservoirs(payload)                       [read-only]
   │     inventory iterated from payload["cascade"] (no hardcoded list)
   ├─ COMMAND PROXY  ──► POST /api/simulation/{play,pause,step,reset,speed}
   │                     POST /api/storm
   │                     POST /api/controller/mode
   │                     POST /api/gate/reservoir_{1..4}
   │                            └─ then st.rerun() re-READS the resulting state
   └─ time.sleep(1.0) + st.rerun()   ("Live follow", read-only)      [refresh]
```

The page gained exactly one new power: **sending bounded commands**. It gained
no simulation, no clock and no state.

---

## 5. Simulation-owner audit

| Question | Answer | Evidence |
|---|---|---|
| Who owns the live simulation? | `src/dashboard/api/state_manager.py` → `sim_state` (ONE `GlobalSimulationState`) | `authoritative_instance_count() == 1`; only module in `src/` containing `SimBridge(` or `GlobalSimulationState()` |
| Does Streamlit create one? | **No** | the entrypoint's AST contains no import and no reference of `SimBridge`, `SimulationEngine`, `VirtualCascade`, `LiveCascadeAdapter`, `ReservoirNetwork`, `GlobalSimulationState`, `MassBalanceMonitor` |
| Does running the real Streamlit script create one? | **No** | runtime audit: instance count `1` before **and** after executing the page |
| Does Streamlit keep an independent gate/reservoir/controller state? | **No** | `st.session_state` writes are limited to the single transient flash key `_cmd_flash`; no `sim_running` / `sim_tick` / `manual_gates` / `manual_inflows` names exist |
| Does Streamlit keep an independent simulation clock? | **No** | see §6 |
| Is a second `SimBridge` created anywhere? | **No** | only `src/dashboard/api/state_manager.py` contains `SimBridge(` |
| Does the dormant research page create one? | It *can*, but it is unreachable: zero callers, not imported by the entrypoint, no `pages/` directory for auto-discovery. **Retained deliberately** (Stage 4 contract, asserted by `test_offline_research_engine_is_isolated_and_dormant`) and re-asserted here | `test_research_simulation_page_is_not_wired_into_streamlit` |

**The duplicate authority is the dormant `simulation_page.py`, and it was already
neutralised in Stage 4.** Stage 13 verified and pinned that rather than
rewriting it, exactly as instructed. What Stage 13 *added* is a proxy — which is
the opposite of a duplicate authority: it has no state to be authoritative with.

---

## 6. Simulation-clock audit

| Pattern searched | Result in the Streamlit surface |
|---|---|
| `while` loops advancing simulation | **ZERO** (`ast.While` count = 0) |
| `threading` / `asyncio` / `multiprocessing` / `concurrent` / `sched` imports | **ZERO** |
| `Thread(...)` / `Timer(...)` / `create_task` / `run_forever` / `tick` / `advance` | **ZERO** |
| `.step(...)` / `init_cascade(...)` | **ZERO** |
| incrementing timestep counters | **ZERO** — the displayed timestep is `state_identity.network_timestep`, read from the payload |
| local PLAY/PAUSE/STEP/RESET behaviour | **ZERO** — all four are HTTP commands; the page then re-reads state |
| periodic refresh | exactly one: `st.time.sleep(1.0)` + `st.rerun()` under the operator's "Live follow" checkbox |

The single `time.sleep` is a **UI refresh throttle, not a clock**: an AST check
asserts the enclosing `if auto_refresh:` branch calls nothing but `sleep` and
`rerun`, and contains no `dispatch_command`. It advances nothing; it only re-runs
the script so the page re-reads the backend. The authoritative clock
(`ReservoirNetwork.timestep`, incremented only by an authoritative
`ReservoirNetwork.step()`) lives in the backend and is unchanged.

---

## 7. State mutation audit

| Physical quantity | Can Streamlit write it? | Mechanism |
|---|---|---|
| reservoir storage / level | **No** | no such object exists in the page; `POST|PUT /api/state` → **405** |
| reservoir inflow / outflow | **No** | same |
| gate fraction / position | **Only** via `POST /api/gate/reservoir_{1..4}` → `manual_gates` on the authoritative object | bounded 0–100, clamped by the backend's `GateCommand` model |
| routing queues / spill | **No** | no route and no local object |
| simulation timestep | **No** | `STEP` asks the backend to step; the page never counts |
| controller / safety / guard decisions | **No** | the page cannot reach them (see §8) |
| simulation running / speed / mode / storm | **Only** via the bounded command endpoints | `running`, `sim_speed`, `mode`, `storm_intensity` on the authoritative object |

The one write Streamlit performs on its own is
`st.session_state["_cmd_flash"]` — a transient UI message string carrying the
backend's reply, never a physical value. A test asserts this is the **only**
`session_state` key written, and another asserts no `session_state` attribute
is ever named after a physical quantity.

The runtime audit adds the strongest form: a gate command through the page
changed `sim_state.manual_gates["Virtual Reservoir D"]` to `20.0` **and did not
move `network_timestep` at all** — the page mutated intent, not physics.

---

## 8. Command-flow audit

```text
Streamlit widget (operator intent, transient)
        │  dispatch_command(label, path, payload)
        ▼
post_command()  ── allow-list check ──► urllib POST (verbatim payload)
        ▼
FastAPI command model (Pydantic: finite, bounded, mode-literal)
        ▼
GlobalSimulationState  (the ONE authoritative object)
        ▼
MPC → SafetyLayer → DownstreamCapacityGuard → ReservoirNetwork.step()
        ▼
GET /api/state  ── st.rerun() ──► the page displays the RESULTING state
```

| Command | Proxied route | Page payload |
|---|---|---|
| PLAY | `POST /api/simulation/play` | none |
| PAUSE | `POST /api/simulation/pause` | none |
| STEP | `POST /api/simulation/step` | none |
| RESET | `POST /api/simulation/reset` | none |
| SET_SPEED | `POST /api/simulation/speed` | `{"speed": <0.05..50>}` |
| storm | `POST /api/storm` | `{"value": <0.0..1.0>}` |
| mode | `POST /api/controller/mode` | `{"mode": "MANUAL"|"AI"}` |
| gate ×4 | `POST /api/gate/reservoir_{1..4}` | `{"value": <0..100>}` |

Guarantees enforced by tests:

* the page's only `/api/...` literals are `/api/state` (read), the 11 declared
  command routes, and the `/api/gate/` prefix of the payload-driven gate route;
* `post_command` **refuses at runtime** any path outside
  `COMMAND_PROXY_ENDPOINTS`, so a hypothetical fifth reservoir key would be
  reported as rejected rather than smuggled through (fail-closed);
* the proxied set equals the backend's entire POST surface **both ways** — no
  dead proxy call, and no backend command the page cannot reach (read from the
  app's own OpenAPI schema);
* the payload is forwarded **verbatim**: the backend still rejects `"wide open"`,
  `NaN`, unknown reservoir ids, bad modes and non-numeric speeds (422/400);
* neither frontend imports the MPC, the SafetyLayer or the capacity guard, and
  `routes.py` itself imports none of them — the API cannot skip the chain, and
  the frontends cannot even see it.

---

## 9. REST/API integration

* Read path: `GET /api/state`, one `urllib.request.urlopen`, 2 s timeout,
  `json.loads`. On failure the page renders an explicit "backend not reachable"
  notice and **no** physical value (`flow_text` / `residual_text` return `--` for
  missing or non-finite input — unit-tested against `None`, strings, lists,
  `NaN` and `inf`).
* Write path: `POST` only, always to a declared command route, always JSON.
* No second state schema was created: the page consumes the Stage 12 payload
  blocks verbatim (`state_identity`, `simulation`, `downstream`, `storm`,
  `forecast_summary`, `forecast_provenance`, `mass_balance`, `control`,
  `cascade`, `reservoirs`, `hardware_status`).
* `AQUAFLOW_TWIN_HOST` / `AQUAFLOW_TWIN_PORT` environment overrides were added so
  the page can be pointed at a backend on another port for auditing; the
  defaults (`127.0.0.1:8000`) are unchanged.

---

## 10. WebSocket / state integration

Not applicable to Streamlit: the page is HTTP-only by design (it re-reads state
on demand), and Stage 13 did not add a WebSocket client to it. The authoritative
WebSocket feed (`/ws/state`) remains exactly as Stage 12 left it and continues to
serve the Three.js twin; the runtime audit confirms the page and the WebSocket
report the same authoritative state (identical `state_id`).

---

## 11. Four-reservoir coverage

| Requirement | Status | Evidence |
|---|---|---|
| All four reservoirs displayed | ✅ | runtime: cards for `Anayirankal`, `Ponmudi`, `Idamalayar`, `Idukki` all rendered |
| Inventory is payload-driven, not hardcoded | ✅ | `render_authoritative_reservoirs` iterates `payload["cascade"]["reservoirs"]`; a test forbids the literals `reservoir_1`, `Virtual Reservoir A`, `Anayirankal`, `Idukki` in the renderer; a payload without an inventory shows a notice instead of a guessed list |
| Idukki / Reservoir D is commandable | ✅ | four gate sliders (`cmd_gate_reservoir_1..4`); runtime gate command reached `manual_gates["Virtual Reservoir D"] = 20.0` and the card showed `GATE 20 %` |
| Per-reservoir identity from the backend | ✅ | `repository_name`, `node_id`, `cascade_position`, `is_terminal` are rendered from the payload |

Known asymmetry (pre-existing, Stage 9/12): the **Streamlit-embedded** Three.js
viewer still renders 3 basins because Reservoir D has no mesh. The new
Streamlit reservoir cards cover all four, and the authoritative
`web/index.html` twin was already four-reservoir since Stage 9.

---

## 12. Mass-balance / forecast / controller / safety status preservation

All Stage 5–12 semantics are preserved and are now **visible in Streamlit**:

| Block | Rendered as | Runtime check |
|---|---|---|
| `mass_balance` | status, residual/tolerance, `checked/expected` reservoirs | backend `PASS`, residual `2.84e-14 MCM` over `4/4` reservoirs — displayed value equals the backend's |
| `forecast_provenance` | the Stage 5/6 DEMONSTRATION banner (kept verbatim) | banner shown; `live_forecasts_are_validated` is `False` |
| `forecast_summary` | coverage status, per-horizon totals, validated-metrics flag | backend `UNAVAILABLE` in this run (no live forecast for D by policy) |
| `control` | controller type + status, forecast eligibility, `safety_layer_status`, `downstream_status`, final action source | displayed `Safety layer` and `Downstream guard` both equal the backend's verdicts |
| `downstream` | flow vs the authoritative capacity limit, utilisation, severity | `578.7 m³/s` limit read from the payload; no invented constant |
| `state_identity` | `STATE #` + physics clock + operator steps | display `step0-t0` → `step1-t1` followed the backend exactly |
| `simulation` | `RUNNING` / `PAUSED` + speed, from the backend | from `simulation.running` |
| `hardware_status` | the four hardware entries + `hardware_connected` | not connected; nothing faked |

---

## 13. Behavioural test evidence

Performed for real, not simulated — see
`scripts/run_stage13_streamlit_runtime_audit.py` and
`stage13_streamlit_runtime_evidence.json` (VERDICT: **PASS**, exit 0).

Method: a **real** uvicorn backend on `127.0.0.1:8031`, and the **real**
`src/dashboard/app.py` executed by `streamlit.testing.v1.AppTest` (Streamlit
1.64.0), pointed at that backend over real HTTP; widget clicks are real.

| # | Phase 12 requirement | Observed |
|---|---|---|
| 1 | Streamlit loads | script ran with **37 metrics rendered**, **zero exceptions** |
| 2 | Streamlit obtains backend state | `Connected — read-only`; `GET /api/state` served the page |
| 3 | Four reservoirs shown | Anayirankal / Ponmudi / Idamalayar / Idukki cards all present |
| 4 | Timestep/identity from backend | displayed `step0-t0` = backend `state_identity.state_id` |
| 5–6 | STEP through Streamlit → backend advances | `network_timestep` **0 → 1**, exactly one |
| 7 | Streamlit reflects the new state | displayed `step1-t1` = the new backend `state_id` |
| 8–10 | Gate command through Streamlit → backend receives it → state changes | slider `cmd_gate_reservoir_4` = 20 → `manual_gates["Virtual Reservoir D"] = 20.0`; payload gate `0.5 → 0.2` |
| 11 | Streamlit reflects that state | R4 card gate updated from the re-read payload |
| 12 | Streamlit executes no reservoir physics | gate command left `network_timestep` **unchanged**; page imports no simulation code |
| 13 | Mass balance corresponds to backend | displayed `PASS` = backend `PASS` (`4/4`, residual `2.84e-14`) |
| 14 | No second simulation instance | `authoritative_instance_count()` = **1** after running the page |

Console noise observed during the run (pre-existing, unrelated to Stage 13): the
16-dam analytics path logs `Live inference failed for <dam>: History dates are not
consecutive daily observations` and correctly falls back to the frozen static
test predictions with `source = VALIDATED_TEST_PREDICTION`.

---

## 14. Stage 13 test count

`tests/test_stage13_streamlit_command_proxy.py` → **48 passed** (0 failed).

Coverage map (required list → test): 1 owners / 2 clocks / 3 `VirtualCascade` /
4 `SimBridge` / 5 `GlobalSimulationState` → §5 tests + `test_streamlit_has_no_*`;
6 direct mutation → `test_streamlit_session_state_holds_only_the_transient_flash`,
`test_streamlit_has_no_local_play_pause_reset_step_logic`; 7–8 commands reach
FastAPI → `test_streamlit_commands_reach_the_authoritative_simulation`,
`test_every_proxied_route_exists_in_the_backend`; 9 display authority →
`test_streamlit_displays_the_backend_blocks`; 10 four reservoirs / 11 Idukki →
`test_all_four_reservoirs_are_commandable_including_idukki`,
`test_four_reservoir_inventory_is_payload_driven_not_hardcoded`;
12–16 status visibility → `test_streamlit_displays_the_required_verdicts`;
17–18 no bypass → `test_frontends_cannot_bypass_the_control_layers`; 19 no second
clock → `test_streamlit_has_no_simulation_clock_constructs`,
`test_streamlit_sleep_is_a_read_only_refresh_not_a_tick`; 20 consumers →
`test_only_the_authoritative_backend_owns_the_live_simulation`.

---

## 15. Full test-suite count

`py -m pytest -q --no-header -p no:cacheprovider` → **641 passed** in 120.92 s
(593 Stage-12 baseline + 48 Stage 13). Zero failures, zero errors.

---

## 16. Stage 3–12 regression count

All `tests/test_stage{3..12}_*.py` → **408 passed** in 96.38 s
(Stage 3–10: 324 · Stage 11: 47 · Stage 12: 37) — no regressions.

---

## 17. Frozen artifact integrity

`py scripts/stage3_verify_frozen_artifacts.py` → **exit 0**,
`VERDICT: ALL FROZEN ARTIFACTS UNCHANGED` (3/3 `MATCH`).

| Artifact | Result |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | `MATCH` (raw SHA256) |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | `MATCH` (raw + LF-normalised) |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | byte-identical (LF-normalised `MATCH`; raw hash differs only by CRLF working-tree endings) |

Stage 13 modified no model, scaler, prediction or physics file.

---

## 18. Phase 15.3 reproduction

`py scripts/stage3_phase15_3_reproduction.py` → **exit 0**:

* protected artifacts re-hashed: **6/6 UNCHANGED**;
* re-derived outputs vs the frozen run: **3/3 IDENTICAL**
  (`validation_metrics.csv`, `daily_simulation_baseline.csv`,
  `daily_simulation_mpc.csv`);
* V3 integrity: **PASSED**.

`VERDICT: protected artifacts untouched; reproduction complete.`

---

## 19. Performance impact

Measured by the runtime audit (real HTTP, real backend):

| Operation | Cost |
|---|---|
| `GET /api/state` (the page's read) | **8.39 ms** |
| Proxied command round trip (`POST /api/simulation/step`, incl. one full authoritative step + broadcast) | **75.32 ms** |
| Full Streamlit script re-run (37 metrics + 4 cards + diagnostics + 2 embedded components) | **198.19 ms** |

The page adds **no physics work**: a command costs one HTTP round trip on top of
whatever the backend already had to do, and one control cycle (`decide()`) is
~660 ms. So the proxy overhead is **< 12 % of a control cycle** and, more
importantly, zero when no operator acts. Nothing in the per-step path changed —
the adapter cost measured in Stage 12 (0.03–0.06 ms/call) is untouched.

Stage 13's own test suite adds **~11 s** to the suite; the runtime audit takes
~40 s (it starts a real server).

---

## 20. Remaining limitations

1. **The dashboard's 16-dam analytics section is a separate subsystem, not the
   authoritative cascade simulation.** `data_bridge.assess_all_reservoirs()`
   reads raw telemetry JSON for the 16 real Kerala dams, loads its own
   `LiveForecaster` instance, runs the Phase 14.1C risk engine and returns a
   table. It has no clock, advances no physics, writes no file, produces no twin
   state and cannot reach the authoritative object; it was retained by Stage 4 as
   read-only analytics and Stage 13 was instructed not to redesign it. It is now
   explicitly captioned in the UI as a separate, non-authoritative analytics
   subsystem. Strictly speaking, the page therefore still *instantiates* a
   forecasting model for that analytics path — it is not the cascade's forecast
   authority and cannot publish anything, but it is an honest exemption worth
   stating rather than hiding.
2. **The command-proxy route allow-list is a static 11-entry list** matching the
   validated four-reservoir topology. A hypothetical fifth reservoir would be
   refused by `post_command` (fail-closed, shown to the operator as rejected)
   rather than silently commanded. The *displayed* inventory is fully
   payload-driven; only the command surface is fixed.
3. **Widget values are operator intent and are not re-synced from the backend.**
   A `RESET` resets the simulation but leaves the operator's pending slider
   positions intact — the same semantics the Stage 12 report documented for the
   twin's sliders. Every *displayed* physical value comes from the payload.
4. **The Streamlit-embedded Three.js viewer still renders 3 basins** (Reservoir D
   has no mesh) — inherited from Stage 9, unchanged.
5. **The runtime audit is a script, not part of the pytest suite** (it starts a
   real uvicorn server and executes the real page). The committed pytest tests
   are deterministic and import no Streamlit, so the suite passes even where
   Streamlit is not installed; the runtime evidence is recorded in §13 rather
   than replayed in CI.
6. **Deprecation warnings** appear when the page runs under Streamlit 1.64
   (`st.components.v1.html` → `st.iframe`, `use_container_width` → `width`).
   They are warnings only; the existing code style was matched rather than
   rewritten, since UI redesign was out of scope.
7. **Pre-existing repository junk remains untouched** (`check_index.py`,
   `debug_browser.py`, `patch*.py`, `test.js`, `tmp_*.txt` at the repository
   root). It is not on any runtime path, was not created by this stage, and
   deleting it would modify unrelated files.

---

## 21. Hardware status

**No hardware. Nothing faked.** `hardware_status` still reports
`esp32` / `water_level_sensor` / `flow_sensor` / `gate_actuator` as
`NOT_CONNECTED`, and the page displays the backend's own
`hardware_connected: False`.

Stage 13 did not implement telemetry, actuators, PLC/Modbus or watchdogs, and it
introduced nothing that would obstruct them: every operator command now flows
through exactly one place — the authoritative backend's bounded command models —
which is the natural future location for hardware command routing,
acknowledgement and simulation/hardware mode separation. The page is a proxy, so
a future hardware backend can be substituted behind the same REST surface without
touching the frontend.

---

## 22. Exact git status

```text
$ git status --short
 M results/phase15_stage12_authoritative_twin_state/PHASE_15_STAGE12_REPORT.md
 M results/phase15_stage3_reproduction/REPRODUCTION_CHECK.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/PHASE_15_3_V3_VALIDATION_REPORT.md
 M results/phase15_stage3_reproduction/phase15_3_reproduction/provenance_audit.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/v3_integrity_check.json
 M results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json
 M scripts/run_stage12_state_authority_audit.py
 M src/dashboard/app.py
?? results/phase15_stage12_authoritative_twin_state/stage12_state_authority_evidence.json
?? results/phase15_stage13_streamlit_command_proxy/
?? scripts/run_stage13_streamlit_runtime_audit.py
?? tests/test_stage13_streamlit_command_proxy.py
```

Notes: the Stage 12 entries are the previously completed Stage 12 finalisation.
The three `results/phase15_stage3_*` JSON files and the two reproduction
artifacts changed **timestamps only** (verdicts and hashes identical) because
the Stage 3 verification and Phase 15.3 reproduction scripts were re-run as
regression checks. **No `tmp_*` or scratch file created during Stage 13
remains** — the temporary outputs used while iterating were deleted, and the
audit writes its evidence into `results/`.

---

## 23. Scope statement

# STAGE 14 HAS NOT BEEN STARTED.

* No GNN change (still advisory-only).
* No RL / MARL.
* No hardware, PLC/Modbus or telemetry.
* No MPC, SafetyLayer or DownstreamCapacityGuard redesign.
* No new fail-safe.
* `ReservoirNetwork` physics untouched.
* Frozen artifacts untouched.
* The Three.js Digital Twin was not redesigned.
* Stage 13 changed only the Streamlit page and added tests, an evidence script
  and this report.
