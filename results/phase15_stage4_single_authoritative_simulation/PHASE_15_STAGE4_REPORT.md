# STAGE 4 REPORT — SINGLE AUTHORITATIVE SIMULATION

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 4 boundary (Stage 5 NOT started)
**Goal:** Exactly ONE authoritative live simulation/state producer for the Digital Twin.

---

## 1. Simulation Producers Discovered (Req. 1 & 2)

A full inventory of everything in the repository that produces or holds
simulation / Digital Twin state.

| # | Producer | File | Classification | Evidence |
|---|---|---|---|---|
| 1 | `GlobalSimulationState` (module singleton `sim_state`) | `src/dashboard/api/state_manager.py` | **LIVE / AUTHORITATIVE** | Owns `SimBridge` → `LiveCascadeAdapter` → `ReservoirNetwork`; consumed by the REST routes and the `/ws/state` WebSocket |
| 2 | Streamlit simulation in `main()` | `src/dashboard/app.py` | **DUPLICATE — competing live producer** | Held its own `st.session_state.sim_bridge = SimBridge(...)`, its own `sim_tick` / `sim_running` / `manual_gates` / `manual_inflows`, advanced its own physics on Play/Step/Reset, and pushed the result into the embedded twin via `postMessage` |
| 3 | `SimulationEngine` | `src/simulator/engine.py` | **OFFLINE / RESEARCH-ONLY** | Batch backtest over synthetic or historical scenarios; constructs its own `VirtualCascade` per run |
| 4 | `render_simulation_page()` | `src/dashboard/simulation_page.py` | **OFFLINE / RESEARCH-ONLY, DORMANT** | Zero callers anywhere in `src/`; not part of the Streamlit entrypoint |
| 5 | `run_simulation.py` CLI | `src/simulator/run_simulation.py` | **OFFLINE / RESEARCH-ONLY** | Stand-alone CLI runner |
| 6 | `validation_script.py` | repo root | **OFFLINE / AD-HOC** | Constructs its own `GlobalSimulationState()` instances for local experiments; not the server singleton |
| 7 | `VirtualCascade` / `VirtualReservoir` | `src/simulator/environment.py` | **LEGACY CLASS (retained)** | Only remaining runtime consumer is `SimulationEngine` (offline). Not used by the live path since Stage 3 |
| 8 | `SimBridge` | `src/dashboard/sim_bridge.py` | **LIVE, single-owner** | Now constructed in exactly one place: `state_manager.py` |
| 9 | `data_bridge.assess_all_reservoirs()` | `src/dashboard/data_bridge.py` | **READ-ONLY ANALYTICS** | Verified: writes no files, produces no Digital Twin state (risk/forecast assessment only) |

### 1.1 The ambiguity that existed before Stage 4

Streamlit and FastAPI **each** ran a full `SimBridge` simulation. The Streamlit
copy was a genuine second live producer: it advanced its own physics and injected
its own state into a rendered 3D twin. Two clients opening the two UIs would see
two different, independently-evolving "Digital Twins".

**No ambiguity in the target:** your approval specified FastAPI →
`GlobalSimulationState` → `LiveCascadeAdapter` → `ReservoirNetwork` →
authoritative state → WebSocket → Three.js. That is what was implemented.

---

## 2. Which One Is Authoritative

**`src/dashboard/api/state_manager.py::GlobalSimulationState`** — one module-level
instance, `sim_state`.

```
        FastAPI  (src/dashboard/api/app.py)
           │
           ├── REST commands ──► routes.py ──┐
           │                                 ▼
           └── GET /api/state ──► GlobalSimulationState  (sim_state, ONE instance)
                                          │
                                          ▼
                                SimBridge / LiveCascadeAdapter
                                          │
                                          ▼
                                ReservoirNetwork  (validated physics)
                                          │
                                          ▼
                                 authoritative state
                                          │
                                          ▼
                                   /ws/state  WebSocket
                                          │
                                          ▼
                             Three.js Digital Twin (web/index.html)
```

### 2.1 Requirements 3 & 10 — one instance, one source for both commands and the feed

* `routes.py` does `from src.dashboard.api.state_manager import sim_state`
* `app.py` (FastAPI) does `sim_state.get_adapted_state()` for the WebSocket
* Both therefore reference the **same object**.

Proven by `test_all_entry_points_share_the_same_instance`,
`test_websocket_uses_the_authoritative_singleton_object`, and behaviourally by
`test_websocket_state_changes_when_authoritative_simulation_changes`.

A registry (`_LIVE_INSTANCES`, `authoritative_instance_count()`) now counts
instances so an accidental second live simulation is caught:
`test_exactly_one_live_simulation_instance_exists` asserts the count is 1.

---

## 3. Architecture Before / After

### Before Stage 4

```
                 ┌──────────────────────────┐
   Browser ──────►│ FastAPI GlobalSimState   │──► WebSocket ──► web/index.html (twin)
                 │  SimBridge → LiveAdapter │
                 └──────────────────────────┘

                 ┌──────────────────────────┐
   Browser ──────►│ Streamlit session_state  │   *** SECOND LIVE SIMULATION ***
                 │  sim_bridge = SimBridge  │──► postMessage ──► embedded twin
                 │  sim_tick / sim_running  │
                 └──────────────────────────┘

   web/index.html ALSO accepted postMessage state from ANY frame  <<< injection
   /api/storm, /api/simulation/speed, /api/controller/mode were UNVALIDATED
```

### After Stage 4

```
                 ┌──────────────────────────┐
   Browser ──────►│ FastAPI GlobalSimState   │◄── bounded commands ONLY
                 │  SimBridge → LiveAdapter │
                 │  → ReservoirNetwork      │
                 └───────────┬──────────────┘
                             │ authoritative state (single instance)
                             ▼
                        /ws/state  ──► web/index.html  (WebSocket is its ONLY source)

   Browser ──────► Streamlit (READ-ONLY viewer)
                     reads GET /api/state  ──► mirrors into the embedded twin
                     owns no simulation, holds no simulation parameters
```

---

## 4. What Was Changed

### 4.1 Streamlit became a read-only viewer (Req. 4, 5)

`src/dashboard/app.py`:

* **Removed** `from sim_bridge import SimBridge` and the `st.session_state.sim_bridge` instance.
* **Removed** the local simulation controls that advanced a competing sim:
  Play / Pause / Step / Reset, playback speed, control-mode radio, storm slider,
  per-reservoir inflow and gate sliders, `advance_simulation()`, `reset_simulation()`.
* **Removed** `sim_running`, `sim_tick`, `sim_history` session state.
* **Added** `fetch_authoritative_state()` — a read-only HTTP GET of `/api/state`
  with graceful "backend offline" handling. Nothing is fabricated when the
  backend is down.
* The authoritative payload is forwarded **verbatim** into the embedded Three.js
  viewer (no local adaptation, no local modification).
* A "Live follow" checkbox merely *re-reads* the authoritative state; it never
  advances anything.
* Read-only status panel (controller mode, storm intensity, downstream flow,
  authoritative gate for the selected reservoir).

**Retained (Req. 7):** all read-only analytics — `data_bridge.assess_all_reservoirs()`,
risk status, water-level gauge, forecast charts, LSTM V3 performance panel,
reservoir selector, hardware status panel.

### 4.2 Offline research tooling isolated, not deleted (Req. 6, 7)

* `SimulationEngine`, `run_simulation.py`, `simulation_page.py` and
  `VirtualCascade` are **untouched and retained**.
* `render_simulation_page()` is verified to have **zero callers** — it is dormant
  offline tooling, not wired into the live dashboard.
* `test_offline_research_engine_is_isolated_and_dormant` pins both facts.

### 4.3 Frontend can no longer inject state (Req. 11)

`src/dashboard/web/index.html` (the **authoritative** twin):

* **Removed** the `window.addEventListener("message", …)` handler that accepted
  `{type:"streamlit:render", args:{state:…}}` from *any* frame/window and called
  `updateState(state)`. The authoritative twin now has exactly one state source:
  the `/ws/state` WebSocket.
* Verified: no `"message"` listener and no `streamlit:render` token remain.

`src/dashboard/twin_component/reservoir_twin.html` (the Streamlit-embedded
**mirror**): the postMessage channel is retained because it is that component's
only input — but it is now explicitly documented as a **NON-AUTHORITATIVE
MIRROR**, and the Streamlit page feeding it is itself read-only. Injecting into it
can spoof a *rendering*, never authoritative state.

### 4.4 REST command surface hardened (Req. 11)

`src/dashboard/api/routes.py` previously accepted bare `float` / `str` values, so
a browser could push `NaN`, `Infinity`, `1e308`, a zero playback speed, or an
arbitrary controller mode into the authoritative simulation.

| Endpoint | Before | After |
|---|---|---|
| `POST /gate/{id}` | bare `float`, arbitrary value | non-finite → **422**; finite clamped to [0, 100] |
| `POST /storm` | bare `float`, arbitrary value | non-finite → **422**; finite clamped to [0.0, 1.0] |
| `POST /simulation/speed` | bare `float`; `0` → ZeroDivisionError in the live loop | non-finite → **422**; clamped to [0.05, 50.0] |
| `POST /controller/mode` | bare `str`, any value | `Literal["MANUAL","AI"]` only, else **422** |
| `POST /api/state` | (does not exist) | still does not exist — **405** for POST/PUT/DELETE |

**There is no endpoint that accepts simulation state.** Clients may only issue
bounded commands.

### 4.5 Root-cause fix: 422 responses could themselves crash (found during Stage 4)

A hostile client *can* put `NaN`/`Infinity` on the wire (Python's `json.loads`
accepts them). The validators correctly rejected them — but FastAPI then echoed
the offending value inside its 422 body, and Starlette's `JSONResponse` renders
with `allow_nan=False`, so the encoder raised and the request became a **500**.
A client could therefore still destabilise the handler by "injecting" a value.

`src/dashboard/api/app.py` now registers a `RequestValidationError` handler that
renders non-finite floats as strings, so the API **always** answers with a clean
422 and never a 500. `test_non_finite_{storm,gate,speed}_is_rejected` cover all
three NaN/±Inf cases per endpoint.

### 4.6 Legacy topology JSON (Req. 13) — still LIVE, not dead

`configs/simulation/four_reservoir_demo.json` is **still used by live code**:

* `SimBridge.__init__` / `init_cascade()` — reservoir inventory (capacity, initial
  storage, max release) and the live reservoir names;
* `state_manager.py` — the `repository_derived_source` → real-reservoir mapping.

Only the file's inner `topology.routing_delays_days` block is dead on the live
path (superseded by `topology_config.yaml` in Stage 3), but that same block is
still read by the offline `SimulationEngine`. **Left untouched**, as instructed.

---

## 5. Files Modified

| File | Change |
|---|---|
| `src/dashboard/app.py` | Streamlit → read-only viewer; local simulation removed |
| `src/dashboard/api/routes.py` | Bounded command validation; no state-write endpoint |
| `src/dashboard/api/app.py` | `RequestValidationError` handler (422 never becomes 500) |
| `src/dashboard/api/state_manager.py` | Authoritative-instance registry + accessor + docs |
| `src/dashboard/web/index.html` | Removed postMessage state-injection path |
| `src/dashboard/twin_component/reservoir_twin.html` | Documented as non-authoritative mirror |
| `tests/verify_twin_integration.py` | Phase-13 test 9 updated to the Stage 4 contract (AST-based) |

**Created**

| File | Purpose |
|---|---|
| `tests/test_stage4_single_authoritative_simulation.py` | 41 integration tests |

**NOT modified (Req. 8, 9):** frozen LSTM artifacts, `src/controller/*` (MPC,
SafetyLayer, objective), `src/network_env/reservoir_network.py`,
`src/network_env/topology_config.yaml`, `results/phase15_v3_validation/*`,
`models/lstm_pytorch_v3_logtarget/*`.

---

## 6. Tests (Req. 12)

`tests/test_stage4_single_authoritative_simulation.py` — **41 tests**:

| Group | What it proves |
|---|---|
| Exactly ONE authoritative simulation | instance count == 1; routes/WS/accessor share one object; only `state_manager` constructs a bridge or a state; the instance runs the Stage 3 physics |
| Commands affect the authoritative sim | gate / storm / mode / step / reset all mutate `sim_state` (incl. reset → 50% capacity) |
| WebSocket comes from the authoritative sim | WS payload reflects an injected authoritative gate (0.37); WS and REST agree; a command changes the WS feed (0.05 → 0.65); WS endpoint reads the singleton |
| Frontend cannot overwrite backend state | no state-write endpoint (405); forged `reservoirs`/`storage`/`spill` payloads ignored; forged payload cannot set mode; unknown reservoir ids → 400 with no state change; invalid mode → 422; NaN/±Inf → 422 for storm, gate **and** speed; non-numeric → 422; speed can never be 0; out-of-range clamps without injection |
| Streamlit cannot create competing state | AST proof app.py imports/constructs no simulation class; does not import the research page; reads only the authoritative HTTP state; `SimulationEngine` isolated and `render_simulation_page` has zero callers |
| Frontend single source | `web/index.html` has no message listener / no `streamlit:render`; `api.js` exposes commands only; embedded twin documented as mirror |
| Stage 3 topology preserved | A→B→C→D, delays 2/1/1, attenuation 0.90/0.85/0.80; survives a reset |

---

## 7. Test Results (Req. 15, 17)

| Suite | Result |
|---|---|
| **Complete test suite** (`python -m pytest`) | ✅ **254 passed, 0 failed** |
| Stage 4 integration tests | ✅ 41/41 passed |
| Stage 3 authoritative-network regression tests (Req. 17) | ✅ 28/28 passed |
| Stage 3 + Stage 4 combined run | ✅ 69/69 passed |
| Phase 13 `tests/verify_twin_integration.py` (9 tests) | ✅ 9/9 passed |

Suite growth: 213 (end of Stage 3) → **254** (Stage 4, +41).

---

## 8. Frozen Artifact SHA256 Hashes (Req. 16)

Verifier: `scripts/stage3_verify_frozen_artifacts.py`

| Artifact | raw | LF-normalised | Status |
|---|---|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | MATCH | — | ✅ UNCHANGED |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | MATCH | MATCH | ✅ UNCHANGED |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | differs (CRLF) | **MATCH** | ✅ UNCHANGED |

**Verdict: ALL FROZEN ARTIFACTS UNCHANGED** (exit code 0).

The CSV's raw working-tree hash difference remains the `core.autocrlf=true`
line-ending artefact documented in Stage 3 — the LF-normalised hash is an exact
match to the manifest and to the git blob at `HEAD`. No frozen bytes changed.

---

## 9. Phase 15.3 Impact

| Check | Result |
|---|---|
| Protected artifacts (`results/phase15_v3_validation/`, 6 files) | ✅ **byte-for-byte unchanged** (hashes compared against the Stage 3 baseline) |
| `ReservoirNetwork` physics | ✅ untouched |
| Validated MPC / SafetyLayer / objective | ✅ untouched |
| Stage 3 topology (A→B→C→D, 2/1/1, 0.90/0.85/0.80) | ✅ preserved (asserted at runtime, incl. after reset) |

Because Stage 4 changed only the **live dashboard/API layer**, and the Phase 15.3
validation path (`scripts/run_phase15_3_validation.py`) never imports that layer,
the validated results are unaffected. The Stage 3 reproduction remains valid; it
was not re-run because nothing on its code path changed.

---

## 10. Remaining Duplicate / Legacy Components

| Component | Status | Why it remains |
|---|---|---|
| `SimulationEngine` | Offline, non-competing | Req. 6/7 — useful research backtest |
| `simulation_page.py::render_simulation_page` | Dormant, zero callers | Req. 7 — retained research UI |
| `src/simulator/run_simulation.py` | Offline CLI | Req. 7 |
| `VirtualCascade` / `VirtualReservoir` | Legacy class | Still used by `SimulationEngine`; off the live path since Stage 3 |
| `validation_script.py` (root) | Ad-hoc, self-contained | Creates its own instances; not the server singleton |
| `configs/simulation/four_reservoir_demo.json` | **Still live** | Reservoir inventory for the live sim (Req. 13) |

No duplicate **live** producer remains.

---

## 11. `.gitattributes` Investigation (Req. 14) — STOPPED, NOT APPLIED

Instruction: inspect first; if the rule would modify protected files or create a
large unrelated diff, **STOP and report**.

### Findings

| Measurement | Value |
|---|---|
| Tracked `*.csv` files | **93** |
| Already LF (unaffected) | 0 |
| Would be rewritten by `*.csv text eol=lf` | **93 (100%)** |
| Total lines that would change | **335,377** |
| Protected / frozen CSVs affected | **8** |

The 8 protected/frozen CSVs that would be rewritten:

```
results/lstm_pytorch_v3_logtarget/lstm_v3_metrics_original_units.csv     (FROZEN)
results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv    (FROZEN)
results/lstm_pytorch_v3_logtarget/training_history.csv                   (FROZEN)
results/lstm_pytorch_v3_logtarget/v3_period_comparison.csv               (FROZEN)
results/lstm_pytorch_v3_logtarget/v3_vs_v2_comparison.csv                (FROZEN)
results/phase15_v3_validation/daily_simulation_baseline.csv              (PROTECTED)
results/phase15_v3_validation/daily_simulation_mpc.csv                   (PROTECTED)
results/phase15_v3_validation/validation_metrics.csv                     (PROTECTED)
```

### Decision: **DO NOT APPLY**

Both stop conditions are met:

1. It **would modify protected files** (8 of them, including the frozen
   `test_predictions_original_units.csv`).
2. It would **create a very large unrelated diff** — 93 files, 335,377 lines,
   across `data/`, `results/` and `models/` — unrelated to Stage 4.

`core.autocrlf=true` remains in effect, and Stage 3's CRLF-aware verifier
(`raw OR LF-normalised == manifest`) already neutralises the practical risk.

**Recommendation for a future stage (not Stage 4):** if byte-stable hashing is
wanted, introduce the rule as its own dedicated, reviewable change —
e.g. scoped to new files only (`*.csv text eol=lf` added together with a
one-time normalisation commit, performed deliberately with protected artifacts
re-baselined and re-hashed). Do not fold it into a functional stage.

---

## 12. Risks

| # | Risk | Severity | Notes / Mitigation |
|---|---|---|---|
| 1 | **Streamlit control removal is a UX change** — operators must now use the Digital Twin UI to play/step/override gates | Medium | Required by Req. 4/5. Analytics and monitoring are unaffected. A future stage could add *command proxy* buttons that POST to the FastAPI API instead of running local physics. |
| 2 | **Streamlit ↔ FastAPI coupling** — the viewer needs the backend on `127.0.0.1:8000` | Low | Degrades explicitly ("backend not reachable"); nothing is faked. Host/port are module constants. |
| 3 | `get_adapted_state()` recomputes ML forecasts on **every** call, including every WS broadcast | Low–Medium | Pre-existing behaviour, not introduced here. A caching stage would help throughput. |
| 4 | `RequestValidationError` handler is app-wide | Low | Strictly widens correctness (422 instead of 500); message text for ordinary validation errors is unchanged. |
| 5 | `_LIVE_INSTANCES` grows if code constructs extra instances at runtime | Low | Observability only; the test asserts 1 in the app process. |
| 6 | CRLF/`autocrlf` still unresolved repo-wide | Low | Documented; CRLF-aware verifier in place (§11). |
| 7 | CORS is `allow_origins=["*"]` | Low–Medium | Pre-existing. Commands are bounded and validated, and there is no state-write endpoint, so the blast radius is limited to legitimate simulation commands. Worth tightening in a later stage. |
| 8 | Only automated tests were run; no manual browser session | Low–Medium | Recommended before a demo: load the Digital Twin, issue a gate command, confirm the WS feed moves. |

---

## 13. Requirement Compliance

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Identify every simulation/state producer | ✅ | §1 (9 producers classified) |
| 2 | Classify live / offline / duplicate / legacy | ✅ | §1 |
| 3 | ONE authoritative FastAPI simulation instance | ✅ | §2, §2.1; registry + tests |
| 4 | Streamlit must not maintain a competing sim | ✅ | §4.1; AST test |
| 5 | Streamlit kept as read-only client/viewer | ✅ | §4.1 |
| 6 | `SimulationEngine` may remain offline, must not compete | ✅ | §4.2 |
| 7 | Don't delete useful research code | ✅ | §4.2, §10 |
| 8 | Don't modify frozen artifacts / MPC / SafetyLayer / ReservoirNetwork / Phase 15.3 artifacts | ✅ | §5, §8, §9 |
| 9 | Preserve Stage 3 topology | ✅ | §7, §9 |
| 10 | WebSocket publishes from the same instance that processes commands | ✅ | §2.1; `test_websocket_uses_the_authoritative_singleton_object` |
| 11 | Frontend cannot inject authoritative state | ✅ | §4.3, §4.4, §4.5 |
| 12 | Integration tests for all five behaviours | ✅ | §6 |
| 13 | Check whether legacy topology JSON is still used | ✅ | §4.6 — still live; left untouched |
| 14 | `.gitattributes` inspected first; STOP if harmful | ✅ | §11 — **STOPPED, not applied** |
| 15 | Run the complete test suite | ✅ | §7 — 254 passed |
| 16 | Verify all frozen artifact SHA256 hashes | ✅ | §8 |
| 17 | Re-run Stage 3 regression tests | ✅ | §7 — 28/28 passed |
| 18 | Do NOT proceed to Stage 5 | ✅ | Stage 5 not started |

---

## 14. Recommendation

1. **Accept Stage 4.** There is now exactly one live simulation, it is the one the
   WebSocket publishes from, and neither the browser nor Streamlit can create or
   overwrite authoritative state.
2. **Before a demonstration**, do a short manual browser pass: start
   `python app.py`, open the twin, move a gate, confirm the WS feed updates, then
   confirm the Streamlit viewer mirrors it and shows "Connected — read-only".
3. **Stage 5 candidates** (in priority order):
   * Streamlit *command proxy* buttons (POST to the FastAPI API) to restore
     one-click control without reintroducing a local simulation;
   * cache `get_adapted_state()`'s ML pipeline so WS broadcasts are cheap;
   * tighten CORS to the known UI origins;
   * handle the CRLF/`.gitattributes` normalisation as its own dedicated change
     with protected artifacts re-baselined.

---

*End of Stage 4 Report. Stopped at the Stage 4 boundary — Stage 5 not started.*
