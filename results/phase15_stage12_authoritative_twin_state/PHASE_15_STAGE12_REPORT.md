# PHASE 15 — STAGE 12 REPORT
## Authoritative Digital Twin State

**Status:** COMPLETE — Stage 12 only. **Stage 13 has NOT been started.**
**Date:** 2026-09-15
**Goal:** make the Three.js Digital Twin a pure display/interaction client of ONE
authoritative backend simulation state, rendered from the backend's own verdicts.

---

## 1. Exact files changed

### Created

| File | Purpose |
|---|---|
| `tests/test_stage12_authoritative_twin_state.py` | 37 Stage 12 tests |
| `scripts/run_stage12_state_authority_audit.py` | Evidence: ownership, flow, transports, frontend audit, integrity |
| `results/phase15_stage12_authoritative_twin_state/stage12_state_authority_evidence.json` | Machine-checkable evidence |
| `results/phase15_stage12_authoritative_twin_state/PHASE_15_STAGE12_REPORT.md` | This report |

### Modified

| File | Change |
|---|---|
| `src/dashboard/twin_component/state_adapter.py` | added the backend-computed display blocks: `state_identity`, `simulation`, `downstream` (flow, **authoritative limit**, utilisation, severity), `storm` (intensity + level), `forecast_summary` (aggregated horizons), per-reservoir `net_flux_m3_s` / `trend`; added `classify_trend` / `classify_downstream_status` / `classify_storm_level` |
| `src/dashboard/api/state_manager.py` | publishes `state_identity` (existing authoritative counters) and `simulation` (running / speed) |
| `src/dashboard/api/routes.py` | **fixed a real command-flow gap**: `POST /simulation/pause` stopped the backend without broadcasting (the twin kept showing `RUNNING`); PAUSE / PLAY / SET_SPEED now broadcast the resulting authoritative state |
| `src/dashboard/web/index.html` | removed the fabricated `INITIAL_STATE`, the console state-push handle, the hardcoded `150 m³/s` safe limit, the JS flow thresholds, the JS `inflow − release` trend and the sine-fabricated forecast bars; added the `STATE #` row and backend-supplied limit/severity/trend/storm/forecast rendering; `RUNNING` now comes from the backend |
| `src/dashboard/twin_component/reservoir_twin.html` | same removals in the Streamlit-embedded viewer, plus the mirror channel now accepts **only** payloads carrying the authoritative marker (and `window.__twin` is gone) |

### Explicitly NOT touched (asserted by test)

`src/network_env/reservoir_network.py`, `src/controller/mpc_controller.py`,
`src/controller/safety.py`, `src/controller/downstream_capacity_guard.py`,
`src/controller/live_mpc_orchestrator.py`, `src/network_env/mass_balance.py`
(Stage 11, unchanged), the frozen LSTM V3 artifacts and `results/phase15_v3_validation/*`.

No new simulation instance was created, and no file outside the Digital Twin /
API surface was modified.

---

## 2. Architecture before / after

### Before (as found)

```text
GlobalSimulationState → LiveCascadeAdapter → SimBridge → state_manager
   → state_adapter → FastAPI → WebSocket → index.html
                                            ├─ INITIAL_STATE (hardcoded physics, applied at boot)
                                            ├─ ds < 120 ? NORMAL : ds < 150 ? WARNING : CRITICAL
                                            ├─ "SAFE LIMIT 150 m³/s" (invented constant)
                                            ├─ trend = inflow − release
                                            ├─ forecast bars = f(sin(render clock))
                                            ├─ RUNNING (hardcoded label)
                                            └─ window.__twin.updateState({...})  ← browser could push state
```

### After

```text
authoritative backend simulation (GlobalSimulationState — ONE instance)
        ↓
ReservoirNetwork.step()  (audited by MassBalanceMonitor, Stage 11)
        ↓
StateAdapter (adapt_state_for_twin — computes the display verdicts)
        ↓
FastAPI  /api/state  +  /ws/state
        ↓
Three.js Digital Twin  →  DISPLAY ONLY
        ↑
REST commands (play / pause / step / reset / speed / gate / storm / mode)
```

The browser holds **no** state object of its own: it renders the payload and
sends bounded commands.

---

## 3. Authoritative state owner

| Fact | Evidence |
|---|---|
| Exactly one live simulation object | `state_manager.sim_state` (`GlobalSimulationState`), `authoritative_instance_count() == 1` |
| One declaration site | only `src/dashboard/api/state_manager.py` contains `sim_state = GlobalSimulationState()` |
| REST and WebSocket share it | `routes.sim_state is state_manager.sim_state`, and `app.py` imports the same object |
| No browser-side instance | no `GlobalSimulationState`, `ReservoirNetwork` or `SimBridge` construction exists in any frontend artifact |

The Digital Twin consumes only `/ws/state`; the Streamlit page reads `/api/state`
read-only and mirrors it into its embedded viewer.

---

## 4. Frontend state / injection audit

Everything below was found by reading the frontend, and every item was removed.

| # | Finding | Status |
|---|---|---|
| 1 | `INITIAL_STATE`: hardcoded storages/levels/inflows/releases/gates/storm/downstream, applied at construction before the backend spoke | **removed** → explicit `NO_AUTHORITATIVE_STATE`; the twin shows `--` until the backend speaks |
| 2 | `window.__twin = app` with documented `__twin.updateState({...})` usage — any page script or the console could push arbitrary state | **removed** → read-only `window.__twinReady` flag only (boot watchdog updated) |
| 3 | Hardcoded `SAFE LIMIT 150 m³/s` (two places) — an invented physical constant presented as the river limit | **replaced** by the authoritative `downstream.capacity_m3_s` (50 MCM/day = **578.7 m³/s**) |
| 4 | JS risk thresholds `ds < 120 / ds < 150` (two places: HUD + flow labels) | **replaced** by the backend's `downstream.status` |
| 5 | JS trend `inflow − release` (R1–R3 and R4) | **replaced** by the backend's `net_flux_m3_s` / `trend` |
| 6 | Forecast bars fabricated as `clamp01(s + 0.14·sin(clock…))` | **replaced** by visual interpolation of the backend's `forecast_summary` horizons (no value invented; flat when unavailable) |
| 7 | Hardcoded `RUNNING` labels | **replaced** by the backend's `simulation.running` (`PAUSED`/`RUNNING`/`--`) |
| 8 | Storm `LIGHT/NORMAL/HEAVY/SEVERE` decided in JS | **replaced** by the backend's `storm.level` |
| 9 | Floating 3D labels used a *second*, mismatched fallback name list (`Anathode`, `Idamalayar`, `Idukki`) | **replaced** by the single `resNames` fallback; the displayed name always comes from the payload's `cascade` block |
| 10 | Console state-push handle in the Streamlit-embedded viewer | **removed**; its one-way mirror channel now ignores any payload without the authoritative marker |
| 11 | `postMessage` state injection into the authoritative twin | none (removed in Stage 4) — re-verified and pinned |

The audit is executed by `scripts/run_stage12_state_authority_audit.py` and pinned
by `test_frontend_does_not_calculate_physical_state`,
`test_frontend_contains_no_fabricated_initial_state`,
`test_no_console_state_push_handle_remains`,
`test_authoritative_twin_has_no_postmessage_injection_path`,
`test_streamlit_mirror_only_accepts_authoritative_payloads`.

**What the browser still does (allowed):** mesh positions, camera tweens, particle
systems, water shader time, HUD throttling, unit *formatting* (percent/`m³/s`
display, ratios → bar widths) and mapping a backend string to a CSS class.

---

## 5. WebSocket / REST verification

* Every existing field is **preserved** — `reservoirs` (all four, incl. Idukki/D),
  `cascade`, `control` (controller/safety/downstream statuses),
  `mass_balance`, `forecast_provenance`, `hardware_status`, `metadata`,
  `downstream_flow`, `controller_mode`, `simulation_time`. Stage 12 only **adds**
  (`state_identity`, `simulation`, `downstream`, `storm`, `forecast_summary`,
  per-reservoir `net_flux_m3_s` / `trend`).
* `GET /api/state` and `/ws/state` return the **same** authoritative state
  (compared field-by-field in the evidence: `reservoirs`, `cascade`, `control`,
  `mass_balance`, `forecast_provenance`, `state_identity`, `downstream`,
  `forecast_summary`, `storm`, `simulation`, `hardware_status` — all equal).
* No state-write endpoint exists: `POST`/`PUT /api/state` → `405`; forged fields
  posted to command endpoints are ignored; unknown reservoirs → `400`.

### State freshness / identity (requirement E)

**What identifies a state update:**

| Field | Meaning |
|---|---|
| `state_identity.sim_step_index` | authoritative steps taken by the ONE `GlobalSimulationState` |
| `state_identity.network_timestep` | `ReservoirNetwork.timestep` — incremented **only** by an authoritative `ReservoirNetwork.step()` |
| `state_identity.state_id` | label derived from those two: `step<N>-t<M>` |

No new clock was introduced: both counters already existed. A **RESET** is
distinguishable because `network_timestep` returns to `0` while `sim_step_index`
keeps counting (observed live: `step1-t1` → `step1-t0` after RESET). The twin
displays this in the new `STATE #` HUD row, taken verbatim from the payload.

`simulation_time` remains a wall-clock stamp generated per payload and is
deliberately **not** used as the identity (it changes without any state change).

---

## 6. Command-flow verification

```text
frontend command → FastAPI → authoritative simulation state
                → simulation/control pipeline → WebSocket broadcast
                → frontend displays the resulting state
```

| Command | Backend effect | Verified by |
|---|---|---|
| `PLAY` | `sim_state.running = True` + broadcast | REST + WS test, browser |
| `PAUSE` | `running = False` **+ broadcast** (fixed) | REST + WS test, browser |
| `STEP` | `running = False`, one authoritative step, broadcast | `network_timestep` +1 exactly |
| `RESET` | `init_cascade(50.0)`, broadcast | `network_timestep` → 0 |
| `SET_SPEED` | `sim_speed` (clamped ≥ 0.05) + broadcast | payload `simulation.speed` |
| gate (`reservoir_1..4`) | `manual_gates[node]` (clamped 0–100), broadcast | browser G4 → Idukki |
| storm | `storm_intensity` (clamped 0–1), broadcast | validation test |
| mode | `mode` (`MANUAL`/`AI`), broadcast | validation test |

The frontend never mutates state: `api.js` only POSTs bounded commands, and the
gate/storm sliders write only their own label text before calling the API.

**Stage 12 defect found and fixed:** `POST /api/simulation/pause` changed the
backend without broadcasting, so a connected twin kept displaying `RUNNING`
until some other command happened to push a state. PAUSE (and PLAY/SET_SPEED)
now broadcast the resulting state — this is exactly the "backend merely displays
it" failure mode the stage was written to prevent.

---

## 7. Browser smoke-test evidence

A **real browser** was used (the VS Code integrated browser against a live
`uvicorn` server on `127.0.0.1:8130`). No results are simulated.

| Check | Observed |
|---|---|
| Page loads | title `AQUA FLOW — Multi-Reservoir Digital Twin`; HUD rendered; `window.__twinReady === true` |
| WebSocket connects & drives the UI | reservoir values appeared with no REST call from the UI |
| No console state-push handle | `typeof window.__twin === 'undefined'` |
| Initial (unstepped) state is truthful | `STATE # step0-t0`, `PAUSED`, `MASS BALANCE: NOT CHECKED`, `MB CHECKED: NO`, `0 CHECKED`, `SAFE LIMIT 578.7 m³/s`, names Anayirankal / Ponmudi / Idamalayar / **Idukki**, storm `LIGHT` |
| `STEP` advances exactly one backend timestep | `step0-t0` → `step1-t1` |
| Mass balance reaches the browser | after the step: `MASS BALANCE: PASS`, `MB CHECKED: YES`, `MB RESERVOIRS: 4 CHECKED`, residual `2.84e-14 / 1e-9 MCM` |
| Reservoir values follow the backend | R1 `59.2%` ▲ RISING, R2 `52.4%`, R3 `54.3%`, **R4/Idukki `25.1%` ▼ FALLING** |
| Downstream severity is the backend's | `1157.4 m³/s` against the authoritative `578.7 m³/s` limit → `CRITICAL` (and the DS chip renders DANGER) |
| `PLAY` changes backend state and the UI follows | `RUNNING`; state advanced `step1-t1` → `step5-t5` in 3.5 s (1 s cadence) |
| `PAUSE` stops backend progression | state frozen at `step5-t5` across 2.5 s and the UI reports `PAUSED` |
| `RESET` restores backend state | `step1-t0` (physics clock restarted), levels back to `50.0%`, downstream `NORMAL`, mass balance back to `NOT CHECKED` |
| Gate command reaches the backend | G4 slider → `20`; backend applied it; R4 card shows `GATE 20%` |
| No client-side injection needed | the whole session ran on `/ws/state` + REST commands only |

Incidental robustness observation: during a server restart the page's WebSocket
reconnect logic retried (`ERR_CONNECTION_REFUSED`) and recovered on reload —
behaviour that exists in `api.js`, unchanged by Stage 12.

---

## 8. Tests and exact counts

| Suite | Result |
|---|---|
| `tests/test_stage12_authoritative_twin_state.py` | **37 passed** |
| Full suite | **593 passed** (556 Stage-11 baseline + 37) |

Stage 12 coverage maps to the required list: one owner (1, 14), no injection
(2), WebSocket delivery (3), REST state (4), REST/WS agreement (5), no JS physics
(6), commands don't mutate locally (7), PLAY/PAUSE/STEP/RESET/SET_SPEED backend-
authoritative (8, 15), four reservoirs incl. Idukki (9, 13), mass balance
through the path (10), forecast provenance (11), controller/safety/downstream
statuses (12), no second clock (14).

## 9. Stage 3–11 regression

`371 passed` (Stage 3–10: 324 · Stage 11: 47) — no regressions.

Two Stage 4 *documentation* contracts in the embedded mirror comment were
initially broken by the Stage 12 rewrite and were restored verbatim
(`NON-AUTHORITATIVE MIRROR`, `web/index.html`), so the Stage 4 assertions remain
meaningful rather than being edited away.

## 10. Phase 15.3 reproduction

STAGE_12_PHASE153_RESULT

## 11. Frozen artifact integrity

STAGE_12_FROZEN_RESULT

---

## 12. Performance impact

STAGE_12_PERF_RESULT

The per-command broadcasts added to PAUSE/PLAY/SET_SPEED are one state
serialisation each (`json.dumps` of the twin payload) and only occur on operator
commands, not per simulation step.

---

## 13. Remaining limitations

1. **The Streamlit-embedded viewer is still a 3-reservoir scene** (no mesh for D),
   inherited from Stage 9. The authoritative `web/index.html` renders all four
   cards incl. Idukki. Redesigning Streamlit was explicitly out of scope for
   Stage 12.
2. **The mirror channel is still a parent→iframe push.** It is now restricted to
   payloads carrying the authoritative marker, and it has no write-back path, but
   a frame that reproduces that marker can still *display* arbitrary numbers in
   that non-authoritative viewer. It cannot affect the authoritative simulation
   (no browser→state path exists), and the authoritative twin has no such channel
   at all. Removing the channel entirely would break the Stage 4 Streamlit
   read-only mirror, which Stage 12 was instructed not to redesign.
3. **The gate/storm/speed sliders keep their own positions** as operator command
   inputs; they are not re-synced from the backend. The *displayed* gate/limit/
   status values always come from the payload. (A RESET therefore resets the
   simulation but leaves the operator's pending commands intact — pre-existing
   semantics.)
4. **`simulation_time` is a wall-clock stamp**, not a simulation clock. It is
   preserved for compatibility; the authoritative identity is
   `state_identity`. A client should not use `simulation_time` for freshness.
5. **The browser was verified by one interactive session**, not by a committed
   headless end-to-end runner. The automated suite covers REST/WS/identity/
   command semantics; the browser evidence is recorded in §7 rather than replayed
   in CI.
6. **Pre-existing repository junk remains untouched** (`check_index.py`,
   `debug_browser.py`, `patch*.py`, `test.js` at the repository root). They are
   not part of any runtime path and were not created by this stage; deleting them
   would have modified unrelated files.

---

## 14. Scope statement

Stage 12 changed **only** the Digital Twin state authority and its presentation
path.

* No GNN change (still advisory-only).
* No hardware.
* No RL/MARL.
* No new fail-safe.
* No UI redesign.
* `reservoir_network.py`, `mpc_controller.py`, `safety.py`,
  `downstream_capacity_guard.py` and `live_mpc_orchestrator.py` were **not**
  modified (asserted by test).
* **Stage 13 has NOT been started.**
