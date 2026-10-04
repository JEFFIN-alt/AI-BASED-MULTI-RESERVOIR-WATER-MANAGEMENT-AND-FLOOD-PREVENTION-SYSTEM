# AquaFlow classroom demonstration

For the refreshed interface, use [the new step-by-step presentation guide](PRESENTATION_DEMO_GUIDE.md).
It includes the PowerShell launch command, current button names and speaking notes.

This is a deterministic, daily reservoir-management simulation. It does not
control physical dams or guarantee flood prevention.

## Run

With the existing environment, launch the live application from this folder:

```powershell
python app.py
```

Open `http://127.0.0.1:8000/` and click **Initialize scenario**. This pauses the
simulation, resets all reservoirs to 50% storage, closes
the gates, clears forecast history, and selects deterministic local inflows
A=1, B=0.5, C=8, D=0 MCM/day. Storm changes do not alter this preset. Normal RESET
exits the preset. The backend advances one physical day per STEP; playback speed
changes wall-clock cadence only.

## Present the causal chain

1. STEP once with the gates closed. A storage rises from 5.41 to **6.41 MCM**.
2. Enter **50** in A's **Requested %** cell and press Enter. The request is
   accepted; physical gates change on STEP. **Applied %** shows the actual gate.
3. STEP again. A controlled release becomes **2.5 MCM/day** and storage becomes
   **4.91 MCM**. B receives no routed water yet.
4. STEP once: A storage becomes **3.41 MCM**; B still has no routed arrival.
5. STEP again: A storage becomes **1.91 MCM**. B receives **2.25 MCM/day** from
   A's day-2 release, after its two-day delay and 0.90 transmission factor.
   B storage increases from **12.13 to 14.88 MCM**, including its 0.5 local inflow.
6. Select **HISTORICAL REPLAY**, then **AUTO**, then STEP. Actual frozen LSTM V3
   prediction records drive live MPC. In the verified sequence, MPC selects
   A=0%, B=50%, C=0%, D=0%, replacing the preceding manual vector.
7. Select MANUAL, request D=100%, and STEP. The final safety boundary reduces
   the applied D gate to **25%**, corresponding to 50 MCM/day maximum controlled
   release in this state. Requested and applied gates are separate telemetry.
8. PLAY advances the same daily backend loop; PAUSE stops advancement. STEP
   pauses playback and advances exactly one day. Reloading the classroom preset
   restarts the deterministic sequence using **Restart scenario**.

The table, route queues, gate meshes, reservoir water surfaces, and release/spill
visuals all read backend state. Visual interpolation smooths presentation and
does not advance the water balance. The table uses MCM and MCM/day; the older
asset cards use m³/s, converted by `1 MCM/day = 1,000,000 / 86,400 m³/s`.

## Explain the scientific assumptions

- Daily storage: `S_next = S + local_inflow + routed_inflow - controlled_release - spill`.
- Requested release is `gate_fraction * max_release`; actual controlled release
  cannot exceed available water. Excess storage spills; negative storage is prevented.
- A→B: two days, 0.90 transmission; B→C: one day, 0.85; C→D: one day, 0.80.
  Queued water and attenuation losses are explicitly included in conservation.
- **Only controlled releases route between reservoirs.** Upstream spill exits
  via assumed separate lateral outlets. It is accounted for, but does not reach
  D in this model. D's controlled release plus spill defines terminal river flow.
  Do not present this as a detailed floodplain or spillway-routing model.
- Gate-to-release mapping is linear, not a hydraulic head/orifice calculation.
  Water height is a storage-fraction visual proxy. Evaporation is not implemented.
- Live MPC searches joint nominal gate targets over eight daily steps, clipping each simulated day to movement limits and a 30% prototype storage reserve. It uses
  model points at 1/3/7 days, with explicitly assumed linear interpolation between
  them. It is a finite-grid receding-horizon controller, not a continuous global solver.
- Historical replay supplies real stored predictions from frozen model evaluation.
  Their issue dates advance with simulated days. This is a historical **what-if**
  experiment on synthetic simulation states, not a forecast of their current inflows.
- The default SIMULATION forecast source remains honestly ineligible for automatic
  control when its inputs/provenance are insufficient. GCN-LSTM remains advisory;
  its 16-node training correlation graph does not define the four-node simulated
  hydraulic topology. Its input history requires seven distinct simulation days.
- Safety checks every applied live action, including manual and forecast-failure
  holds. A finite downstream search failure is reported as unprotected; it does
  not prove that no continuous action exists or that flooding can be prevented.
  Forecast protection is conditional on the tested scenario, not a guarantee.

## Reproduce verification

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_classroom_correctness.py
python -B -m pytest -q -p no:cacheprovider
```

The browser test starts an isolated real FastAPI server, sends real HTTP commands,
captures real WebSocket frames, inspects actual Three.js gate positions, and writes
JSON evidence plus manual/AI screenshots to its pytest temporary directory.
Notification delivery is disabled/mocked in tests. Frozen models are never retrained.
A completed review run is retained in `results/classroom_demonstration/`.

Backup branch: `backup/before-classroom-repairs-20261003` at `1ec5045`.
Working branch: `fix/classroom-reservoir-demonstration`.

## Repair record

| Component | Diagnosis and repair | Regression evidence |
|---|---|---|
| `downstream_capacity_guard.py`, `evaluate` | All-minimum gates were incorrectly treated as a proof of minimum future flow. Search the bounded finite lattice even when that corner fails; report search limits honestly and use the least-peak tested action when no safe candidate is found. | `test_lower_gates_are_not_a_multiday_feasibility_proof`: closed D gives a 60 peak; D=25% gives 50 throughout. |
| `safety.py`, `validate` | NaN/Inf/non-numeric fallback skipped the movement limit. Fallback now passes through bounds and rate checks; booleans are invalid. | Four invalid-input cases from a fully open gate. |
| `state_manager.py`, `step`, `_final_action_boundary` | Manual and unavailable-forecast paths bypassed final safety. Every live action now crosses bounds/rate and downstream checks before physics. | Manual, blocked AI, and adapter-error tests; browser D=100% request constrained to 25%. |
| `state_manager.py`, `_apply_ai_control` | Adapter errors used stale manual commands instead of current physical gates. Hold the current physical gates, then apply final safety. | Adapter-error regression. |
| `mpc_controller.py`, `_build_inflow_scenarios`, `decide` | Live three-step rollout placed the day-3 forecast on the second future daily step and ignored day 7. Live mode uses offsets 0..7 and movement-feasible joint candidates. | Exact interpolation timeline; changing only day-7 forecast changes the decision. |
| `live_mpc_orchestrator.py`, `decide` | Post-optimization correction left the old objective/release explanation attached to the final action. Recompute final cost and per-node release with the actual corrected vector and same daily scenario. | Final score finite, applied gates and actual releases match the explanation. |
| `state_manager.py`, reset/history/replay methods | Reset retained old forecast history; records repeated the latest replay date indefinitely; history recorded pre-step data. Clear history, advance replay issue dates daily, record completed steps, preserve issue dates in forecast bundles. | Reset/advancing replay regression and existing forecast provenance suite. |
| `reservoir_network.py`, `step` | Negative/non-finite inflows could corrupt release or partially advance queues. Validate the whole network command before mutation; validate node inputs too. | Negative/NaN/Inf rejection with unchanged nodes, queues and timestep. |
| `web/api.js` | REST failure or an older REST snapshot could mark healthy WS data stale; slider requests could complete out of order. Keep healthy WS presentation active, compare step ordering, reject older WS states, serialize gate requests. | Node-based transport regression and actual browser gate command. |
| `state_manager.py`, `broadcast_state` | Sequential sends could stall all clients; removal could race disconnect cleanup. Serialize snapshots once, send concurrently with deadlines, discard safely and close timed-out sockets for reconnect. | Slow-client isolation/reconnect test. |
| `web/index.html` | Diagnostics dereferenced null context attributes after WebGL context loss. Use nullable attributes and report context loss explicitly. | Browser requires a live context, positive FPS and actual gate-mesh movement. |

Existing tests were updated where they explicitly required the superseded unsafe
manual behavior, the three-day live horizon, or scoring rate-infeasible vectors.
The historical reproduction still requires byte-identical numerical outputs.
Its generated audit documents now go to a temporary directory rather than
rewriting tracked results. Git commit-message wording is no longer treated as a
scientific validity test. Model hashes, numerical assertions, and the real
browser acceptance checks remain enforced.

## AUTO storage reserve update

The live controller now protects a 30% prototype storage reserve using actual
local inflows and routed water due on the next step. Forecast replenishment
cannot fund today's release. Existing below-target storage is protected from
further depletion; incompatible movement limits refuse the step. Manual mode
retains operator requests with gate and downstream checks, without this AUTO
reserve. See [the updated comparison guide](MANUAL_AUTO_COMPARISON.md) for the
restart procedure and revised 12-day results. Frozen benchmark defaults and
model artifacts remain unchanged.
