# STAGE 3 — TOPOLOGY & PHYSICS RECONCILIATION REPORT

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at Stage 3 boundary (Stage 4 NOT started)
**Objective:** Make the validated `ReservoirNetwork` the ONE authoritative physics model for the live Digital Twin.

---

## 1. Executive Summary

The live Digital Twin now runs on the validated `ReservoirNetwork`.

| Before Stage 3 | After Stage 3 |
|---|---|
| `SimBridge` → `VirtualCascade` (Phase 14.4 proof-of-concept) | `SimBridge` → `LiveCascadeAdapter` → `ReservoirNetwork` |
| Delays 1/1/1 (live JSON) | Delays 2/1/1 (validated `topology_config.yaml`) |
| No attenuation (implicit 1.0) | Attenuation 0.90 / 0.85 / 0.80 |
| Spill lumped into release and routed downstream | Spill separated from release; non-terminal spill leaves the network |
| No global mass accounting | Closed global mass balance (residual < 1e-10) |
| No provenance | OBSERVED / VERIFIED / ASSUMED_FOR_PROTOTYPE per parameter |

`VirtualCascade` was **not deleted** and **not modified**. `ReservoirNetwork` was
**not modified**. The validated MPC and `SafetyLayer` were **not modified**. The
frozen LSTM artifacts were **not read, written or touched**.

**Headline evidence**

| Check | Result |
|---|---|
| Complete test suite | **213 passed, 0 failed** |
| Frozen artifact SHA256 verification | **ALL FROZEN ARTIFACTS UNCHANGED** |
| Protected Phase 15.3 artifacts | **6/6 byte-for-byte unchanged** |
| Phase 15.3 baseline/MPC reproduction | **byte-for-byte identical** to protected run |
| Implementation-equivalence proof | `VirtualCascade` ≡ `ReservoirNetwork` to **0.000e+00** when parameters coincide |

---

## 2. Scope Discipline — What Was and Was Not Touched

### 2.1 Created

| File | Purpose |
|---|---|
| `src/network_env/live_cascade_adapter.py` | The live facade over `ReservoirNetwork` (the adapter) |
| `scripts/stage3_physics_parity_report.py` | Controlled A/B/C parity harness + semantic difference register |
| `scripts/stage3_verify_frozen_artifacts.py` | Frozen artifact SHA256 verification |
| `scripts/stage3_phase15_3_reproduction.py` | Safe Phase 15.3 re-run with protected-artifact shielding |
| `tests/test_stage3_live_network_authority.py` | 28 parity / regression tests |
| `results/phase15_stage3_topology_reconciliation/` | Parity report, JSON traces, semantic difference CSV, integrity JSON |
| `results/phase15_stage3_reproduction/` | Reproduced 15.3 outputs + protection/fidelity evidence |

### 2.2 Modified

| File | Change |
|---|---|
| `src/dashboard/sim_bridge.py` | Live physics switched from `VirtualCascade` to `LiveCascadeAdapter`; legacy import removed |
| `src/network_env/__init__.py` | Exports `LiveCascadeAdapter`, `LiveReservoirView` |

*(These are the ONLY source files changed by Stage 3.)*

### 2.3 Explicitly NOT touched

| Artifact | Status |
|---|---|
| `src/simulator/environment.py` (`VirtualCascade`, `VirtualReservoir`) | UNCHANGED — still present, still behaves as before |
| `src/network_env/reservoir_network.py` | UNCHANGED — the authority was preserved exactly |
| `src/controller/mpc_controller.py`, `safety.py`, `objective.py` | UNCHANGED |
| `models/lstm_pytorch_v3_logtarget/` | UNCHANGED |
| `results/lstm_pytorch_v3_logtarget/` | UNCHANGED |
| `results/phase15_v3_validation/` | UNCHANGED (protected; verified by before/after hash) |
| `src/network_env/topology_config.yaml` | UNCHANGED (already matched the Stage 3 targets) |

> Note: `src/simulator/environment.py` and `src/simulator/controllers.py` already
> carried uncommitted changes **before** Stage 3 began (Stage 2 unit-contract work).
> Stage 3 did not add to them.

### 2.4 Out of scope (deliberately)

`src/simulator/engine.py::SimulationEngine` — the research / backtest simulator
behind the Streamlit *Simulation* page — still uses `VirtualCascade`. It is not
part of the live Digital Twin and was left untouched, as required.

---

## 3. Target Verification (Requirements 5–7)

Requested targets, as measured from the live code path:

| Requirement | Target | Measured | Status |
|---|---|---|---|
| Topology | A → B → C → D | `['Virtual Reservoir A', 'B', 'C', 'D']`; edges A→B, B→C, C→D | ✅ |
| Delay A→B | 2 | 2 | ✅ |
| Delay B→C | 1 | 1 | ✅ |
| Delay C→D | 1 | 1 | ✅ |
| Attenuation A→B | 0.90 | 0.90 | ✅ |
| Attenuation B→C | 0.85 | 0.85 | ✅ |
| Attenuation C→D | 0.80 | 0.80 | ✅ |

Delays and attenuation were verified **behaviourally**, not just by reading config:
an impulse released at A (5.0 MCM) produces

* day 1–2: B routed inflow = 0 (2-day delay)
* day 3: B routed inflow = 5.0 × 0.90 = **4.50**
* day 4: C receives 4.50 × 0.85 = **3.825**
* day 5: D receives 3.825 × 0.80 = **3.06**

Pinned by `tests/test_stage3_live_network_authority.py::test_measured_routing_delays_are_two_one_one`.

**No change to `topology_config.yaml` was required** — the validated configuration
already encoded the requested targets.

---

## 4. Controlled Comparison Methodology (Requirements 2–3)

Both models were run under **identical** initial states, inflows, gate commands
(identical percent values at the shared boundary) and timestep (1 day).

Three engines:

| Key | Implementation | Delays | Attenuation |
|---|---|---|---|
| `legacy` | `VirtualCascade` (as the live twin used it) | 1/1/1 | none (1.0) |
| `network_legacy_equivalent` | `ReservoirNetwork` | 1/1/1 | 1.0 |
| `authoritative` | `LiveCascadeAdapter` → `ReservoirNetwork` | 2/1/1 | 0.90/0.85/0.80 |

This design separates *implementation* differences from *parameter* differences.

---

## 5. Parity Results

### 5.1 Scenario S1 — routing only, no spill

Initial storages `{A:5.0, B:10.0, C:170.0, D:200.0}`, inflows `{A:1, B:2, C:20, D:0}`,
gates `{A:40%, B:35%, C:50%, D:100%}`, 10 days.

| Comparison | storage | release_total | inflow_routed | downstream_flow | identical |
|---|---|---|---|---|---|
| `legacy` vs `network_legacy_equivalent` | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | **YES** |
| `network_legacy_equivalent` vs `authoritative` | 2.800e+00 | 1.500e+01 | 1.500e+01 | 1.500e+01 | NO |
| `legacy` vs `authoritative` | 2.800e+00 | 1.500e+01 | 1.500e+01 | 1.500e+01 | NO |

**This is the key result.** When their assumptions coincide, the two
implementations agree to the last floating-point bit. Every Stage 3 behavioural
change is therefore attributable to a *named parameter*, not to a rewritten or
subtly different simulation.

### 5.2 Scenario S2 — upstream spill

Initial storages `{A:10.0, B:5.0, C:50.0, D:100.0}`, inflows `{A:6, B:1, C:10, D:0}`,
gates `{A:10%, B:20%, C:40%, D:100%}`, 8 days. Reservoir A overflows.

| Comparison | storage | release_total | inflow_routed | downstream_flow |
|---|---|---|---|---|
| `legacy` vs `network_legacy_equivalent` | 2.076e+01 | 5.000e+00 | 5.500e+00 | 5.000e+00 |
| `legacy` vs `authoritative` | 2.126e+01 | 1.200e+01 | 1.200e+01 | 1.200e+01 |

Divergence here is attributable to the **spill-routing** semantic difference
(§6, `spill`).

### 5.3 Scenario S3 — authoritative full

Initial storages `{A:8.0, B:15.0, C:120.0, D:150.0}`, inflows `{A:5, B:8, C:60, D:0}`,
gates `{A:60%, B:50%, C:70%, D:100%}`, 12 days. Terminal reservoir reaches capacity.

| Comparison | storage | release_total | inflow_routed | downstream_flow |
|---|---|---|---|---|
| `legacy` vs `network_legacy_equivalent` | 0.000e+00 | 8.000e+00 | 8.000e+00 | 8.000e+00 |
| `legacy` vs `authoritative` | 7.500e-01 | 2.394e+01 | 2.394e+01 | 2.394e+01 |

### 5.4 Attribution

| Check | Result |
|---|---|
| `s1_legacy_backed_network_matches_virtualcascade` | **PASS** |
| `s2_spill_semantics_diverge` | **PASS** |
| `s3_authoritative_differs_from_legacy` | **PASS** |

Full traces: `physics_parity.json`. Human report: `PARITY_REPORT.md`.

---

## 6. Semantic Difference Register (Requirement 3)

Machine-readable: `semantic_differences.csv`.

| # | Dimension | Legacy `VirtualCascade` | Authoritative `ReservoirNetwork` | Impact | Resolution |
|---|---|---|---|---|---|
| 1 | **topology** | Hardcoded 4 reservoirs, names baked into Python source | Topology-agnostic DAG read from `topology_config.yaml` (A→B→C→D) | None for the 4-reservoir cascade; enables N-reservoir extension with no code change | Node ids are the live virtual names, mapped to canonical ids via `display_name`, then `repository_reference` ↔ `repository_derived_source`. Unresolvable mapping **raises** rather than guessing |
| 2 | **routing delays** | 1 day on all links (from `four_reservoir_demo.json`) | A→B = 2, B→C = 1, C→D = 1 | A→B travel time doubles; upstream flood waves reach B one day later | **Adopted the validated 2/1/1 target** (see §9.1) |
| 3 | **attenuation** | None (implicit factor 1.0); transmission loss never occurs | 0.90 / 0.85 / 0.80; loss booked as `total_routing_loss` | 10% / 15% / 20% of routed volume is lost in transit; downstream routed inflows and terminal flow decrease | **Adopted the validated attenuation.** Loss is explicit and mass-balanced — never silently destroyed |
| 4 | **storage** | `storage_mcm`, `capacity_mcm` in MCM | `storage`, `capacity` in MCM | None | No change. Adapter exposes `storage_mcm` / `capacity_mcm` verbatim |
| 5 | **release** | `release_mcm_day` = controlled release **+ spill** (lumped) | `controlled_release` and `spill` are separate; `total_outflow = controlled + spill` | Legacy callers cannot distinguish a controlled release from an overflow | Adapter keeps `release_mcm_day = total_outflow` (legacy meaning preserved) **and** adds `controlled_release_mcm_day`, `spill_mcm` |
| 6 | **spill** | Overflow is added to release **and routed downstream** to the next reservoir | Non-terminal spill **leaves the network** (not routed); terminal spill is part of terminal outflow | Legacy inflated B's inflow with A's overflow; the network conserves it as a separate outflow term | Intentional accepted difference; quantified in S2/S3; pinned by `test_non_terminal_spill_is_not_routed_downstream` |
| 7 | **downstream capacity** | Compared against `res_D.release_mcm_day` (controlled + spill) | Compared against terminal node `total_outflow` (controlled + spill) | None — both compare the same quantity at the terminal node | No change; `current_downstream_flow = network.terminal_outflow`. Limit 50.0 MCM/day in both |
| 8 | **mass balance** | No global accounting; water in queues and routing loss untracked | Closed balance: `inflow = Δstorage + terminal outflow + non-terminal spill + routing loss + water in transit` | Live twin gains an auditable conservation law it never had | Exposed via `LiveCascadeAdapter.mass_balance_check()`; residual < 1e-10 in tests |
| 9 | **units** | Gates PERCENT [0,100]; storage MCM; flow MCM/day | Gates internally FRACTION [0,1]; storage MCM; flow MCM/day | A fraction/percent mix-up would under-release ~100× | **Single** conversion boundary in `src/common/units.py`; adapter converts percent→fraction exactly once. Non-finite gates **raise** `UnitContractError` instead of failing open |
| 10 | **missing inputs** | `KeyError` if an inflow or gate entry is absent | Absent entries default to 0.0 | Strictly safer; a partially-specified command no longer aborts the live loop | Follows `ReservoirNetwork`; documented robustness change |
| 11 | **gate clamping** | Finite out-of-range clamped into [0,100] | Finite out-of-range clamped into [0,1]; non-finite raises | Identical for finite input; both already route through `src/common/units.py` | No behavioural change |
| 12 | **provenance** | None | Every parameter classified OBSERVED / VERIFIED / ASSUMED_FOR_PROTOTYPE | The live twin can now distinguish observed data from prototype assumptions | Exposed via `LiveCascadeAdapter.physics_provenance()` |

### 6.1 Live physics provenance summary

Registry counts from the live adapter: `{'OBSERVED': 4, 'VERIFIED': 0, 'ASSUMED_FOR_PROTOTYPE': 16}`.

The 4 OBSERVED entries are the reservoir capacities. Topology, routing delays,
attenuation, initial storages, max releases and downstream capacity remain
**ASSUMED_FOR_PROTOTYPE** and must continue to be labelled as such.

---

## 7. Adapter Design (Requirement 10)

```
        Live Digital Twin callers
   (state_manager / routes / app.py / tests)
                    │  gate PERCENT, MCM, MCM/day
                    ▼
        LiveCascadeAdapter          ← the ONE live boundary
                    │  gate FRACTION
                    ▼
     ReservoirNetwork (UNMODIFIED)  ← validated research logic = authority
        built from topology_config.yaml
```

* **Composition, not modification.** `LiveCascadeAdapter` *holds* a
  `ReservoirNetwork`; it never subclasses or patches it.
* **Parameter precedence.** Reservoir inventory (capacity, initial storage,
  max release) comes from the live config so existing live parameterisation is
  preserved exactly; connections (delays + attenuation) and downstream capacity
  come from the validated `topology_config.yaml`.
* **Capacity drift is logged**, never silently absorbed.
* **Single unit boundary**: `units.gate_percent_to_fraction` is called exactly once
  per gate, per step.
* `LiveCascadeAdapter.from_network_config(...)` provides an explicit constructor
  for already-composed network configs (used by the parity harness).

---

## 8. Test Evidence (Requirement 11)

`tests/test_stage3_live_network_authority.py` — **28 tests, all passing**:

| Group | Tests |
|---|---|
| Live path IS the authoritative network | adapter type; `ReservoirNetwork` backing; full legacy public surface preserved |
| Live path no longer depends on `VirtualCascade` | AST check over 5 live modules; no import of `src.simulator.environment` |
| `VirtualCascade` preserved | importable; legacy 1-day/no-attenuation routing still holds; source intact |
| Topology | linear cascade A→B→C→D; edges exactly A→B, B→C, C→D |
| Targets | delays 2/1/1 and attenuation 0.90/0.85/0.80 from config |
| Measured behaviour | impulse-response delay proof; attenuation 4.50 → 3.825 → 3.06; routing loss booked |
| Implementation equivalence | `VirtualCascade` ≡ `ReservoirNetwork` to `abs=1e-12` over 10 steps |
| Unit boundary | 75% → 0.75 × max_release; 75% not read as 0.75%; NaN/Inf raise |
| Conservation | mass balance residual < 1e-10; no negative storage; no capacity breach |
| Semantics | non-terminal spill not routed downstream |
| Determinism | two identical runs produce identical storages |
| Twin contract | `adapt_state_for_twin` schema valid; water level tracks storage fraction; provenance exposed |

**Complete suite: 213 passed, 0 failed, 0 errors** (`python -m pytest`).

---

## 9. Decisions and Ambiguities (Requirement: report, do not invent)

### 9.1 RESOLVED by your approval — routing delay conflict

The live config `configs/simulation/four_reservoir_demo.json` declares
`A_to_B = 1 day`, while the validated `topology_config.yaml` declares
`A_to_B = 2 days`. These disagree.

Your Stage 3 approval explicitly fixed the target at **A→B = 2**, matching
`topology_config.yaml`. The live path therefore adopts 2/1/1.

**Consequence:** the legacy JSON block `topology.routing_delays_days` is now
**unused by the live path**. It was **left in place, unmodified**, because
`SimulationEngine` (research/backtest, out of scope) still reads it. If you want
the live JSON de-duplicated, that is a Stage 4 decision.

### 9.2 RESOLVED by your approval — attenuation introduced

The live config had **no attenuation field at all**, i.e. the live twin modelled
zero transmission loss. Your approval fixed attenuation at 0.90/0.85/0.80. This
is the largest quantitative behaviour change for the live twin and it is
mass-conserving, but it is a **new physical loss term** on the live path.

### 9.3 FLAGGED — no physical ambiguity was invented

No physical parameter was guessed. Where a value was needed:

* topology / delays / attenuation — taken from the validated `topology_config.yaml`;
* reservoir capacities / max releases — identical in both configs, so no conflict;
* initial storages — driven by the existing `init_cascade(pct)` behaviour, unchanged;
* downstream capacity — 50.0 MCM/day in both configs, so no conflict;
* name mapping — resolved from `display_name`, else `repository_reference`;
  **raises** if unresolvable rather than assuming a physical identity.

---

## 10. Frozen Artifact SHA256 Verification (Requirement 14)

Verifier: `scripts/stage3_verify_frozen_artifacts.py`
Evidence: `results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json`

| Artifact | Manifest SHA256 (recorded 2026-09-10) | raw | LF-normalised | Status |
|---|---|---|---|---|
| `models\lstm_pytorch_v3_logtarget\best_model.pt` | `448cb9659a91ea21…` | MATCH | n/a | ✅ UNCHANGED |
| `models\lstm_pytorch_v3_logtarget\log_target_scaler.pkl` | `63d7325e2ad61bde…` | MATCH | MATCH | ✅ UNCHANGED |
| `results\lstm_pytorch_v3_logtarget\test_predictions_original_units.csv` | `dadbfabab2e8ef6e…` | differs | **MATCH** | ✅ UNCHANGED |

**Verdict: ALL FROZEN ARTIFACTS UNCHANGED** (exit code 0).

### 10.1 IMPORTANT — the CSV "mismatch" is a git CRLF artifact, not a data change

The predictions CSV's **raw** working-tree SHA256 is
`9231323c7ffdbdd3b5667ca60e1ce951ae0febb8a6e2e48b5ded56926a71e094`
(86,117 bytes) and does **not** match the manifest. Investigated and explained:

* the manifest was computed on **LF** bytes (85,016 bytes);
* this checkout has `core.autocrlf=true`, so the working tree contains CRLF;
* normalising CRLF→LF yields 85,016 bytes and SHA256
  `dadbfabab2e8ef6e0e15313a9c57af516f8523f0e8e850f29a6160fbac400c5c`
  — an **exact match** to the manifest;
* `git show HEAD:<csv>` is byte-identical to the LF-normalised file
  (blob size 85,016; blob SHA256 identical);
* `git status` reports the file as **clean**.

**Conclusion: the frozen V3 artifact is byte-for-byte intact in the repository.**
The raw-hash difference is purely a working-tree line-ending artefact of
`core.autocrlf=true`. The verifier now reports both representations so this can
never be mistaken for tampering — and so a genuine content change still fails.

**Recommendation (Stage 4 input):** add a `.gitattributes` rule
(`*.csv text eol=lf`) so byte-level integrity checks are stable across machines.

---

## 11. Phase 15.3 Baseline/MPC Reproduction (Requirement 15)

Runner: `scripts/stage3_phase15_3_reproduction.py`
Evidence: `results/phase15_stage3_reproduction/REPRODUCTION_CHECK.json`

The Phase 15.3 script was **imported, not executed**, and its `OUTPUT_DIR` was
redirected to `results/phase15_stage3_reproduction/phase15_3_reproduction/`.
The protected `results/phase15_v3_validation/` directory was hashed before and
after.

### 11.1 Protection result

| Protected file | Before == After |
|---|---|
| `daily_simulation_baseline.csv` | ✅ UNCHANGED |
| `daily_simulation_mpc.csv` | ✅ UNCHANGED |
| `PHASE_15_3_V3_VALIDATION_REPORT.md` | ✅ UNCHANGED |
| `provenance_audit.json` | ✅ UNCHANGED |
| `v3_integrity_check.json` | ✅ UNCHANGED |
| `validation_metrics.csv` | ✅ UNCHANGED |

**6/6 protected artifacts untouched** (`protected_artifacts_untouched: true`).

### 11.2 Fidelity result

| File | Protected SHA256 | Reproduced SHA256 | Identical |
|---|---|---|---|
| `validation_metrics.csv` | `d35c7e500c6f9b62…` | `d35c7e500c6f9b62…` | ✅ |
| `daily_simulation_baseline.csv` | `3ffeb5ce93535f38…` | `3ffeb5ce93535f38…` | ✅ |
| `daily_simulation_mpc.csv` | `3aba273bc34a71a5…` | `3aba273bc34a71a5…` | ✅ |

**Byte-for-byte identical.** Stage 3 did not perturb the validated Phase 15.3 results.

### 11.3 Reproduced metrics (unchanged)

| Metric | Baseline | MPC | Diff | Verdict |
|---|---|---|---|---|
| Overflow events | 7 | 0 | −7 | MPC_BETTER |
| Overflow volume (MCM) | 10.7500 | 0.0000 | −10.7500 | MPC_BETTER |
| Downstream violations | 8 | 0 | −8 | MPC_BETTER |
| Peak downstream flow (MCM/day) | 60.0 | 30.0 | −30.0 | MPC_BETTER |
| Total release (MCM) | 2017.25 | 2031.00 | +13.75 | MPC_WORSE |
| Forecast dates used | 0 | 74 | — | — |
| Total steps | 74 | 74 | 0 | SAME |
| Mass-balance residual | −6.82e−13 | −6.82e−13 | 0.00e+00 | PASS |

---

## 12. Requirement Compliance

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Do NOT simply delete `VirtualCascade` | ✅ | §2.3 — class present, unmodified, still tested |
| 2 | Compare under identical initial states / inflows / gates / timestep | ✅ | §4 — 3-engine controlled harness |
| 3 | Document every semantic difference | ✅ | §6 — 12 dimensions incl. all requested topics; `semantic_differences.csv` |
| 4 | Preserve validated `ReservoirNetwork` behaviour | ✅ | §2.3 — file untouched; exact-equivalence proof §5.1 |
| 5 | Target topology A→B→C→D | ✅ | §3 |
| 6 | Target delays 2/1/1 | ✅ | §3 |
| 7 | Target attenuation 0.90/0.85/0.80 | ✅ | §3 |
| 8 | Do not modify validated MPC or `SafetyLayer` | ✅ | §2.3 |
| 9 | Do not change frozen LSTM artifacts | ✅ | §10 — all hashes verified |
| 10 | Prefer an adapter over modifying validated logic | ✅ | §7 |
| 11 | Add parity/regression tests | ✅ | §8 — 28 new tests |
| 12 | Do not remove `VirtualCascade` until safely replaced + proven | ✅ | Live path AST-free of `VirtualCascade`; class retained |
| 13 | Run the complete test suite | ✅ | §8 — 213 passed |
| 14 | Verify all frozen artifact SHA256 hashes | ✅ | §10 |
| 15 | Run Phase 15.3 reproduction safely | ✅ | §11 |
| 16 | Do NOT proceed to Stage 4 | ✅ | Stage 4 not started |

---

## 13. Reproduction Commands

```bash
# Complete test suite
python -m pytest -q

# Stage 3 controlled parity harness + semantic difference register
python scripts/stage3_physics_parity_report.py

# Frozen artifact SHA256 verification (exit 0 = all intact)
python scripts/stage3_verify_frozen_artifacts.py

# Safe Phase 15.3 baseline/MPC reproduction (protected artifacts shielded)
python scripts/stage3_phase15_3_reproduction.py
```

---

## 14. Metrics That Changed for the Live Twin — Honest Summary

Adopting the validated physics is a **material behavioural change** to the live
Digital Twin, not a no-op. Specifically:

1. **Upstream flood waves arrive later.** A→B travel time went 1 → 2 days.
2. **Less water reaches downstream.** 10% / 15% / 20% of routed volume is now
   lost in transit, so reservoirs B, C and D receive less routed inflow and the
   terminal flow is lower than the legacy model showed.
3. **Upstream overflows no longer inflate downstream inflow.** Spill at a
   non-terminal reservoir now leaves the network instead of being routed on.
4. **Spill is displayed separately from controlled release.** The legacy
   `release_mcm_day` meaning is preserved, but the underlying split is now visible.
5. **The twin now obeys a global mass balance** it previously could not verify.

These changes are **correct per the validated research model** and are the
intended outcome of Stage 3. Any dashboard narrative or operator-facing text
that quotes legacy live numbers should be re-checked against the new physics.

---

## 15. Known Limitations / Residual Risk

1. **Legacy JSON delay block is now dead weight on the live path** (§9.1). It is
   still read by the out-of-scope research simulator, so it was left alone.
2. **No `.gitattributes` yet** — byte-level integrity checks remain sensitive to
   `core.autocrlf` (§10.1). Recommended fix for Stage 4.
3. **Streamlit/Streamlit-page numbers may shift visually** because the live
   physics changed (delays/attenuation). This is expected, not a defect.
4. **No UI/manual verification was performed** — only automated tests. A visual
   dashboard pass is advisable before demonstration.
5. **`tests/test_streamlit_simulation.py` still uses high default inflows**
   (A:10, B:20, C:100 MCM/day) that exceed some release capacities; behaviour is
   correct but the scenario is a stress case, not a typical operating point.

---

## 16. Rollback

Reverting the live physics to the legacy cascade requires exactly two edits:

1. `src/dashboard/sim_bridge.py` — restore
   `from src.simulator.environment import VirtualCascade` and
   `self.cascade = VirtualCascade(run_config)` in `init_cascade`.
2. `src/network_env/__init__.py` — drop the `live_cascade_adapter` export.

`VirtualCascade` is untouched, so this restores the pre-Stage-3 live behaviour
exactly. No data migration or artifact regeneration is required.

---

*End of Stage 3 Report. Stopped at the Stage 3 boundary — Stage 4 not started.*
