# STAGE 16 — PHASE 15.3 REGRESSION HARD GATE — COMPLETE

**Status:** COMPLETE — Stage 16 only.
**Date:** 2026-09-16
**Nature:** regression gate, not feature development. No AI, GNN, LSTM, MPC,
SafetyLayer, physics or live-architecture change was made.

---

## 1. Scope

Prove that the engineering/integration work of **Stages 0–15 did NOT change the
validated Phase 15.3 research result**.

In scope: re-run the canonical Phase 15.3 reproduction exactly as previously
validated, verify the hard expected metrics, verify frozen-artifact and
protected-output hashes, verify topology / MPC / SafetyLayer / physics identity,
prove the scientific sources were not modified, and record the live-vs-research
distinction.

Out of scope (and not done): any optimisation, any model or controller change,
any relaxation of the live provenance rules, Stage 17.

### Two paths named in the Stage 16 brief do not exist

| Path named in the brief | Actual path | Exists |
|---|---|---|
| `src/simulator/reservoir_network.py` | `src/network_env/reservoir_network.py` | ❌ / ✅ |
| `src/forecasting/v3_forecast_adapter.py` | `src/network_env/v3_forecast_adapter.py` | ❌ / ✅ |

The authoritative physics engine and the V3 forecast adapter both live under
`src/network_env/`. There is no `src/forecasting/` package and no
`reservoir_network.py` under `src/simulator/` (that package holds the offline
Phase 14.4 research engine). The correct modules were audited.

---

## 2. Frozen artifacts

Manifest: `results/phase15_v3_validation/v3_integrity_check.json`
(recorded `2026-09-10T15:04:00.918745`, `all_match: true`, *"V3 artifacts
UNCHANGED"*).

| Artifact | Expected SHA256 | Actual SHA256 | Verdict |
|---|---|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | `448cb9659a91ea21…58632d` | `448cb9659a91ea21…58632d` | **MATCH** |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | `63d7325e2ad61bde…114781` | `63d7325e2ad61bde…114781` | **MATCH** |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | `dadbfabab2e8ef6e…c400c5c` | raw `9231323c7ffdbdd3…a71e094` · LF-normalised `dadbfabab2e8ef6e…c400c5c` | **MATCH (LF-NORMALISED)** |

**The known Windows CRLF case, stated precisely:**

* The manifest records the **LF-normalised** hash `dadbfab…`. The working-tree
  file uses CRLF (`core.autocrlf=true`), so its **raw** hash differs.
* The size difference is exactly the line-ending overhead, verified numerically:
  raw `86 117` bytes = LF `85 016` bytes (the manifest's recorded size) +
  **1 101** CRLF pairs × 1 byte.
* The file is therefore **byte-identical to the frozen artifact** under LF
  normalisation. **No line ending was "fixed"**, globally or locally; nothing
  was rewritten. `best_model.pt` and `log_target_scaler.pkl` match raw (binary,
  no line-ending ambiguity).

Post-run re-hash: all three **unchanged** by the reproduction and by the tests.

---

## 3. Canonical reproduction command

```text
py scripts/stage3_phase15_3_reproduction.py
```

That script imports (does not execute)
`scripts/run_phase15_3_validation.py`, redirects its module-global `OUTPUT_DIR`
to `results/phase15_stage3_reproduction/phase15_3_reproduction/`, hashes
`results/phase15_v3_validation/` before and after, and compares the three
re-derived outputs byte-for-byte against the protected ones.

**Exit code 0. Runtime 147.7 s.** No assumption was changed, no new benchmark
was created, and no script was modified.

---

## 4. Dataset and forecast provenance

From the protected `provenance_audit.json`:

| Fact | Value |
|---|---|
| Forecast model | **LSTM_V3** (frozen) |
| Forecast artifact | `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` |
| Records | **1 100** |
| Unique dates | **74** (the evaluation horizon; `total_steps = 74` in both runs) |
| Unique reservoirs | **16** |
| Adapter | `V3ForecastAdapter` (read-only) |
| Cascade mapping | `Reservoir_A→Anayirankal`, `Reservoir_B→Ponmudi`, `Reservoir_C→Idamalayar`, `Reservoir_D→Idukki` |
| Mapping classification | **`ASSUMED_FOR_PROTOTYPE`** (from `topology_config.yaml`) |
| Topology classification | **`ASSUMED_FOR_PROTOTYPE`** |
| Network provenance summary | `OBSERVED 4 · VERIFIED 0 · ASSUMED_FOR_PROTOTYPE 16` |
| Forecast semantics | point forecasts of inflow (MCM/day) at forecast_date **+1 / +3 / +7** days |
| Recorded warnings | *"POINT FORECASTS, NOT cumulative volumes" · "Do NOT multiply by horizon length" · "Do NOT interpolate between horizons" · "Reservoir mapping is ASSUMED_FOR_PROTOTYPE"* |

Both simulations consume the **same** observed inflows for fairness (the 1-day
actual from the frozen CSV), and the MPC additionally consumes the full
`NetworkForecastSnapshot`.

Honest note: the frozen provenance audit records the artifact path **as it was
on the original authoring machine** (`…\Downloads\AI-BASED-MULTI-RESERVOIR-CURRENT\…`).
It is a frozen artifact and was not edited. The reproduction reads the artifact
from the current repository root and still produces byte-identical outputs, which
shows the recorded path is provenance metadata rather than an input dependency.

---

## 5. Topology verification

`ReservoirNetwork(config_dict=topology_config.yaml)`:

| Property | Value | Expected | OK |
|---|---|---|---|
| Node ids | `Reservoir_A, Reservoir_B, Reservoir_C, Reservoir_D` | same | ✅ |
| Chain | `A→B`, `B→C`, `C→D` | same | ✅ |
| Terminal node | `Reservoir_D` (Idukki) | same | ✅ |
| Routing delays | **`[2, 1, 1]`** | `2/1/1` | ✅ |
| Attenuation | **`[0.90, 0.85, 0.80]`** | `0.90/0.85/0.80` | ✅ |
| Downstream capacity | **50.0 MCM/day** | (authoritative, not an invented threshold) | ✅ |
| Timestep semantics | one `step()` advances exactly **+1** | unchanged | ✅ |
| Mass balance on a step | `abs(residual) < 1e-9` | conserved | ✅ |

Storage / inflow / release / spill equations are those of the unmodified
`reservoir_network.py` (see §18).

---

## 6. MPC verification

| Check | Result |
|---|---|
| Candidate space | **6⁴ = 1296** joint gate vectors |
| Gate levels | `[0.0, 0.15, 0.3, 0.5, 0.7, 1.0]` (6 per reservoir) |
| Lookahead | 3 steps |
| Max gate change (rate limit) | 0.5 |
| Candidates actually evaluated | **1296 on every one of the 74 days** (from the canonical log) |
| Reservoirs optimised | **all four**, jointly (`Reservoir_A…D`) |
| Forecast actually used | `forecast_used == True` on **74/74** days |
| Direct single-decision proof | one real decision on the canonical date returned gates for **all four** reservoirs and `candidates_evaluated == 1296` |

---

## 7. SafetyLayer verification

The canonical run shows the SafetyLayer **in the loop and acting**:

| Evidence | Value |
|---|---|
| `mpc_status` values in the canonical log | `OPTIMAL`, **`CORRECTED`** |
| Days corrected by the SafetyLayer | **2** |
| Behaviour | unchanged Phase 15.3 behaviour (bounds + per-step rate limit); no new rule, no relaxation |

A `CORRECTED` status cannot appear unless the layer modified an MPC proposal, so
the canonical benchmark is not merely *calling* the layer — it is exercising it.

---

## 8. Baseline metrics

`validation_metrics.csv` (protected, re-derived identically):

| Metric | Expected | Actual | Exact |
|---|---|---|---|
| `overflow_events` | 7 | **7** | ✅ |
| `overflow_volume_mcm` | 10.75 | **10.75003168999999** | ✅ |
| `ds_violations` | 8 | **8** | ✅ |
| `peak_ds_flow` | 60 | **60.0** | ✅ |
| `total_release_mcm` | 2017.25 | **2017.25** | ✅ |
| `total_steps` | 74 | **74** | ✅ |

---

## 9. MPC metrics

| Metric | Expected | Actual | Exact |
|---|---|---|---|
| `overflow_events` | 0 | **0** | ✅ |
| `overflow_volume_mcm` | 0 | **0.0** | ✅ |
| `ds_violations` | 0 | **0** | ✅ |
| `peak_ds_flow` | 30 | **30.0** | ✅ |
| `total_release_mcm` | 2031.00 | **2031.0** | ✅ |
| `forecast_dates_used` | 74 | **74** | ✅ |

Comparison is **exact string equality** against the protected CSV — not
"approximately similar". `total_release_mcm` remains `+13.75` (verdict
`MPC_WORSE`) in the frozen record: the MPC spends slightly more release to avoid
all overflow and all downstream violations. That trade-off is unchanged and was
not "improved".

---

## 10. Mass-balance verification

| Quantity | Expected | Actual | Exact |
|---|---|---|---|
| Baseline residual | −6.82e−13 | **−6.821210263296962e-13** | ✅ (bit-exact float) |
| MPC residual | −6.82e−13 | **−6.821210263296962e-13** | ✅ (bit-exact float) |
| Verdict in the frozen CSV | `PASS` | `PASS` | ✅ |

The residual is 3.5 orders of magnitude inside the `1e-9` tolerance and is
identical to the last bit, so no floating-point drift was introduced anywhere in
the pipeline.

---

## 11. Candidate-space verification

```text
len(gate_levels) ** 4  =  6 ** 4  =  1296      (computed from MPCConfig defaults)
candidates_evaluated in the canonical log      =  {'1296'}   (all 74 days)
```

This is the **original** search space. It was not widened, narrowed, pruned or
replaced.

---

## 12. D-reservoir optimization proof

Reservoir D (Idukki) must be an ordinary decision variable, never pinned.

| Source | `Reservoir_D_gate` values | Distinct |
|---|---|---|
| Canonical MPC log (74 days) | `{0.0, 0.15}` | **2** |
| Canonical baseline log (74 days) | `{0.05, 0.3}` | **2** |

A pinned D could only ever repeat a single value, so D is demonstrably
optimised. For completeness, **all four** reservoirs vary across the canonical
MPC run: A `{0.0, 0.15, 0.3, 0.5}`, B `{0.15, 0.3, 0.5, 0.65, 0.7, 1.0}`,
C `{0.0, 0.15}`, D `{0.0, 0.15}`.

---

## 13. Output / hash comparison

**Protected Phase 15.3 outputs** (six files) — hashed before and after the
reproduction and after the whole test suite:

| File | Before == After |
|---|---|
| `validation_metrics.csv` | ✅ UNCHANGED (`d35c7e50…`) |
| `daily_simulation_baseline.csv` | ✅ UNCHANGED (`3ffeb5ce…`) |
| `daily_simulation_mpc.csv` | ✅ UNCHANGED (`3aba273b…`) |
| `provenance_audit.json` | ✅ UNCHANGED (`4927e3cb…`) |
| `v3_integrity_check.json` | ✅ UNCHANGED (`eeae3871…`) |
| `PHASE_15_3_V3_VALIDATION_REPORT.md` | ✅ UNCHANGED (`49ef558c…`) |

**Re-derived vs protected outputs** (the fidelity requirement):

| File | Identical |
|---|---|
| `validation_metrics.csv` | ✅ byte-identical |
| `daily_simulation_baseline.csv` | ✅ byte-identical |
| `daily_simulation_mpc.csv` | ✅ byte-identical |

**3/3 byte-identical.** The reproduction runs the *same* procedure — including
its own `json.dump` / `csv.writer` serialisation, so timestamps and float
formatting are part of the comparison.

---

## 14. Stage 16 tests

`tests/test_stage16_regression_hard_gate.py` → **25 passed** (138.20 s), exit 0.

| Requirement from the brief | Test |
|---|---|
| canonical Phase 15.3 reproduction | `test_canonical_reproduction_reproduces_the_frozen_outputs_byte_identically` (runs the real script, requires exit 0, requires the protected dir unchanged, requires byte-identity) |
| exact expected metrics | `test_canonical_metrics_are_exact` (parametrised over all 7 metrics) |
| frozen artifact hashes | `test_frozen_v3_artifacts_match_the_recorded_manifest`, `test_prediction_csv_crlf_case_is_understood_not_hidden` |
| output identity | `test_the_last_reproduction_outputs_are_still_byte_identical`, `test_protected_outputs_match_the_recorded_reproduction_hashes` |
| topology identity | `test_topology_identity_from_the_canonical_config`, `test_physics_equations_and_timestep_semantics_unchanged` |
| 6⁴ candidate count | `test_candidate_space_is_six_to_the_four`, `test_canonical_logs_were_produced_with_1296_candidates_every_day` |
| D is optimized | `test_reservoir_d_is_optimised_not_pinned_in_the_canonical_run`, `test_a_single_phase15_3_mpc_decision_covers_all_four_reservoirs` |
| mass-balance residual | `test_mass_balance_residual_is_the_canonical_float` |
| no live-path modification can silently alter the benchmark | `test_phase15_3_pipeline_imports_only_scientific_modules`, `test_no_live_module_is_reachable_from_the_benchmark`, `test_scientific_sources_are_unmodified_since_the_validated_baseline` |
| live provenance not relaxed | `test_live_provenance_gate_was_not_relaxed_to_pass_this_gate` |

Companion evidence: `scripts/run_stage16_regression_hard_gate.py` →
**17/17 checks PASS**, exit 0
(`results/phase15_stage16_regression_hard_gate/stage16_regression_evidence.json`).

---

## 15. Stage 3–15 regression

All `tests/test_stage{3..15}_*.py` → **542 passed**, **exit 0** (237.45 s).
(Stage 3–10: 324 · Stage 11: 47 · Stage 12: 37 · Stage 13: 48 · Stage 14: 30 ·
Stage 15: 56.)

Spot-checks of the most recent stages individually:

| Suite | Result |
|---|---|
| Stage 15 end-to-end | **56 passed**, exit 0 |
| Stage 14 GNN advisory | **30 passed**, exit 0 |

No regressions.

---

## 16. Full-suite regression

`py -m pytest -q --no-header -p no:cacheprovider` → **752 passed**, **exit 0**,
732.87 s (12:12).

727 (Stage-15 total) + 25 (Stage 16) = 752. Zero failures, zero errors.

---

## 17. Runtime

| Measurement | Value |
|---|---|
| **Canonical Phase 15.3 reproduction** | **147.70 s** (exit 0) |
| Same script inside the Stage 16 audit | 147.70 s |
| Stage 16 test suite (includes the reproduction) | 138.20 s |
| Stage 3–15 regression | 237.45 s |
| Full suite | 732.87 s |

The reproduction dominates because the MPC performs 1296-candidate searches over
74 days. **Nothing was optimised** — these numbers are evidence only. The
practical consequence is that the full suite now takes ~12 minutes instead of
~3, because the hard gate re-runs the canonical benchmark (see §21.2).

---

## 18. Scientific integrity

**The single strongest result of this stage: not one line of the Phase 15.3
scientific computation has changed.**

`git diff --quiet HEAD -- <path>` for every scientific source → **all CLEAN**,
and each was last committed by the Phase 15.3 work itself:

| File | Working tree vs HEAD | Last commit |
|---|---|---|
| `src/controller/mpc_controller.py` | CLEAN | `2dbc3fc` 2026-09-10 *Complete Phase 15.3 Validation and Audits* |
| `src/controller/safety.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `src/controller/objective.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `src/controller/baseline_controller.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `src/network_env/reservoir_network.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `src/network_env/v3_forecast_adapter.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `src/network_env/topology_config.yaml` | CLEAN | `2dbc3fc` 2026-09-10 |
| `scripts/run_phase15_3_validation.py` | CLEAN | `2dbc3fc` 2026-09-10 |
| `scripts/stage3_phase15_3_reproduction.py` | CLEAN | `e3e1788` 2026-09-16 (Stage 3 harness) |

The commit `2dbc3fc` (2026-09-10) predates all Stage 10–15 work (2026-09-15/16,
present as uncommitted working-tree changes). Combined with the byte-identical
reproduction, this closes the question from both directions: the sources are
untouched **and** the computation they perform is unchanged.

Per requirement:

* **MPC** — candidate generation (6 levels × 4 reservoirs), objective,
  trajectory simulation and all-four-reservoir optimisation: untouched; 1296
  candidates logged every day.
* **SafetyLayer** — validated behaviour preserved (bounds + rate limit); it
  corrected 2 of 74 canonical days exactly as before.
* **ReservoirNetwork** — topology, delays `2/1/1`, attenuation
  `0.90/0.85/0.80`, storage/inflow/release/spill equations and one-timestep
  semantics all unchanged.
* **Forecast** — the frozen V3 artifacts, the `V3ForecastAdapter` and the
  horizon handling (`+1/+3/+7` point forecasts) are unchanged.

---

## 19. Live-vs-research distinction

**What this gate proves:** the validated **historical research pipeline**
(LSTM V3 frozen predictions → forecast adapter → MPC → SafetyLayer →
ReservoirNetwork → metrics) still produces its canonical result.

**What it does NOT prove:** that the current live Digital Twin has a validated
real-world forecast source. It does not, and nothing was changed to pretend
otherwise.

Verified in the same run:

| Live check | Result |
|---|---|
| Live AI-mode forecast eligibility | **False** |
| Live final action source | **`HELD_CURRENT_GATES`** |
| Live blocked reason | `FORECAST_NOT_ELIGIBLE_FOR_CONTROL` |
| `live_forecasts_are_validated` | **False** |

The live provenance gate is **still blocking**. `SYNTHETIC_DEMO` was not promoted
to `VALIDATED`, the provenance gate was not bypassed, and no live provenance rule
was relaxed to make this regression pass. The live path blocks for two
independent reasons (demonstration-only live forecasts; Reservoir D has no live
forecast), exactly as Stage 15 documented.

---

## 20. File hygiene

Stage 16 created and then removed only its own scratch files
(`tmp_s16t.txt`, `tmp_s16full.txt`, plus two `%TEMP%` capture files). **No
pre-existing user file was deleted.** The `tmp_full12.txt`, `tmp_reg12.txt` and
`tmp_s12.txt` files in the repository root are pre-existing and untouched.

```text
$ git status --short
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
?? results/phase15_stage16_regression_hard_gate/
?? scripts/run_stage13_streamlit_runtime_audit.py
?? scripts/run_stage14_gnn_advisory_audit.py
?? scripts/run_stage15_end_to_end_audit.py
?? scripts/run_stage16_regression_hard_gate.py
?? src/modeling/gnn_advisory.py
?? tests/test_stage13_streamlit_command_proxy.py
?? tests/test_stage14_gnn_advisory.py
?? tests/test_stage15_end_to_end_integration.py
?? tests/test_stage16_regression_hard_gate.py
```

### Stage 16's own footprint

| File | Why |
|---|---|
| `tests/test_stage16_regression_hard_gate.py` | **new** — the 25 hard-gate tests |
| `scripts/run_stage16_regression_hard_gate.py` | **new** — evidence harness |
| `results/phase15_stage16_regression_hard_gate/` | **new** — evidence JSON + this report |

**Stage 16 modified no existing file.** No scientific source, no frozen artifact,
no protected output and no live-architecture file was touched.

### Every other entry, explained

| Entry | Cause |
|---|---|
| `src/controller/live_mpc_orchestrator.py` | **Stage 15** — a comment correction (SafetyLayer invocation count). Not part of Phase 15.3. |
| `src/dashboard/api/state_manager.py`, `src/dashboard/app.py`, `src/dashboard/twin_component/state_adapter.py`, `src/dashboard/web/index.html`, `src/modeling/gnn_inference.py`, `src/network_env/gnn_forecast_adapter.py`, `src/modeling/gnn_advisory.py` | **Stages 13–14** — live Digital Twin / GNN advisory work. None on the Phase 15.3 path. |
| `scripts/run_stage12_state_authority_audit.py` | **Stage 12** — audit-probe correction. |
| `results/phase15_stage12_…/PHASE_15_STAGE12_REPORT.md` | **Stage 12** — report finalisation. |
| `results/phase15_stage3_reproduction/{REPRODUCTION_CHECK.json, phase15_3_reproduction/*}` | **Regenerated by the reproduction runs** (Stage 3 harness by design). These are the *harness outputs*, **not** the protected Phase 15.3 result: the protected directory is untouched, and only timestamps differ. |
| `results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json` | **Stage 3 verification harness** — timestamp only. |
| All `?? results/phase15_stage1{2,3,4,5,6}_*` and `?? tests/test_stage1{3,4,5,6}_*`, `?? scripts/run_stage1{3,4,5,6}_*` | New evidence/test artifacts from the completed Stages 12–16. |

Nothing was committed or pushed.

---

## 21. Limitations

1. **The CRLF caveat is real and unresolved by design.** The prediction CSV's
   *raw* hash does not equal the manifest's recorded hash; only the
   LF-normalised hash does. The brief explicitly forbids a global line-ending
   fix, so this is documented rather than "corrected". Any future verification
   must normalise CRLF when hashing this file, or the gate will report a false
   mismatch.
2. **The full suite is now expensive (~12 min).** The hard gate re-runs the
   canonical reproduction (≈148 s) inside pytest, and the Stage 15 suite is
   ~173 s. That is the cost of making the gate real rather than asserting a
   stored hash; it is a deliberate trade-off, not an oversight.
3. **The topology is still `ASSUMED_FOR_PROTOTYPE`.** The protected provenance
   audit records `OBSERVED 4 · VERIFIED 0 · ASSUMED_FOR_PROTOTYPE 16`, and the
   reservoir→cascade mapping is likewise assumed. The reproduction is faithful to
   that assumption; it does not make the assumption truer. This is unchanged by
   Stage 16 and must not be read as validation of real-world connectivity.
4. **`total_release_mcm` gets worse under the MPC** (`+13.75`, verdict
   `MPC_WORSE`) in the canonical result. That is the original, unchanged
   trade-off — the MPC releases slightly more water to eliminate all overflow and
   all downstream violations. It was not "improved" to make the gate look better.
5. **The frozen `provenance_audit.json` embeds an absolute path from the original
   authoring machine** (`…\Downloads\AI-BASED-MULTI-RESERVOIR-CURRENT\…`). It is a
   protected artifact and was not edited; the reproduction reads from the current
   repository root and still reproduces byte-identically.
6. **The gate covers the research pipeline, not the live twin.** See §19. A
   passing Phase 15.3 gate says nothing about the live system's forecast
   provenance, which remains `DEMONSTRATION_ONLY` with Reservoir D unavailable.
7. **The canonical metrics are compared as strings** (exact equality against the
   protected CSV), plus a bit-exact float comparison for the mass-balance
   residual. That is stricter than a tolerance check, so a formatting change in
   the harness would fail the gate even if the physics were identical — which is
   the intended behaviour for a byte-fidelity gate, but worth knowing.

---

## 22. Final verdict

| Hard gate | Result |
|---|---|
| Canonical reproduction exits 0 | ✅ |
| Frozen V3 artifacts match the manifest | ✅ |
| Frozen artifacts unchanged by the run | ✅ |
| Protected Phase 15.3 outputs unchanged (6/6) | ✅ |
| Re-derived outputs byte-identical (3/3) | ✅ |
| Baseline metrics exact (7 / 10.75 / 8 / 60 / 2017.25) | ✅ |
| MPC metrics exact (0 / 0 / 0 / 30 / 2031.0) | ✅ |
| Mass-balance residual bit-exact (−6.821210263296962e-13) | ✅ |
| Topology identity (A→B→C→D, delays 2/1/1, attenuation .90/.85/.80) | ✅ |
| Candidate space 6⁴ = 1296 (logged every day) | ✅ |
| All four reservoirs optimised; D not pinned | ✅ |
| Scientific sources unmodified since the validated baseline | ✅ |
| Live provenance gate not relaxed | ✅ |

```text
Stage 16 tests               25 passed   (exit 0)
Stage 3–15 regression       542 passed   (exit 0)
Full suite                  752 passed   (exit 0)
Stage 16 audit             17/17 checks PASS (exit 0)
Canonical reproduction      exit 0 · 147.70 s · 3/3 byte-identical
```

# FINAL VERDICT: PASS

The validated Phase 15.3 research result **survived Stages 0–15 unchanged**: the
same dataset, the same frozen V3 artifact, the same adapter, topology, delays,
attenuation, objective, candidate grid, SafetyLayer behaviour, horizon, baseline
controller and metrics — producing byte-identical outputs and a bit-exact
mass-balance residual.

**STAGE 17 HAS NOT BEEN STARTED.**
