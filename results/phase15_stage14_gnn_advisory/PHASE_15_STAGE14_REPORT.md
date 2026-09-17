# STAGE 14 — GNN SPATIAL DEPENDENCY / ADVISORY INTEGRATION — COMPLETE

**Status:** COMPLETE — Stage 14 only.
**Date:** 2026-09-16
**Goal:** integrate the existing experimental GNN honestly as a **spatial
dependency / advisory** component, with provable control-inertness.

Stages 3–13 were treated as the protected baseline and were not reopened. No
regression was found in them, so none was touched.

---

## 1. Executive summary

The forensic inspection found that the GNN's scientific role in the repository
was **already correct but invisible, and its claims were overstated**:

* The advisory inference ran on **every** state read and its result was then
  **thrown away**. Nothing consumed it: `self.gnn_adapter.update_from_inference`
  filled a cached snapshot that no code ever read, and the FastAPI/WebSocket
  payload carried no GNN information at all.
* Meanwhile the Digital Twin **claimed** `Forecast → "LSTM V3 + GNN"` and
  `GCN-LSTM GNN → LIVE` — hardcoded strings with **zero payload backing**. The
  first implied the experimental GNN was part of the forecast path (it is not),
  the second claimed a live GNN with nothing behind it.
* The control forecasts were built **through the GNN adapter**
  (`get_control_forecasts(..., policy="lstm_primary")`). That call happens to
  return LSTM-only values, but the same adapter also implements two policies that
  **would** inject GNN numbers into control (`gnn_primary`, `risk_envelope`).
  Nothing prevented a future edit from selecting one.

Stage 14 therefore did three things:

1. **Exposed** the advisory as a structured, provenance-carrying
   `gnn_advisory` block in the authoritative payload, including the model's real
   learned node representations and an explicitly labelled *embedding
   similarity* — computed from real inference, never invented.
2. **Hardened** the advisory boundary with a **fail-closed guard**: the cycle is
   refused if any control horizon ever differs from the validated frozen-LSTM
   value. "The GNN cannot reach control" is now a property of the running system
   rather than a reading of its source.
3. **Corrected the scientific claims** in the live UI and the GNN's own
   docstrings, and made the twin's GNN rows payload-driven.

The mandated **A/B control-inertness test** was run and passes with a
**non-trivial** decision: with a normal, a modified and a disabled advisory the
pipeline produced an identical fully-applied decision
(`MPC OPTIMAL → SAFETY SAFE → PROTECTED → FINAL action DOWNSTREAM_CAPACITY_GUARD`,
gates A 15 % / B 0 % / C 0 % / D 0 %), and the entire authoritative payload was
byte-identical except the advisory block and the wall-clock stamp.

**STAGE 15 HAS NOT BEEN STARTED.**

---

## 2. Existing GNN forensic inventory

### Active experimental implementation

| Question | Answer |
|---|---|
| Model class | `GatedGCNLSTM` (`src/modeling/gnn_inference.py`, a verbatim copy of the training class in `src/modeling/train_gcn_lstm_gated_v1.py`) |
| Active checkpoint | `models/gcn_lstm_gated_v1/best_model.pt` (188,149 bytes, a raw 15-tensor `state_dict`) |
| Parameter count | **45,636** (counted from the checkpoint at run time, and independently from the constructed model) |
| Inference entry point | `LiveGNNForecaster` (`predict_all`, `predict_from_flat_arrays`, `get_diagnostics`) |
| Live wiring | `src/dashboard/api/state_manager.py` → `_build_gnn_history()` + `gnn_forecaster.predict_all()` |
| Application adapter | `src/network_env/gnn_forecast_adapter.py` (`GNNForecastAdapter`, snapshot/bundle schemas) |
| Tensor shape | `(B, 16, 7, 5)` |
| Node order | `CANONICAL_NODE_ORDER` — 16 Kerala reservoirs, alphabetical |
| Feature order | `[inflow, water_level, live_storage, rainfall, total_outflow]` |
| Targets | `target_1d`, `target_3d`, `target_7d` in MCM/day (log1p + StandardScaler, inverted with `inverse_transform → expm1`) |
| Graph artifacts | `data/processed/graph/graph_D_correlation_v1_2/{edges.csv,metadata.json}` |
| Graph builders present | A (correlation), B (geographic k-NN), C (hybrid), D (correlation v1.2), `build_reservoir_graph.py` |
| Other checkpoints present | `gcn_lstm_v1/`, `gcn_lstm_v1_1_identity/`, `gcn_lstm_v1_2/` — the earlier ablation variants, **not** on any live path |
| Consumers of the output | **None, before Stage 14.** After Stage 14: the `gnn_advisory` block, rendered by the twin and Streamlit. Still none in control. |

The **Gated GCN-LSTM V1** is the active experimental implementation. The other
three checkpoints are the frozen ablations referenced by
`results/gcn_lstm_gated_v1/MILESTONE_12_8_REPORT.md` and were left untouched.

---

## 3. Exact GNN architecture

```text
input (B, 16, 7, 5)                    5 features × 7 days × 16 nodes, scaled
        │
        ├── LOCAL branch    LSTM(5 → 64)                  → h_local   (64)
        │
        └── SPATIAL branch  GCNConv(5 → 32, self-loops, normalized) + ReLU
                            → permute → LSTM(32 → 64)     → h_spatial (64)
        │
   gated fusion   h = h_local + sigmoid(alpha) · h_spatial        (64)
        │
   shared head    FC(64 → 32) → ReLU → FC(32 → 3)
        │
   output (B, 16, 3)  →  target_1d / target_3d / target_7d (MCM/day)
```

| Fact | Value | Source |
|---|---|---|
| Parameters | **45,636** | counted from `best_model.pt` and from the constructed model |
| Embedding width (fused hidden) | **64** | `EMBEDDING_DIM` (new Stage 14 constant) |
| Learned gate `alpha` | **−4.1778779** | checkpoint tensor |
| Learned gate `sigmoid(alpha)` | **0.0150995** | matches `results/gcn_lstm_gated_v1/gate_diagnostics.json` (`final_gate = 0.0150995`) |
| Gate interpretation (from the artifact) | `gate < 0.05 = effectively ignored spatial` | `gate_diagnostics.json` |

The gate is the most important architectural fact: at **≈ 1.5 %** the model
itself learned to ignore almost all of the graph signal. This is why the GNN is
presented as an advisory *representation* and never as a predictor that should
drive anything.

### What Stage 14 added to the inference API (no redesign, no retraining)

| Addition | Nature |
|---|---|
| `EMBEDDING_DIM = 64` | constant |
| `_build_input_tensor()` | the tensor-building block **extracted verbatim** from `predict_all`, so the new accessor cannot drift from the prediction path |
| `node_representations()` | returns the fused `h` that `forward()` already computes then discards — **`forward()` itself is untouched** |
| `graph_provenance()` | reads the graph's own `metadata.json` / `edges.csv` |

Parity is not asserted, it is **measured**: applying the model's own FC head to
the exposed representations reproduces `forward()`'s output with a maximum
absolute difference of **0.0** (bit-identical), in both the tests and the
evidence script.

---

## 4. Graph construction / provenance

| Field | Value |
|---|---|
| Graph | **Graph D (correlation_v1_2)** |
| Nodes | **16** |
| Undirected edges | **41** |
| Directed edges | **82** (bidirectional `edge_index`, asserted `(2, 82)` at load) |
| Connected components | **2** |
| Isolated reservoir | **Pamba** (degree 0) |
| Min / max / average degree | 0 / 8 / 5.125 |
| Density | 0.3417 |
| Edge rule | *positive Pearson inflow correlation ≥ 0.55 with ≥ 365 common **training-period** dates* |
| Data window | `training only (via train.csv split)` |
| Leakage safe | `true` |

This is a **statistical correlation graph**. It is explicitly not a hydraulic or
routing network, and the payload says so in three places
(`graph_provenance.provenance_note`, `relationship_summary.note`,
`two_graphs_note`).

---

## 5. Existing validation metrics

Read from `results/gcn_lstm_gated_v1/gcn_lstm_gated_v1_metrics.csv` and
reproduced **unchanged** in the advisory block. A test asserts the block's copy
still equals the artifact, so the numbers cannot silently drift or be
embellished.

| Horizon | MAE | RMSE | R² | Bias | Negative predictions |
|---|---|---|---|---|---|
| 1-day | 2.342946 | 3.764655 | 0.628784 | 0.024434 | 0 |
| 3-day | 2.353850 | 3.869269 | 0.582139 | 0.150354 | 0 |
| 7-day | 2.736001 | 4.524306 | 0.511439 | 0.047422 | 0 |

These agree exactly with the figures supplied in the Stage 14 brief. **No
discrepancy was found.** They are published alongside the note that they are
*not* the validated forecasting baseline and that the validated V3 metrics do
**not** apply to this model.

---

## 6. Before / after architecture

### Before

```text
state_manager._run_ml_pipeline()
    ├─ lstm_forecasts          (validated frozen LSTM V3)
    ├─ GNN inference ──► gnn_adapter.update_from_inference(...)   [computed]
    │                        └─ cached snapshot … READ BY NOTHING      [discarded]
    └─ ctrl_forecasts = gnn_adapter.get_control_forecasts(..., "lstm_primary")
                              └─ returns LSTM values, but the adapter ALSO
                                 implements gnn_primary / risk_envelope
                                 — unguarded, one edit away from control

payload: reservoirs · cascade · control · mass_balance · forecast_* · …
                                                       (no GNN anything)

twin HUD:  Forecast  "LSTM V3 + GNN"     ← hardcoded, unsupported
           GCN-LSTM GNN  "LIVE"          ← hardcoded, no payload backing
```

### After

```text
state_manager._run_ml_pipeline()
    ├─ lstm_forecasts          (validated frozen LSTM V3)   ── UNCHANGED
    ├─ GNN inference ──► node_representations() + predict_all()
    │                        └─ build_gnn_advisory() ──► self.gnn_advisory
    │                                                        (DISPLAY ONLY)
    ├─ ctrl_forecasts = gnn_adapter.get_control_forecasts(..., "lstm_primary")
    └─ _assert_control_forecasts_are_lstm_only(...)   ← FAIL-CLOSED GUARD
             refuses the cycle if any horizon ≠ the validated LSTM value

payload: … + gnn_advisory { status, model, graph, embeddings,
                            embedding_similarity, provenance,
                            advisory_only: true, affects_control: false }

twin HUD:  Forecast       ← payload (LSTM V3)
           GNN ADVISORY   ← payload status
           GNN GRAPH      ← payload 16N / 41E
           GNN EMB. SIM   ← payload top embedding similarity
```

Control path (unchanged, and now guarded):

```text
Forecast → MPC → SafetyLayer → DownstreamCapacityGuard
        → FINAL_SAFE_CONTROL_ACTION → ReservoirNetwork
```

---

## 7. Advisory output schema

Published at `gnn_advisory` in `GET /api/state` and `/ws/state`:

| Field | Meaning |
|---|---|
| `status` | `AVAILABLE` / `UNAVAILABLE` / `WARMUP_INSUFFICIENT_HISTORY` |
| `reason` | why, when not available (`MODEL_NOT_LOADED`, `INSUFFICIENT_HISTORY_FOR_GNN`, `NON_FINITE_EMBEDDINGS`, …) |
| `model_name` / `model_version` | `Gated GCN-LSTM V1` / `gcn_lstm_gated_v1` |
| `graph`, `graph_nodes`, `graph_undirected_edges`, `graph_directed_edges` | `Graph D (correlation_v1_2)`, 16, 41, 82 |
| `inference_timestamp` | when this advisory was produced |
| `embedding_dimensions` | 64 |
| `node_embeddings` | 16 × 64 fused representations, rounded to 4 dp |
| `nodes_with_live_input` / `nodes_zero_padded` | **which nodes the representation is actually informed by** |
| `embedding_similarity` | `label`, `method`, `matrix` (16×16 cosine), `top_relationships`, `matrix_note`, `ranking_restricted_to` |
| `relationship_summary` | graph type/construction/counts/components, `learned_spatial_gate`, gate interpretation |
| `gate_value`, `inference_latency_ms` | sigmoid(alpha); measured cost |
| `validation_metrics` + `_source` + `_note` | the published metrics, their artifact, and the non-applicability statement |
| `graph_provenance` | the full statistical-graph provenance record |
| `physical_control_topology` | the **4-node** authoritative cascade with delays 2/1/1 and attenuations 0.90/0.85/0.80 |
| `two_graphs_note` | the explicit "do not conflate" statement |
| `advisory_only` | **`true`** |
| `affects_control` | **`false`** |
| `disclaimer` | "ADVISORY ONLY … Removing or altering it cannot change any control decision" |

**Nothing is fabricated.** On any failure the block reports `UNAVAILABLE` with a
reason and null payload values; when inference has not run yet the twin payload
carries an explicit `NO_GNN_ADVISORY_PROVENANCE` block with the *same* schema
(a test pins the adapter's default to the canonical key set).

### Embedding similarity — correct labelling

```text
similarity(i, j) = cosine(embedding_i, embedding_j)
```

presented as **"embedding similarity (cosine between learned node
representations) — NOT hydraulic influence, NOT causal influence, NOT physical
connectivity"**. A test asserts that every mention of a stronger claim appears
only as an explicit negation.

### A real numerical trap that was handled rather than hidden

The model **zero-pads** nodes with no live input (its documented training
contract). Two padded nodes therefore receive *identical* inputs and produce
near-identical representations — a cosine of **1.0 that reflects shared padding,
not spatial structure**. The first live run produced exactly that artefact at the
top of the ranking (`Anathode ↔ Banasura Sagar = 1.0`).

Stage 14 now reports which nodes are live-informed and **restricts the
relationship ranking to pairs where both endpoints have live input**, states the
reason in `matrix_note`, and still exposes the full matrix for transparency.
Node embeddings for padded nodes are disclosed, never presented as live analysis.

---

## 8. Proof that the GNN does not affect control

| Requirement | Mechanism | Evidence |
|---|---|---|
| GNN cannot change gate commands | the applied gates come only from `decision.gate_positions_pct`, produced by `LiveMPCOrchestrator.decide()`; the advisory is never assigned into `gate_commands` | source assertions + A/B |
| cannot bypass the MPC | `live_mpc_orchestrator.py` and `mpc_controller.py` import and read nothing GNN-related | AST import audit |
| cannot bypass the SafetyLayer | same, plus `safety.py` is GNN-free | AST import audit |
| cannot bypass the DownstreamCapacityGuard | same | AST import audit |
| `ReservoirNetwork` independent | `reservoir_network.py` contains no GNN reference or import | source + import audit |
| `MassBalanceMonitor` independent | `mass_balance.py` contains no GNN reference or import | source + import audit |
| frozen LSTM V3 remains the source | payload reports `LSTM_V3_LOGTARGET` / `FROZEN_UNMODIFIED`; every live forecast carries `FROZEN_LSTM_V3` | payload assertion |
| **fail-closed boundary** | `_assert_control_forecasts_are_lstm_only()` compares every control horizon against the validated LSTM value and **raises** on any difference | test drives the guard with a tampered forecast and asserts it fires |

The guard is the substantive addition: previously the inertness depended on the
*literal string* `"lstm_primary"`. Now, even if that policy is changed, the
pipeline refuses to control with an unvalidated forecast source instead of
silently using one.

---

## 9. A/B control-inertness test (mandatory)

### Part 1 — decision level (real, fully-applied decision)

Identical network and identical forecast bundle, three runs; the only variable is
the advisory object.

| Run | Advisory |
|---|---|
| **A** | real advisory from live inference |
| **B** | **modified**: embeddings replaced with a constant 12345.0 field, gate 0.999, `graph_nodes` 999 |
| **C** | **disabled**: `empty_block(...)` and `gnn_ready = False` |

Reference (Run A) — and **identical in B and C**:

```text
mpc_status                     OPTIMAL
mpc_proposal_fraction          A 0.15  B 0.00  C 0.00  D 0.00
controller_status              ACTIVE
control_applied                true
safety_layer_status            SAFE          (safety_modified: false)
downstream_status              PROTECTED     (capacity_achieved: true)
applied_gates_pct              A 15.0  B 0.0  C 0.0  D 0.0
final_safe_control_action_pct  A 15.0  B 0.0  C 0.0  D 0.0
final_safe_control_action_src  DOWNSTREAM_CAPACITY_GUARD
network state (storage, release, spill, outflow, gate)   identical
```

This is deliberately **not** a trivially-blocked case: the MPC produced an
optimal proposal, the SafetyLayer passed it, the capacity guard protected the
downstream boundary and the action was **applied**.

### Part 2 — end to end through the authoritative payload

Simulation paused, history/gates/forecast inputs untouched between captures, so
the only variable is the GNN path:

| | Run A | Run B |
|---|---|---|
| `gnn_ready` | True | False |
| `gnn_advisory.status` | **AVAILABLE** | **UNAVAILABLE** (`MODEL_NOT_LOADED`) |
| Differing payload fields (all 17 top-level blocks compared) | **NONE** | |

`reservoirs`, `cascade`, `control`, `downstream`, `mass_balance`,
`forecast_provenance`, `forecast_summary`, `state_identity`, `simulation`,
`storm`, `hardware_status`, `metadata` — all identical. The only excluded keys
are the advisory itself and `simulation_time` (a per-payload wall-clock stamp,
excluded by design; the identity is `state_identity`).

Both parts are reproduced by the evidence script
(`scripts/run_stage14_gnn_advisory_audit.py`, 17/17 checks PASS).

---

## 10. API / WebSocket integration

* `gnn_advisory` was **added** to the single existing payload produced by
  `state_manager.get_adapted_state()`; it reaches `GET /api/state` and
  `/ws/state` through the same object, so **no second state authority was
  created**.
* Every pre-existing field is preserved. A Stage 14 test asserts REST and
  WebSocket publish the identical key set and that both carry the advisory.
* No new endpoint was added, and no GNN field is accepted as input anywhere —
  the command surface is unchanged (`POST /api/state` still `405`).
* The block is attached **after** the control path has produced its forecast, so
  nothing downstream of it can feed a decision.

---

## 11. Digital Twin integration

`src/dashboard/web/index.html` — display only:

| Before | After |
|---|---|
| header `Forecast → "LSTM V3 + GNN"` (hardcoded, wrong) | `data-ref="fceng"`, filled from `forecast_provenance.model` → **`LSTM V3`** |
| HUD row `GCN-LSTM GNN → "LIVE"` (hardcoded, unsupported) | three payload-driven rows: **GNN ADVISORY** (`gnn_advisory.status`), **GNN GRAPH** (`16N / 41E`), **GNN EMB. SIM** (top *embedding similarity*, with the label and the padding caveat as the row tooltip) |

The page **renders** and never computes: it does not run the GNN, does not build
an embedding, does not compute a similarity and does not derive a relationship.
A test asserts the page contains no model-runtime token (`torch`, `tensorflow`,
`onnx`, `GatedGCNLSTM`, `LiveGNNForecaster`, `gnn_inference`, `state_dict`,
`checkpoint`) and that it reads `state.gnn_advisory` from the payload.

Inline JavaScript was parse-checked with `node --check` after the change.

---

## 12. Streamlit integration

`src/dashboard/app.py` gained a read-only **"GNN ADVISORY — SPATIAL DEPENDENCY
(advisory only)"** panel showing status, model name/version, `affects control`,
node/edge counts, components, isolated reservoirs, the construction method and
edge rule, the strongest embedding similarity and the similarity label.

Streamlit remains exactly what Stage 13 made it: **read-only viewer + command
proxy**. It imports no GNN module (`gnn_inference`, `gnn_advisory`,
`LiveGNNForecaster`, `GatedGCNLSTM` — AST-verified), performs no
`node_representations` call and contains no `cosine` computation. There is **no
second GNN runtime**.

---

## 13. Scientific claim audit

Claims found and corrected — all directly Stage 14-related; no unrelated
documentation was rewritten.

| # | Location | Claim | Action |
|---|---|---|---|
| 1 | `web/index.html` (header) | `Forecast → "LSTM V3 + GNN"` — implied the experimental GNN is part of the forecast path | replaced with the payload's own forecast-engine name (`LSTM V3`) |
| 2 | `web/index.html` (HUD) | `GCN-LSTM GNN → "LIVE"` — hardcoded, no payload backing, implies an active role | replaced with payload-driven `GNN ADVISORY` / `GNN GRAPH` / `GNN EMB. SIM` |
| 3 | `src/network_env/gnn_forecast_adapter.py` | *"The results are then merged with LSTM forecasts into a combined forecast dict for the controller and risk engine."* — **false**, and exactly the kind of claim that would imply GNN→control | replaced with an explicit **ADVISORY SCOPE** section stating the GNN never reaches the controller or a live risk evaluation, that the alternative policies are unvalidated offline affordances, and that the live path additionally refuses the cycle fail-closed |
| 4 | `src/modeling/gnn_inference.py` (module docstring) | `Gated GCN-LSTM V1 (Production)` — over-claimed a model that underperformed the baseline and whose gate collapsed | retitled `(ADVISORY)` with an explicit scope section (experimental, not a forecaster, not a controller, not a safety mechanism, graph is statistical) |
| 5 | `src/modeling/gnn_inference.py` (class docstring) | `Production inference interface` | `Live inference interface … (ADVISORY use)` |

Wording added to the system (in the payload, not just the docs): *spatial
dependency representation*, *graph-based relational representation*,
*advisory spatial context*, *statistical dependency analysis*. Wording
deliberately **never** used about the GNN: causal discovery, proof of hydraulic
connectivity, gate controller, MPC replacement, safety mechanism, flood
guarantee.

Historical documents (`results/gcn_lstm_gated_v1/MILESTONE_12_8_REPORT.md`,
`results/v3_final_audit/…`, `results/phase15_*`) were **not** edited: they already
report the honest negative result, and rewriting frozen evidence reports would
damage the audit trail.

---

## 14. Stage 14 test count

`tests/test_stage14_gnn_advisory.py` → **30 passed** (0 failed).

Coverage of the required list: 1 discoverability/documentation;
2 structured advisory from real inference; 3 provenance; 4 `advisory_only`;
5 `affects_control`; 6–9 no gate change / no MPC, SafetyLayer or guard bypass;
10–11 `ReservoirNetwork` + `MassBalanceMonitor` independence; 12 frozen LSTM V3
still the source; 13 GNN labelled experimental/advisory; 14 physical topology
not conflated with the statistical graph; 15 graph provenance preserved;
16 similarity labelled as embedding similarity; 17 no browser inference;
18 no Streamlit inference; 19–20 A/B inertness with a changed and with a
disabled advisory. Plus: embedding/forward parity, similarity cross-check,
zero-padding disclosure, metrics-vs-artifact equality, no-fabrication on bad
input, and the guard-fires test.

The suite includes an autouse finalizer that restores the shared simulation to a
clean paused 50 % state after each test, so driving the singleton here cannot
leave other suites in a saturated corner (see §20.6).

---

## 15. Full test-suite count

`py -m pytest -q --no-header -p no:cacheprovider` → **671 passed** in 102.60 s
(641 Stage-13 baseline + 30 Stage 14). Zero failures, zero errors.

---

## 16. Stage 3–13 regression count

All `tests/test_stage{3..13}_*.py` → **456 passed** in 76.16 s
(Stage 3–10: 324 · Stage 11: 47 · Stage 12: 37 · Stage 13: 48) — no regressions.

---

## 17. Frozen artifact integrity

`py scripts/stage3_verify_frozen_artifacts.py` → **exit 0**,
`VERDICT: ALL FROZEN ARTIFACTS UNCHANGED`.

The Stage 14 evidence script additionally hashes five artifacts before and after
its own run (including the **GNN checkpoint** and the **graph edges**):
`frozen_artifacts_unchanged = True`.

| Artifact | Result |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | unchanged |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | unchanged |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | unchanged |
| `models/gcn_lstm_gated_v1/best_model.pt` | **unchanged** (no retraining, no rewrite) |
| `data/processed/graph/graph_D_correlation_v1_2/edges.csv` | **unchanged** |

---

## 18. Phase 15.3 reproduction

`py scripts/stage3_phase15_3_reproduction.py` → **exit 0**:

* protected artifacts re-hashed: **6/6 UNCHANGED**;
* re-derived outputs vs the frozen run: **3/3 IDENTICAL**
  (`validation_metrics.csv`, `daily_simulation_baseline.csv`,
  `daily_simulation_mpc.csv`);
* V3 integrity: **PASSED**.

`VERDICT: protected artifacts untouched; reproduction complete.`

The validated Phase 15.3 baseline is untouched, as required.

---

## 19. Performance impact

Measured on CPU by the evidence script:

| Measurement | Value |
|---|---|
| Checkpoint load (once per process) | 32.0 ms |
| `predict_all` (16 nodes × 3 targets) | 262.2 ms on the first call — **includes torch warm-up** |
| `node_representations` (fused 16×64) | 8.8 ms |
| `_run_ml_pipeline` with the advisory enabled | 128.2 ms |
| `_run_ml_pipeline` with the advisory disabled | 70.3 ms |
| **Advisory overhead per state read** | **≈ 58 ms** |
| Model parameters (resident memory driver) | 45,636 (≈ 0.18 MB fp32) |

**Does it affect MPC timing? No.** The advisory is built before the controller
runs and the controller never reads it: in the A/B, `decide()` produced identical
decisions regardless of the advisory. The measured `decide()` wall times
differed across runs (2842 / 2052 / 1963 ms) but that spread is host variance
and ordering (the first run is the cold one) — it is *not* attributable to the
advisory, which is why inertness is asserted on the decision **outputs**
(identical) rather than on timing.

Honest note: ≈ 58 ms per state read is **not free**. It is paid on every
`/api/state` read and every broadcast, on top of a ~660 ms control cycle. It was
not optimised: the brief says not to optimise prematurely, and the two-call
design (predict + represent) was chosen over refactoring `predict_all` to keep
the prediction path bit-identical.

---

## 20. Remaining limitations

1. **Most of the exposed advisory is padding in the live cascade.** The live
   simulation maps only four reservoirs (Anayirankal, Ponmudi, Idamalayar,
   Idukki); the other **12 of 16 nodes are zero-padded**, so their embeddings are
   not informed by live measurements. This is disclosed in the payload
   (`nodes_zero_padded`, `matrix_note`) and excluded from the relationship
   ranking — but it means the live advisory's spatial content is genuinely thin.
   Presenting it as a rich spatial analysis would be a fabrication.
2. **The GNN remains a documented negative result.** Its R² is lower than the
   temporal LSTM V3 baseline at every horizon and its learned gate is ≈ 0.015.
   Stage 14 exposes it *because* the brief asks for an honest advisory
   integration, not because it improves anything. Nothing in the report claims
   otherwise.
3. **The similarity matrix is exposed and can be misread.** It is labelled, its
   padding caveat is stated in-band, and the ranking is restricted to
   live-informed pairs — but a reader who takes the 16×16 matrix out of context
   can still draw a spurious "these two reservoirs are connected" conclusion.
4. **Two unused adapter affordances remain in the codebase.**
   `get_combined_forecast_dict()` and `get_risk_envelope_forecasts()` are
   implemented and tested but wired to nothing. They are documented as offline
   research affordances, and the fail-closed guard covers the control path — but
   they are reachable by a future caller that does not read the docstring.
5. **The embedding accessor duplicates the forward pass arithmetic.** It is
   pinned by a **max-abs-diff == 0.0** parity test, so drift would be caught
   immediately; it is a deliberate trade of a little duplication against
   touching the prediction path.
6. **Test isolation required an explicit fix.** Stage 14's tests drive the ONE
   authoritative singleton (stepping it to populate the advisory), which left the
   network saturated and spilling — where storages are capped and a further step
   cannot change them. That broke `test_stage4_…::test_step_command_advances_authoritative_physics`
   in the full-suite run. An autouse finalizer now restores a clean paused 50 %
   state after each Stage 14 test. This was **test state pollution, not a code
   defect** — but it is recorded here because the full suite caught it and the
   component is shared, so any future suite that drives the singleton should do
   the same.
7. **`decide()` timing spread.** The A/B records different `decide()` wall times
   across runs. This is host variance; the *outputs* are identical. Timing was
   never part of the inertness criterion and should not be read as one.
8. **Pre-existing repository junk remains untouched** (`check_index.py`,
   `debug_browser.py`, `patch*.py`, `test.js`, `tmp_*.txt` at the repository
   root). Not on any runtime path, not created by this stage.

---

## 21. Hardware status

**No hardware. Nothing faked.** No PLC, Modbus, OPC-UA or MQTT was added, and no
real-world dam control is claimed.

`hardware_status` still reports ESP32 / level sensor / flow sensor / gate
actuator as `NOT_CONNECTED`, and `gnn_advisory` carries
`"hardware_connected": false` through the evidence. The GNN remains software-only
advisory analysis.

Stage 14 introduced nothing that would obstruct future hardware work: the
advisory is a display-only payload block, so a hardware backend can be added
behind the same REST surface without touching it.

---

## 22. Exact git status

```text
 M results/phase15_stage12_authoritative_twin_state/PHASE_15_STAGE12_REPORT.md
 M results/phase15_stage3_reproduction/REPRODUCTION_CHECK.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/PHASE_15_3_V3_VALIDATION_REPORT.md
 M results/phase15_stage3_reproduction/phase15_3_reproduction/provenance_audit.json
 M results/phase15_stage3_reproduction/phase15_3_reproduction/v3_integrity_check.json
 M results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json
 M scripts/run_stage12_state_authority_audit.py
 M src/dashboard/api/state_manager.py
 M src/dashboard/app.py
 M src/dashboard/twin_component/state_adapter.py
 M src/dashboard/web/index.html
 M src/modeling/gnn_inference.py
 M src/network_env/gnn_forecast_adapter.py
?? results/phase15_stage12_authoritative_twin_state/stage12_state_authority_evidence.json
?? results/phase15_stage13_streamlit_command_proxy/
?? results/phase15_stage14_gnn_advisory/
?? scripts/run_stage13_streamlit_runtime_audit.py
?? scripts/run_stage14_gnn_advisory_audit.py
?? src/modeling/gnn_advisory.py
?? tests/test_stage13_streamlit_command_proxy.py
?? tests/test_stage14_gnn_advisory.py
```

### Stage 14 changes only

| File | Change |
|---|---|
| `src/modeling/gnn_advisory.py` | **new** — advisory builder, schema, cosine similarity, provenance, physical-vs-statistical distinction |
| `src/modeling/gnn_inference.py` | `EMBEDDING_DIM`; `_build_input_tensor` extracted verbatim; `node_representations()`; `graph_provenance()`; honest ADVISORY docstrings |
| `src/dashboard/api/state_manager.py` | build + publish `gnn_advisory`; fail-closed `_assert_control_forecasts_are_lstm_only`; pass the real gate/latency into the adapter |
| `src/dashboard/twin_component/state_adapter.py` | `_gnn_advisory_block` + the block in the twin payload schema |
| `src/dashboard/web/index.html` | corrected GNN claims; payload-driven advisory HUD rows |
| `src/dashboard/app.py` | read-only Streamlit advisory panel |
| `src/network_env/gnn_forecast_adapter.py` | docstring scope correction only (no behaviour change) |
| `tests/test_stage14_gnn_advisory.py`, `scripts/run_stage14_gnn_advisory_audit.py`, `results/phase15_stage14_gnn_advisory/` | tests, evidence, report |

Earlier entries (`stage12_*`, `stage13_*`, the Stage 3 result timestamps) are the
previously completed Stage 12 and Stage 13 finalisations; the Stage 3 JSONs
changed **timestamps only** because those verification scripts were re-run as
regression checks.

**File hygiene:** no `tmp_*` or scratch file created during Stage 14 remains; the
evidence script writes into `results/`, and the throwaway probe used while
validating embedding parity was deleted. The three `tmp_*.txt` files listed in
the repository root are **pre-existing** and untouched.

Not modified: `reservoir_network.py`, `mass_balance.py`, `mpc_controller.py`,
`safety.py`, `downstream_capacity_guard.py`, `live_mpc_orchestrator.py`,
`objective.py`, `sim_bridge.py`, the frozen LSTM V3 artifacts and the GNN
checkpoint, graph artifacts, and the Phase 15.3 protected directory.

---

## 23. Scope statement

# STAGE 15 HAS NOT BEEN STARTED.

* No GNN redesign and no retraining — 45,636 parameters, the same checkpoint,
  the same graph, verified byte-identical before and after.
* No LSTM V3 replacement; the frozen model remains the validated source.
* No MPC, SafetyLayer, DownstreamCapacityGuard, ReservoirNetwork or
  MassBalanceMonitor change.
* No RL / MARL.
* No hardware, PLC/Modbus/OPC-UA/MQTT or telemetry.
* No second simulation runtime.
* The GNN does not control gates, and no causal or hydraulic-connectivity claim
  is made anywhere.
* No frozen artifact was modified.
