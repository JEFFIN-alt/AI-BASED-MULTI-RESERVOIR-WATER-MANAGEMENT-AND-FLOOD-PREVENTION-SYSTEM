# STAGE 6 REPORT — FORECAST ADAPTER

**Date:** 2026-09-14
**Status:** COMPLETE — stopped at the Stage 6 boundary (Stage 7 NOT started)
**Goal:** Create a clean boundary between the live forecast representation and the validated MPC representation.

---

## 1. Architecture Before / After

### Before Stage 6

```
Frozen LSTM V3
      ↓
live forecast payload            ← Stage 5 output: {virtual_name: {...targets + provenance}}
      ↓
   (nothing)                     ← NO boundary existed. The live representation
      ✗                             and the MPC contract were unconnected, and
      ✗                             there was no path that preserved provenance.
      ✗
NetworkForecastSnapshot          ← Phase 15.2 contract, only ever produced by
      ↓                             V3ForecastAdapter (frozen CSV) or the GNN adapter
 (future Stage 7) MPC
```

The two representations were disjoint:

| | Live representation (Stage 5) | MPC representation (Phase 15.2) |
|---|---|---|
| Key | `"Virtual Reservoir A"` (live name) | `"Reservoir_A"` (network node id) |
| Shape | plain `dict` | `ReservoirForecast` dataclass |
| Horizons | keys `forecast_1d/3d/7d` | fields `target_1d/3d/7d` |
| Provenance | rich structured dict | one free-text `provenance` slot |
| Status | `forecast_status` string | `ForecastStatus` enum per horizon |

### After Stage 6

```
Frozen LSTM V3  (UNMODIFIED)
      ↓
live forecast payload            ← Stage 5, source-agnostic
      ↓
LiveForecastAdapter              ← **NEW** — the single boundary
      ↓
NetworkForecastSnapshot          ← EXACT Phase 15.2 contract, unmodified
      ↓
 (future Stage 7) MPC            ← NOT integrated (Req. 15)
```

The adapter is **additive**: nothing on the validated side was touched, and
nothing on the validated side imports the adapter.

---

## 2. Adapter Interface

`src/network_env/live_forecast_adapter.py`

```python
class LiveForecastAdapter:
    def __init__(self, project_root=None, topology_path=None,
                 reservoir_mapping=None): ...

    # primary outputs
    def build_bundle(self, live_forecasts: Dict[str, dict],
                     forecast_date: str) -> LiveForecastBundle
    def build_snapshot(self, live_forecasts: Dict[str, dict],
                       forecast_date: str) -> NetworkForecastSnapshot   # exact MPC input

    # introspection
    @property reservoir_mapping  -> {"Virtual Reservoir A": "Anayirankal", ...}
    @property node_mapping       -> {"Virtual Reservoir A": "Reservoir_A", ...}
    @property network_node_ids   -> ["Reservoir_A", ..., "Reservoir_D"]
    def contract_info(self) -> dict

@dataclass
class LiveForecastBundle:
    snapshot: NetworkForecastSnapshot       # <- pass this to the MPC in Stage 7
    provenance: Dict[str, Dict]             # structured provenance per node id
    forecast_date: str
    status: str
    notes: List[str]
    # convenience passthroughs: .forecasts, .get(nid), .get_provenance(nid),
    #                           .provenance_summary()
```

`build_snapshot()` returns **only** the `NetworkForecastSnapshot`, satisfying
Req. 6 literally. `build_bundle()` additionally carries the full provenance that
the narrow Phase 15.2 contract cannot express.

**Node keys are `Reservoir_A … Reservoir_D`**, matching
`ReservoirNetwork.processing_order`, which is what the MPC looks up — pinned by
`test_snapshot_node_keys_match_the_reservoir_network_processing_order`.

---

## 3. Forecast Mapping (Req. 5, 6)

| Live key | Network node id | V3 reservoir name | Status |
|---|---|---|---|
| `Virtual Reservoir A` | `Reservoir_A` | **Anayirankal** | ✅ |
| `Virtual Reservoir B` | `Reservoir_B` | **Ponmudi** | ✅ |
| `Virtual Reservoir C` | `Reservoir_C` | **Idamalayar** | ✅ |
| `Virtual Reservoir D` | `Reservoir_D` | **Idukki** | ✅ |

The mapping is resolved from `topology_config.yaml`
(`display_name` → `id`, `repository_reference` → V3 name) and then **verified
against a hard-coded expectation** (`REQUIRED_RESERVOIR_MAPPING`). A contradicting
override raises `ValueError` rather than silently re-mapping.

Horizon mapping:

| Live payload key | Snapshot field | Status field |
|---|---|---|
| `forecast_1d` | `target_1d` | `status_1d` |
| `forecast_3d` | `target_3d` | `status_3d` |
| `forecast_7d` | `target_7d` | `status_7d` |

Units are asserted as `MCM/day` on both sides; a payload declaring any other unit
is rejected as `INVALID` with `issue = UNIT_MISMATCH`.

---

## 4. Provenance Handling (Req. 8, 11)

The Phase 15.2 contract has exactly one free-text provenance slot
(`ReservoirForecast.provenance`). Because that contract must not be modified, the
adapter:

1. **fills that slot** with a deterministic, parseable encoding:

   ```
   status=DEMONSTRATION_ONLY;source=SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL;
   is_simulated=True;validated_metrics_apply=False;unit=MCM/day;
   synthetic=water_level+rainfall;unavailable=none
   ```

2. **keeps the full structured record** in `LiveForecastBundle.provenance[node_id]`:
   declared status/provenance, `is_simulated`, `validated_metrics_apply`, the
   synthetic / unavailable / simulated feature lists, the raw values, the
   per-horizon statuses and the issue code.

3. **sets `model = "LSTM_V3_LOGTARGET"`** so the snapshot identifies the frozen
   model version.

A `VALIDATED` / `REAL_MEASUREMENT_INPUTS_FROZEN_MODEL` payload (a future hardware
path) is preserved verbatim — `test_validated_provenance_is_preserved_verbatim`.

---

## 5. Error Handling (Req. 9, 10, 11)

Every case produces an explicit, machine-readable `issue` code. **No value is
ever fabricated, defaulted, clamped, interpolated or carried forward.**

| Case | `issue` | Horizon statuses | Values |
|---|---|---|---|
| Reservoir absent from payload | `MISSING_FORECAST` | `FORECAST_UNAVAILABLE` | `None` |
| Payload not a dict | `MALFORMED_PAYLOAD` | `FORECAST_INVALID` | `None` |
| `forecast_status = WARMUP_INSUFFICIENT_HISTORY` | `WARMUP_INSUFFICIENT_HISTORY` | `FORECAST_UNAVAILABLE` | `None` |
| `forecast_status = FORECAST_UNAVAILABLE` / `unavailable_features` non-empty | `REQUIRED_FEATURE_UNAVAILABLE` | `FORECAST_UNAVAILABLE` | `None` |
| Declared unit ≠ `MCM/day` | `UNIT_MISMATCH` | `FORECAST_INVALID` | `None` |
| Declared horizons ≠ `[forecast_1d, forecast_3d, forecast_7d]` | `HORIZON_MISMATCH` | `FORECAST_INVALID` | `None` |
| `NaN` / `±Inf` | `INVALID_VALUE`, `value_issues["1d"]="non_finite"` | `FORECAST_INVALID` | `None` |
| Non-numeric (str/list/dict/bool) | `INVALID_VALUE`, `non_numeric:*` | `FORECAST_INVALID` | `None` |
| Horizon key absent | `value_missing` | `FORECAST_UNAVAILABLE` | `None` |
| Negative inflow | `negative_inflow:<value>` | `FORECAST_INVALID` | `None` (never clamped to 0) |
| Unmappable payload key | recorded in `bundle.notes` | — | — |

Every mapped reservoir **always** appears in the snapshot, even with no payload
at all (`status == "NO_FORECASTS"`, `available_count("1d") == 0`).

The negative-inflow guard is an **explicit additional** check (not requested):
inflow is physically non-negative, so a negative value is reported rather than
passed to the MPC. It is documented here rather than applied silently.

---

## 6. Files Changed

| File | Change |
|---|---|
| `src/network_env/live_forecast_adapter.py` | **NEW** — the Stage 6 boundary (adapter + `LiveForecastBundle`) |
| `src/network_env/__init__.py` | Export `LiveForecastAdapter`, `LiveForecastBundle` |
| `src/dashboard/web/index.html` | Demonstration provenance badge (Stage 5 decision condition 3) |
| `src/dashboard/app.py` | Streamlit demonstration banner (same condition) |
| `tests/test_stage6_forecast_adapter.py` | **NEW** — 56 tests |

**NOT modified (Req. 1–4):** `src/controller/mpc_controller.py`,
`src/controller/safety.py`, `src/controller/objective.py`,
`src/network_env/reservoir_network.py`, `src/network_env/v3_forecast_adapter.py`,
`best_model.pt`, `log_target_scaler.pkl`, `feature_scaler.pkl`, the frozen V3
prediction CSVs, and all of `results/phase15_v3_validation/`.

`src/network_env/v3_forecast_adapter.py` — which defines the snapshot contract —
was deliberately **not touched**; the adapter consumes its types as-is.

---

## 7. Tests (Req. 12, 13)

`tests/test_stage6_forecast_adapter.py` — **56 tests**:

| Group | What it proves |
|---|---|
| Reservoir mapping | all four map correctly (A/B/C/D → Anayirankal/Ponmudi/Idamalayar/Idukki); snapshot keyed by network node id; each node carries its V3 name; a contradicting mapping raises |
| Horizon mapping | all three horizons map correctly; horizons are not conflated; horizon mismatch rejected explicitly |
| Units | `MCM/day` end-to-end; a declared unit mismatch is rejected |
| Numerical fidelity | values are **bit-identical** (`==`, not `approx`) incl. `repr` equality; round-trip preserved for 5 magnitudes |
| Provenance | survives in both the contract slot and the structured bundle; model version recorded; summary exposes status; a VALIDATED payload is preserved verbatim |
| Invalid handling | NaN/±Inf → INVALID; non-numeric → INVALID/UNAVAILABLE; negative inflow INVALID and **not clamped**; a bad horizon does not corrupt its siblings |
| Unavailable handling | warm-up → UNAVAILABLE; `FORECAST_UNAVAILABLE` respected; missing reservoir explicit; malformed payload explicit; unmapped keys reported |
| No fabrication | empty payload → `NO_FORECASTS`, all `None`; an absent horizon is never defaulted; no synthesis for unknown nodes |
| MPC contract | output **is** a `NetworkForecastSnapshot`; every MPC accessor works; node keys match `ReservoirNetwork.processing_order`; adapter is **not** wired into the MPC; validated modules unmodified; nothing validated imports the adapter |
| End-to-end | live Stage 5 pipeline → adapter → MPC contract (all four nodes present, D explicit UNAVAILABLE, values finite) |
| Stage 5 decision conditions | badge present and driven by backend provenance; Streamlit banner; `live_forecasts_are_validated is False`; strict mode retained (source + attribute + **isolated subprocess** env-var test); placeholders never labelled as measurements; **no storage→level conversion**; hardware-readiness interface is source-agnostic |
| GNN separation | the adapter does not import or reference the GNN |

---

## 8. Test Counts (Req. 18)

| Suite | Result |
|---|---|
| **Stage 6 tests** | ✅ **56 passed** |
| **Complete test suite** | ✅ **348 passed, 0 failed** (41.2 s) |
| Stage 3 regression | ✅ 28/28 |
| Stage 4 regression | ✅ 41/41 |
| Stage 5 regression | ✅ 38/38 |
| Stage 3+4+5+6 combined | ✅ **163 passed, 0 failed** |

Suite growth: 292 (end of Stage 5) → **348** (Stage 6, +56).

---

## 9. Frozen Artifact Verification (Req. 18)

`scripts/stage3_verify_frozen_artifacts.py` → **exit 0 — ALL FROZEN ARTIFACTS UNCHANGED**

| Artifact | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/best_model.pt` | ✅ UNCHANGED (raw match) |
| `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | ✅ UNCHANGED (raw match) |
| `data/processed/scaled/feature_scaler.pkl` | ✅ read-only, not modified |
| `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | ✅ UNCHANGED (LF-normalised match; raw differs only by `core.autocrlf` CRLF) |

---

## 10. Phase 15.3 Impact (Req. 19)

✅ All 6 files in `results/phase15_v3_validation/` are **byte-for-byte unchanged**
(compared against the Stage 3 baseline hashes). The Stage 6 adapter is not
imported by, and does not touch, any Phase 15.3 path. No Phase 15.3 script was
executed.

---

## 11. Hardware-Readiness Implications (Req. 17, decision condition 10–11)

The adapter was designed so the **forecast source can be replaced without
touching the frozen model or the downstream MPC contract**.

| Future requirement | How the interface already supports it |
|---|---|
| real `water_level` in metres | the pipeline simply stops using the `SYNTHETIC_DEMO` placeholder for that feature; provenance becomes `MEASURED_HISTORICAL` |
| real `rainfall` in mm | same |
| real `inflow`, `live_storage`, `total_outflow` | already flow from the authoritative network; would come from telemetry instead |
| timestamp / provenance | `forecast_date` is carried per forecast; the bundle keeps the full provenance record |
| no MPC change | the adapter emits the same `NetworkForecastSnapshot`; the MPC reads `is_available()` / `target_*` exactly as today |
| no frozen-model change | the model is only ever *called*, never modified; the input contract is verified at load |

`test_adapter_accepts_real_telemetry_style_payloads_without_contract_change`
proves a `VALIDATED` / all-`MEASURED_HISTORICAL` payload flows through the *same*
adapter into the *same* snapshot type.

**Not implemented (out of scope):** no telemetry ingestion, no hardware client,
no `water_level`/`rainfall` acquisition. Those remain Stage 6+ data-acquisition /
hardware-integration work.

---

## 12. Remaining Risks

| # | Risk | Severity | Notes |
|---|---|---|---|
| 1 | The Phase 15.2 provenance slot is a single free-text field, so the MPC itself only sees the compact string, not the structured record | Low–Med | Structured provenance lives in `LiveForecastBundle`; a future stage could widen the dataclass (additive, defaulted fields) if the MPC needs to branch on provenance. Deliberately not done now to keep the validated contract frozen. |
| 2 | The negative-inflow guard is an addition not requested by the spec | Low | Reported as `INVALID`, never clamped; documented in §5. |
| 3 | `Reservoir_D` (Idukki) never receives a live forecast (the Stage 5 pipeline skips it), so it is always `UNAVAILABLE` in the snapshot | Low | Explicit and correct; the MPC already degrades to "hold current" when a horizon is unavailable. |
| 4 | Live forecasts are `DEMONSTRATION_ONLY`; if Stage 7 wired them to the MPC, the controller would act on demonstration inputs | **Medium** | This is *why* Stage 6 stops here. Stage 7 must gate on `validated_metrics_apply` / provenance before letting live forecasts drive control. |
| 5 | The adapter raises on a contradictory reservoir mapping | Low | Fail-closed by design; the requirement-5 mapping is asserted. |
| 6 | Hardware path remains unbuilt | Medium | Interface is ready; ingestion is not. |

---

## 13. Requirement Compliance

| # | Requirement | Status |
|---|---|---|
| 1 | Do NOT modify MPC | ✅ untouched; not imported by the adapter |
| 2 | Do NOT modify SafetyLayer | ✅ untouched |
| 3 | Do NOT modify ReservoirNetwork | ✅ untouched |
| 4 | Do NOT modify frozen LSTM artifacts | ✅ verified by hash |
| 5 | Map A/B/C/D → Anayirankal/Ponmudi/Idamalayar/Idukki | ✅ asserted in code and tests |
| 6 | Map `target_1d/3d/7d` into the exact `NetworkForecastSnapshot` | ✅ `build_snapshot()` returns the Phase 15.2 type unmodified |
| 7 | Preserve MCM/day | ✅ unit asserted both ways; mismatch rejected |
| 8 | Preserve provenance | ✅ contract slot + structured bundle |
| 9 | Handle missing / unavailable / warm-up / NaN / Inf / horizon mismatch / missing reservoir | ✅ 7 explicit issue codes |
| 10 | No silent fabricated values | ✅ 0 defaults, 0 clamps, 0 carry-forward |
| 11 | Unavailable represented explicitly | ✅ per-horizon statuses + reasons |
| 12 | Values unaltered | ✅ bit-identical, `repr`-equal |
| 13 | Tests for all listed behaviours | ✅ 56 tests |
| 14 | GNN advisory-only | ✅ not imported, not referenced |
| 15 | Do NOT integrate MPC | ✅ not wired; test asserts absence |
| 16 | Do NOT integrate SafetyLayer | ✅ untouched |
| 17 | Design for a future hardware path | ✅ source-agnostic interface + validated-payload test |
| 18 | Run Stage 6 / full / Stage 3 / 4 / 5 / frozen verification | ✅ §8, §9 |
| 19 | Do not overwrite Phase 15.3 files | ✅ §10 |
| 20 | Do NOT proceed to Stage 7 | ✅ not started |

---

## 14. Recommendation

1. **Accept Stage 6.** The boundary exists, is additive, carries provenance, alters
   no value, and fabricates nothing.
2. **Do not wire the live forecast into the MPC as-is.** Live forecasts are
   `DEMONSTRATION_ONLY`; Stage 7 must explicitly gate on provenance
   (`validated_metrics_apply`) before demonstration forecasts are allowed to
   influence control. Consider requiring `FORECAST_EXPERIMENTAL` mode or running
   MPC only on the offline validated pipeline until a real telemetry source lands.
3. **Stage 7 candidates**
   * integrate the MPC against `LiveForecastAdapter` output **behind a provenance
     gate**;
   * decide whether `ReservoirForecast` should gain defaulted provenance fields
     (additive) so the MPC can branch on provenance directly;
   * resolve `Reservoir_D`'s absent forecast (either forecast it live or accept the
     explicit `UNAVAILABLE`).
4. **Do NOT** integrate the SafetyLayer (Stage 8) — untouched.

---

*End of Stage 6 Report. Stopped at the Stage 6 boundary — Stage 7 not started.*
