# STAGE 3 — PHYSICS PARITY REPORT

**Generated:** 2026-09-14T18:23:36.409047
**Script:** `scripts/stage3_physics_parity_report.py`

Compares the legacy Phase 14.4 `VirtualCascade` against the validated
`ReservoirNetwork` under identical initial states, inflows, gate commands
and timestep (1 day).

| Engine key | Implementation | Delays | Attenuation |
|---|---|---|---|
| `legacy` | `src.simulator.environment.VirtualCascade` | 1/1/1 (live JSON) | none (implicit 1.0) |
| `network_legacy_equivalent` | `ReservoirNetwork` | 1/1/1 | 1.0 |
| `authoritative` | `LiveCascadeAdapter` → `ReservoirNetwork` | 2/1/1 | 0.90/0.85/0.80 |

---

## 1. Scenario results

All deltas are maximum absolute differences over the whole trace.

### S1_routing_only_no_spill

Moderate inflows, no reservoir reaches capacity. Isolates the ROUTING implementation itself: with legacy-equivalent parameters (1/1/1, attenuation 1.0) the two engines must agree exactly.

* Initial storages: `{'Virtual Reservoir A': 5.0, 'Virtual Reservoir B': 10.0, 'Virtual Reservoir C': 170.0, 'Virtual Reservoir D': 200.0}`
* Inflows (MCM/day): `{'Virtual Reservoir A': 1.0, 'Virtual Reservoir B': 2.0, 'Virtual Reservoir C': 20.0, 'Virtual Reservoir D': 0.0}`
* Gate commands (percent): `{'Virtual Reservoir A': 40.0, 'Virtual Reservoir B': 35.0, 'Virtual Reservoir C': 50.0, 'Virtual Reservoir D': 100.0}`
* Timesteps: 10

| Comparison | storage | release_total | inflow_routed | downstream_flow | identical |
|---|---|---|---|---|---|
| `legacy__vs__network_legacy_equivalent` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | YES |
| `network_legacy_equivalent__vs__authoritative` | 2.800e+00 | 1.500e+01 | 1.500e+01 | 1.500e+01 | NO |
| `legacy__vs__authoritative` | 2.800e+00 | 1.500e+01 | 1.500e+01 | 1.500e+01 | NO |

### S2_upstream_spill

Reservoir A overflows (storage reaches capacity with the gate mostly closed). Isolates the SPILL semantics: the legacy cascade routes spill downstream; the network treats non-terminal spill as leaving the network.

* Initial storages: `{'Virtual Reservoir A': 10.0, 'Virtual Reservoir B': 5.0, 'Virtual Reservoir C': 50.0, 'Virtual Reservoir D': 100.0}`
* Inflows (MCM/day): `{'Virtual Reservoir A': 6.0, 'Virtual Reservoir B': 1.0, 'Virtual Reservoir C': 10.0, 'Virtual Reservoir D': 0.0}`
* Gate commands (percent): `{'Virtual Reservoir A': 10.0, 'Virtual Reservoir B': 20.0, 'Virtual Reservoir C': 40.0, 'Virtual Reservoir D': 100.0}`
* Timesteps: 8

| Comparison | storage | release_total | inflow_routed | downstream_flow | identical |
|---|---|---|---|---|---|
| `legacy__vs__network_legacy_equivalent` | 2.076e+01 | 5.000e+00 | 5.500e+00 | 5.000e+00 | NO |
| `network_legacy_equivalent__vs__authoritative` | 7.500e-01 | 1.200e+01 | 1.200e+01 | 1.200e+01 | NO |
| `legacy__vs__authoritative` | 2.126e+01 | 1.200e+01 | 1.200e+01 | 1.200e+01 | NO |

### S3_authoritative_full

Mixed storm scenario exercising delay (2/1/1) and attenuation (0.90/0.85/0.80) together with a terminal flow near the downstream capacity limit.

* Initial storages: `{'Virtual Reservoir A': 8.0, 'Virtual Reservoir B': 15.0, 'Virtual Reservoir C': 120.0, 'Virtual Reservoir D': 150.0}`
* Inflows (MCM/day): `{'Virtual Reservoir A': 5.0, 'Virtual Reservoir B': 8.0, 'Virtual Reservoir C': 60.0, 'Virtual Reservoir D': 0.0}`
* Gate commands (percent): `{'Virtual Reservoir A': 60.0, 'Virtual Reservoir B': 50.0, 'Virtual Reservoir C': 70.0, 'Virtual Reservoir D': 100.0}`
* Timesteps: 12

| Comparison | storage | release_total | inflow_routed | downstream_flow | identical |
|---|---|---|---|---|---|
| `legacy__vs__network_legacy_equivalent` | 0.000e+00 | 8.000e+00 | 8.000e+00 | 8.000e+00 | NO |
| `network_legacy_equivalent__vs__authoritative` | 7.500e-01 | 2.120e+01 | 2.120e+01 | 2.120e+01 | NO |
| `legacy__vs__authoritative` | 7.500e-01 | 2.394e+01 | 2.394e+01 | 2.394e+01 | NO |

---

## 2. Attribution

| Check | Result |
|---|---|
| `s1_legacy_backed_network_matches_virtualcascade` | PASS |
| `s2_spill_semantics_diverge` | PASS |
| `s3_authoritative_differs_from_legacy` | PASS |

Interpretation:

* `legacy` vs `network_legacy_equivalent` agrees **exactly** in the
  no-spill scenario — the two implementations are numerically the same
  model whenever their assumptions coincide.
* Divergence in the spill scenario is fully attributable to the
  **spill-routing** semantic difference.
* Divergence of `authoritative` is attributable to the validated
  **routing delay (2/1/1)** and **attenuation (0.90/0.85/0.80)** targets.

---

## 3. Semantic difference register

| Dimension | Legacy `VirtualCascade` | Authoritative `ReservoirNetwork` | Impact | Resolution |
|---|---|---|---|---|
| **topology** | Hardcoded 4 reservoirs; names baked into Python source. | Topology-agnostic DAG read from topology_config.yaml (A->B->C->D). | None on the live 4-reservoir cascade; enables N-reservoir extension without code changes. | Node ids are the live virtual names, mapped to canonical ids via display_name / repository_reference. |
| **routing_delays** | 1 day on A->B, B->C, C->D (from four_reservoir_demo.json). | A->B = 2, B->C = 1, C->D = 1 (validated topology_config.yaml). | A->B travel time doubles; upstream flood waves arrive at B one day later. | Adopt the validated 2/1/1 target. Live JSON delays are superseded (documented). |
| **attenuation** | None (implicit factor 1.0). No transmission loss ever occurs. | A->B = 0.90, B->C = 0.85, C->D = 0.80; loss booked as total_routing_loss. | 10%/15%/20% of routed volume is lost in transit; downstream routed inflows and terminal flow decrease. | Adopt the validated attenuation. Losses are explicit and mass-balanced, not silently destroyed. |
| **storage** | storage_mcm, capacity_mcm in MCM. Same units as the network. | storage / capacity in MCM. Identical semantics. | None. | No change. Adapter exposes storage_mcm / capacity_mcm verbatim. |
| **release** | release_mcm_day = requested release PLUS forced spill (lumped). | controlled_release and spill are separate fields; total_outflow = controlled_release + spill. | Legacy callers cannot distinguish a controlled release from an overflow. | Adapter keeps release_mcm_day = total_outflow (legacy meaning preserved) and additionally exposes controlled_release_mcm_day and spill_mcm. |
| **spill** | Overflow added to release AND routed downstream to the next reservoir. | Non-terminal spill leaves the network entirely (not routed); terminal spill is part of terminal outflow. | In the legacy cascade, spill at A artificially inflates B's inflow. The network conserves it as a separate outflow term. | Accepted intentional difference; quantified in scenario S2 and covered by tests. |
| **downstream_capacity** | Compared against res_D.release_mcm_day (controlled + spill). | Compared against terminal node total_outflow (controlled + spill). | None. Both compare the same quantity at the terminal node (50.0 MCM/day). | No change; current_downstream_flow = network.terminal_outflow. |
| **mass_balance** | No global accounting. Water in routing queues and routing losses are not tracked. | Closed global balance: inflow = storage change + terminal outflow + non-terminal spill + routing loss + water in transit. | The live twin gains an auditable conservation law it did not have before. | Exposed via LiveCascadeAdapter.mass_balance_check(); verified < 1e-10 in tests. |
| **units** | Gate commands in PERCENT (0-100); storage MCM; flow MCM/day. | Gates internally FRACTION (0.0-1.0); storage MCM; flow MCM/day. | A fraction/percent mix-up would under-release by ~100x. | Single conversion boundary in src/common/units.py; the adapter converts percent->fraction exactly once. Non-finite gates raise instead of failing open. |
| **missing_inputs** | KeyError if an inflow or gate entry is absent. | Absent entries default to 0.0 (ReservoirNetwork.step semantics). | Strictly safer; a partially-specified command no longer aborts the live loop. | Adapter follows ReservoirNetwork; documented as an intentional robustness change. |
| **gate_clamping** | Finite out-of-range gate clamped into [0, 100]. | Finite out-of-range fraction clamped into [0.0, 1.0]; non-finite raises UnitContractError. | Identical for finite input (both clamp). Non-finite input raises in both implementations. | No behavioural change; both already route through src/common/units.py. |
| **provenance** | None. | Every physics parameter carries OBSERVED / VERIFIED / ASSUMED_FOR_PROTOTYPE. | The live twin can now distinguish observed data from prototype assumptions. | Exposed via LiveCascadeAdapter.physics_provenance(). |

---

## 4. Method notes

* The gate boundary is shared: the harness passes the SAME percent values
  to every engine and converts percent -> fraction once via
  `src/common/units.py`.
* `VirtualCascade` and `ReservoirNetwork` are used **unmodified**.
* No frozen LSTM artifact is read or written by this harness.
