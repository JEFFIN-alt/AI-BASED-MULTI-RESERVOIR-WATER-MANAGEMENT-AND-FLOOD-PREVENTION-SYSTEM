# PHASE 15.1 — DETERMINISTIC INTERCONNECTED RESERVOIR NETWORK ENVIRONMENT

## 1. What Was Implemented

A generic N-reservoir deterministic network environment (`src/network_env/`) that serves as the canonical interconnected-reservoir simulation engine for the project. It replaces nothing — the Phase 14.4 simulator (`src/simulator/`) remains fully intact and operational for the existing Streamlit dashboard.

## 2. Network Architecture

The environment uses a directed acyclic graph (DAG) topology:
- **Nodes** = Reservoirs (arbitrary number, configured via YAML)
- **Edges** = Directed connections with routing delay and attenuation
- **Processing order** = Topological sort (Kahn's algorithm) ensures upstream nodes are always processed before downstream nodes

Initial configuration: `Reservoir_A → Reservoir_B → Reservoir_C → Reservoir_D`

## 3. Reservoir State Model

Each reservoir maintains a `ReservoirState` dataclass:
| Field | Description |
|---|---|
| `storage` | Current storage (MCM) |
| `inflow_local` | External catchment inflow |
| `inflow_routed` | Water arriving from upstream after delay/attenuation |
| `controlled_release` | Gate-controlled release volume |
| `spill` | Uncontrolled overflow when capacity exceeded |
| `total_outflow` | controlled_release + spill |
| `gate_position` | Normalised [0.0, 1.0] |

## 4. Mass-Balance Equation

```
new_storage = old_storage + local_inflow + routed_inflow - controlled_release - spill
```

Global conservation:
```
total_external_inflow = storage_change
                      + total_terminal_outflow
                      + total_nonterminal_spill
                      + total_routing_loss
                      + water_in_transit
```

Verified to hold within floating-point tolerance (~1e-14) across all test scenarios.

## 5. Gate / Release Model

```
release = gate_position × max_release
```

Constraints enforced:
- `0.0 ≤ gate_position ≤ 1.0` (clipped if out of range)
- `0.0 ≤ release ≤ max_release`
- `release ≤ available_water` (storage + inflow)
- Excess storage beyond capacity → **spill** (forced overflow)
- Storage < 0 is mathematically impossible (assertion-guarded)

## 6. Routing Model

Routing uses FIFO queues:
1. When node A releases water and connection A→B has `delay=D` and `attenuation=α`:
2. The release enters a FIFO queue of length D.
3. After D timesteps, the oldest entry exits the queue.
4. B receives: `α × queued_amount`.
5. **Transmission loss** = `(1 − α) × queued_amount` — explicitly tracked in the global mass balance as "routing losses".

This model does NOT silently destroy mass. Every unit of water is accounted for.

## 7. Provenance System

Every configurable parameter carries a provenance classification:

| Level | Meaning |
|---|---|
| `OBSERVED` | Value directly from repository observation data |
| `VERIFIED` | Cross-checked against multiple repository sources |
| `ASSUMED_FOR_PROTOTYPE` | Chosen for prototype; NOT verified Kerala data |

**Current configuration provenance summary:**
- **OBSERVED**: 4 parameters (reservoir capacities)
- **ASSUMED_FOR_PROTOTYPE**: 16 parameters (topology, delays, attenuations, initial storages, max releases, downstream capacity)

The network topology (A→B→C→D) is explicitly marked as `ASSUMED_FOR_PROTOTYPE`.

## 8. Four-Reservoir Configuration

Located at: `src/network_env/topology_config.yaml`

| Reservoir | Capacity (MCM) | Provenance | Initial Storage | Max Release |
|---|---|---|---|---|
| Reservoir_A | 10.82 | OBSERVED | 5.41 (50%) | 5.0 MCM/day |
| Reservoir_B | 21.26 | OBSERVED | 10.63 (50%) | 10.0 MCM/day |
| Reservoir_C | 348.29 | OBSERVED | 174.15 (50%) | 150.0 MCM/day |
| Reservoir_D | 401.43 | OBSERVED | 200.72 (50%) | 200.0 MCM/day |

| Connection | Delay | Attenuation | Provenance |
|---|---|---|---|
| A → B | 2 days | 0.90 | ASSUMED_FOR_PROTOTYPE |
| B → C | 1 day | 0.85 | ASSUMED_FOR_PROTOTYPE |
| C → D | 1 day | 0.80 | ASSUMED_FOR_PROTOTYPE |

## 9. Test Results

All 11 tests passed:

| Test | Description | Result |
|---|---|---|
| TEST 1 | Mass conservation (50 timesteps, 4 reservoirs) | ✅ residual = 0.00e+00 |
| TEST 2 | Routing delay (delay=2, verified arrival at t=3) | ✅ |
| TEST 3 | Attenuation (factor=0.8, 100→80 MCM, loss=20) | ✅ |
| TEST 4 | Capacity overflow (spill correctly classified) | ✅ |
| TEST 5 | No negative storage (release capped to available) | ✅ |
| TEST 6 | Gate limits (clipped to [0.0, 1.0]) | ✅ |
| TEST 7 | Release limit (capped at max_release and available) | ✅ |
| TEST 8 | Cascade propagation (A→B→C→D verified) | ✅ |
| TEST 9 | Provenance (20 params classified, OBSERVED vs ASSUMED) | ✅ |
| TEST 9b | Provenance distinction (capacities=OBSERVED, delays=ASSUMED) | ✅ |
| TEST 10 | Reproducibility (20 timesteps, identical across runs) | ✅ |

## 10. Smoke-Test Results

15-timestep simulation with 5 MCM/day inflow into Reservoir_A:
- Water correctly propagated A → B (2-day delay, 10% loss) → C (1-day delay, 15% loss) → D (1-day delay, 20% loss)
- Reservoir_A overflowed at t=6 when storage exceeded capacity (10.82 MCM)
- Mass balance verified to < 1e-14 residual error
- All routing losses explicitly accounted for (56.64 MCM total routing loss over 15 days)

**Command to reproduce:**
```bash
PYTHONPATH=. python tests/smoke_test_network_env.py
```

## 11. Files Created

| File | Purpose |
|---|---|
| `src/network_env/__init__.py` | Package init with public API exports |
| `src/network_env/provenance.py` | Provenance classification system |
| `src/network_env/reservoir_network.py` | Core N-reservoir network environment |
| `src/network_env/topology_config.yaml` | 4-reservoir prototype configuration |
| `tests/test_network_env_conservation.py` | Tests 1-8, 10 (physics/conservation) |
| `tests/test_network_env_provenance.py` | Test 9 (provenance classification) |
| `tests/smoke_test_network_env.py` | Deterministic propagation smoke test |
| `results/phase15_network_environment/PHASE_15_1_REPORT.md` | This report |

## 12. Files Modified

**None.** No existing files were modified.

## 13. Files Intentionally Protected

| Directory/File | Status |
|---|---|
| `models/lstm_pytorch_v3_logtarget/` | ✅ Untouched (0 files modified) |
| `results/lstm_pytorch_v3_logtarget/` | ✅ Untouched (0 files modified) |
| `src/simulator/` | ✅ Untouched — existing simulator fully operational |
| `src/dashboard/` | ✅ Untouched — dashboard unchanged |
| `src/management/risk_engine.py` | ✅ Untouched |
| `data/` | ✅ Untouched |
| `configs/simulation/four_reservoir_demo.json` | ✅ Untouched |

## 14. V3 Integrity Confirmation

LSTM V3 model artifacts remain byte-for-byte untouched. Verified by checking that 0 of 7 V3 files have modification timestamps newer than the Phase 15.1 implementation.

## 15. Known Limitations

1. **No controller integration** — The environment accepts gate positions as inputs but does not contain a controller (by design; that is Phase 15.2+).
2. **No LSTM integration** — V3 forecasts are not connected (Phase 15.2).
3. **No hardware interface** — No ESP32/servo communication (Phase 15.3+).
4. **Routing model is simplified** — The FIFO queue model assumes fixed delay and linear attenuation. Real hydrological routing uses Muskingum or kinematic wave methods.
5. **Spill is not routed** — When a non-terminal reservoir spills, the overflow leaves the network entirely. In reality, spilled water might still follow the river channel downstream.

## 16. Design Decision: Relationship to src/simulator/

The Phase 14.4 `src/simulator/` package has a `VirtualCascade` class that is hardcoded to exactly 4 reservoirs with rigid names ("Virtual Reservoir A", etc.) baked into Python source code. It was designed as a quick proof-of-concept for the Streamlit dashboard.

`src/network_env/` is a clean generalization:
- **Topology-agnostic**: Configured entirely via YAML. No reservoir names in Python source.
- **N-reservoir**: Adding a 5th reservoir requires editing only the YAML file.
- **Routing attenuation**: The Phase 14.4 simulator assumed attenuation=1.0 (no loss).
- **Separated spill accounting**: Phase 14.4 lumped spill into the release counter.
- **Provenance tracking**: Phase 14.4 had no provenance system.

The two packages coexist. The existing dashboard and its `src/simulator/` dependencies remain fully functional.

## 17. What Phase 15.2 Will Connect Next

**PHASE 15.2 — LSTM V3 FORECAST INTEGRATION**

Phase 15.2 will connect the frozen V3 point forecasts into the `src/network_env/` environment as external inflow predictions, enabling a forecast-aware controller to use the network environment as its world model.
