# AquaFlow

**AI-Based Multi-Reservoir Water Management and Flood Prevention System**

AquaFlow is an AI-assisted decision-support and digital-twin framework for coordinating
releases across a cascade of interconnected reservoirs. It combines a frozen deep-learning
inflow forecaster, a graph-based spatial advisory, a Model Predictive Control (MPC) release
planner, an independent safety layer, and a live 3D digital twin that visualises the
authoritative system state.

Repository: [JEFFIN-alt/AI-BASED-MULTI-RESERVOIR-WATER-MANAGEMENT-AND-FLOOD-PREVENTION-SYSTEM](https://github.com/JEFFIN-alt/AI-BASED-MULTI-RESERVOIR-WATER-MANAGEMENT-AND-FLOOD-PREVENTION-SYSTEM)

> **Status:** the software pipeline is implemented and validated through **Stage 17** on a
> four-reservoir prototype network. **Physical hardware integration has not been completed** —
> see [Hardware Integration — Stage 18](#hardware-integration--stage-18).

---

## Overview

Reservoir operators must balance two competing pressures: retaining enough water for
irrigation, municipal supply and hydropower, while releasing enough to avoid overtopping and
downstream flooding during heavy inflow. Rule curves and historical averages are a weak fit
for volatile modern hydrology, and manual coordination across a reservoir cascade is difficult
because a release decision at one reservoir becomes inflow for the next.

AquaFlow addresses this by pairing **temporal inflow forecasting** (an LSTM network) with
**spatial/dependency analysis** (a Graph Neural Network used in an advisory role), then feeding
both into an **MPC release planner** whose proposals are constrained by a **safety layer**
before becoming gate/release commands. The result is published as a single authoritative
digital-twin state that drives a FastAPI/WebSocket backend and a 3D visualisation.

The current system is a **development/validation prototype**. It demonstrates the full
software chain end-to-end on a four-reservoir network; it is not a claim of operational safety
for any real dam.

---

## System Architecture

```text
  CONTROL PATH  (authoritative)              ADVISORY PATH  (read-only)

  Historical Reservoir Data                  Reservoir Graph
            │                                       │
            ▼                                       ▼
  ┌──────────────────────────┐            ┌──────────────────────────┐
  │       Frozen LSTM V3     │            │       GNN Advisory       │
  │   Temporal Forecasting   │            │  Spatial / Dependency    │
  └────────────┬─────────────┘            │  Representation          │
               │                          └────────────┬─────────────┘
               ▼                                       │
  ┌──────────────────────────┐                         │
  │   V3 Forecast Adapter    │                         │
  └────────────┬─────────────┘                         │
               │                                       │
               ▼                                       │
  ┌──────────────────────────┐                         │
  │  Multi-Reservoir Network │    (no connection to    │
  │        / State           │     the control path)   │
  └────────────┬─────────────┘                         │
               │                                       │
               ▼                                       │
  ┌──────────────────────────┐                         │
  │           MPC            │                         │
  │    Release Planning      │                         │
  └────────────┬─────────────┘                         │
               │                                       │
               ▼                                       │
  ┌──────────────────────────┐                         │
  │      Safety Layer        │                         │
  └────────────┬─────────────┘                         │
               │                                       │
               ▼                                       ▼
  ┌──────────────────────────┐            ┌──────────────────────────┐
  │  Gate / Release Commands │            │   API / Dashboard        │
  └────────────┬─────────────┘            │   Advisory Display       │
               │                          └──────────────────────────┘
               ▼
  ┌──────────────────────────┐
  │ Authoritative Twin State │
  │          / API           │
  └────────────┬─────────────┘
               │
               ▼
         Digital Twin (3D)

  The GNN advisory branch is deliberately NOT connected to the control path:
      GNN  ──X──▶  MPC
      GNN  ──X──▶  Safety Layer
      GNN  ──X──▶  Gate / release commands
```

**Role separation**

| Component | Responsibility | Controls gates? |
|---|---|---|
| **LSTM V3** | Temporal inflow forecasting (1/3/7-day point forecasts, MCM/day) | No |
| **GNN** | Spatial / dependency representation and advisory analysis over the reservoir graph | **No** |
| **MPC** | Coordinated, look-ahead release planning across the cascade | Proposes commands |
| **Safety Layer** | Enforces safety constraints on proposed commands (with a downstream capacity guard) | Constrains commands |
| **Digital Twin** | Backend-owned authoritative state; the 3D view + dashboard are read-only clients | — |

The GNN is **advisory only**: it does not control gates, does not replace the LSTM, and is not
claimed to discover causal relationships. The MPC is the current validated release planner.

---

## Current Reservoir Network

The validated prototype network is a linear four-reservoir cascade:

| ID | Reservoir |
|---|---|
| **A** | Anayirankal |
| **B** | Ponmudi |
| **C** | Idamalayar |
| **D** | Idukki |

```text
A ──► B ──► C ──► D
```

Four reservoirs are currently supported and validated end-to-end (Stage 15). The topology is
declared in `src/network_env/topology_config.yaml`. In that configuration the reservoir
parameters are classified as `OBSERVED` or `ASSUMED_FOR_PROTOTYPE`, and the linear cascade is a
**prototype-level mapping** rather than a verified physical model of real Kerala reservoir
connectivity. Adding or removing reservoirs is a configuration change — no Python source change
is required.

---

## AI Components

### LSTM V3 — Temporal Inflow Forecasting

The frozen LSTM V3 model produces **future-day inflow point forecasts** in MCM/day at +1, +3
and +7 day horizons. It is the temporal forecasting component of the pipeline. The trained
weights and scaler are frozen artifacts and must not be modified (see
[Frozen Artifacts](#frozen-artifacts)).

### GNN — Spatial / Dependency Advisory

The Graph Neural Network provides a **spatial / dependency representation** over the reservoir
graph and runs alongside the forecasting path. Its output is **advisory**: it informs analysis
and the digital twin, and it does **not** directly command gates, perform release optimisation,
or replace the LSTM. The Stage 14 work formalised this advisory boundary.

### MPC — Coordinated Release Planning

The MPC planner searches a discrete candidate space of gate levels across the four reservoirs
with a multi-step look-ahead, simulating the resulting trajectories and scoring them against the
control objective. It produces the coordinated release plan that is then passed to the safety
layer. This is the current validated release-planning path.

### Safety Layer

The safety layer evaluates proposed commands against operational constraints before they become
gate/release commands, and includes a downstream capacity guard that can correct a proposal that
would violate downstream limits. Safety enforcement is independent of the planner.

---

## Digital Twin

The **Python/FastAPI backend is the source of truth**. A single authoritative simulation
instance (`src/dashboard/api/state_manager.py`) owns the reservoir network state and publishes it
over REST and WebSocket. Both the 3D digital twin and the Streamlit dashboard are
**read-only clients** of that state — they do not own, produce or overwrite simulation state.

---

## Validation

All figures below are **development / validation results** obtained in the project's
development/test environment. They are evidence of the validated software pipeline, not a
guarantee of behaviour for arbitrary real-world reservoirs.

### LSTM V3 — frozen model performance (test set)

| Horizon | R² |
|---|---|
| 1-day | 0.759 |
| 3-day | 0.654 |
| 7-day | 0.501 |

Source: `results/lstm_pytorch_v3_logtarget/lstm_v3_metrics_original_units.csv`.

### Phase 15.3 — MPC vs baseline regression (74 V3 dates)

| Metric | Baseline | MPC |
|---|---|---|
| Overflow events | 7 | **0** |
| Overflow volume (MCM) | 10.75 | **0** |
| Downstream violations | 8 | **0** |
| Peak downstream | 60 | **30** |
| Total release | 2017.25 | 2031.00 |

Mass-balance residual ≈ −6.82 × 10⁻¹³. This regression is protected by the Stage 16 hard gate.

### Stage 17 — Performance validation

Measured on the real production path (CPU-only environment).

| Metric | Result |
|---|---|
| Decision time (MPC `decide()`, mean) | ≈ 846.1 ms |
| Candidate vectors evaluated | 1296 |
| Throughput probe | 40/40 |
| Concurrency probe | 8/8 |
| Performance budget | Passed |
| Watchdog failure | None |
| Stage 17 test file | 38 passed |
| Frozen artifacts | Unchanged |

Source: `results/phase15_stage17_performance_validation/stage17_performance_evidence.json`.

### Test suite

**Last validated full-suite result: 790 passed, 0 failed, 0 errors, 0 skipped** (full repository
run at the Stage 17 validation point).

---

## Stage Progress

| Stage | Description | Status |
|---|---|---|
| 12 | Authoritative Twin State Audit | Complete |
| 13 | Streamlit Command Proxy | Complete |
| 14 | Advisory GNN Spatial Analysis Boundary | Complete |
| 15 | Live Multi-Reservoir End-to-End Verification | Complete |
| 16 | Phase 15.3 Regression Hard Gate | Complete |
| 17 | Performance Validation | Complete |
| 18 | Hardware Integration | **Planned / Next** |

Stages 12–17 correspond to the known-good checkpoint:

```text
0bff9a0 test(stage17): add performance validation evidence
595f98c test(stage16): add Phase 15.3 regression hard gate
e0df5c3 test(stage15): verify live multi-reservoir end-to-end chain
e4634eb feat(stage14): add advisory GNN spatial analysis boundary
68f074b feat(stage13): add authoritative Streamlit command proxy
14b890d feat(stage12): finalize authoritative twin state audit
```

---

## Hardware Integration — Stage 18

**Real physical integration is not yet complete.** No sensors are connected, no physical gates
are controlled, and no hardware-in-the-loop closed loop has been validated. The following is the
planned direction.

```text
        PC / AquaFlow
   (LSTM + GNN + MPC + Safety)
              │
              ▼
      Hardware Interface
              │
              ▼
            ESP32
              │
              ▼
  Sensors + miniature gate actuators
              │
              ▼
   Physical reservoir prototype
              │
              ▼
    Telemetry back to AquaFlow
```

The PC retains the main intelligence (LSTM → MPC → Safety), with GNN advisory analysis running
alongside the control path. The ESP32 acts as the hardware interface/controller for a miniature
physical system.

For the initial physical prototype the intended architecture is **not** four ESP32 boards. The
current practical prototype direction is:

```text
ONE ESP32
  + FOUR water-level sensors
  + FOUR miniature gate actuators
  + FOUR miniature reservoir tanks
```

Planned Stage 18 workstreams:

| Sub-stage | Scope |
|---|---|
| 18.1 Hardware abstraction | Define sensor and actuator interfaces; mock hardware first; preserve the existing software control pipeline |
| 18.2 ESP32 firmware | Sensor acquisition, actuator/gate control, telemetry, independent safety handling |
| 18.3 Communication layer | AquaFlow backend ↔ ESP32; telemetry from ESP32; commands from AquaFlow |
| 18.4 Dashboard integration | Hardware telemetry and actuator state, synchronised with the authoritative software state |
| 18.5 Hardware-in-the-loop validation | Release decision → command reaches ESP32 → actuator responds → sensor telemetry returns → AquaFlow updates system state |

**Potential initial hardware (not purchased / not integrated):**

- 1 × ESP32 DevKit
- 4 × HC-SR04 ultrasonic sensors
- 4 × SG90 miniature servos
- 4 × transparent miniature tanks
- 1–2 × small water pumps
- Tubing, breadboard, jumper wires
- Resistors / voltage-divider components
- Suitable 5V power supply
- Optional flow sensor(s)

---

## Project Structure

```text
AI-BASED-MULTI-RESERVOIR-WATER-MANAGEMENT-AND-FLOOD-PREVENTION-SYSTEM/
├── app.py                       # One-command launcher (FastAPI backend + 3D twin)
├── pyproject.toml               # pytest configuration (pythonpath = ["."])
├── requirements.txt             # Runtime and test dependencies
│
├── configs/
│   └── simulation/
│       └── four_reservoir_demo.json
│
├── data/
│   ├── raw/                     # Historical reservoir / rainfall / river data
│   ├── interim/
│   ├── processed/               # Cleaned, scaled, sequence and graph datasets
│   └── external/
│
├── docs/
│   ├── ProjectRoadmap.md
│   ├── Resources.md
│   └── Week-1.md
│
├── images/
│
├── models/
│   ├── lstm_pytorch_v3_logtarget/     # FROZEN LSTM V3 artifacts
│   ├── gcn_lstm_gated_v1/             # GNN / GCN-LSTM research models
│   └── lstm_pytorch*/                 # Earlier LSTM research iterations
│
├── notebooks/
│
├── results/
│   ├── lstm_pytorch_v3_logtarget/     # Frozen metrics + test predictions
│   ├── phase15_stage12_authoritative_twin_state/
│   ├── phase15_stage13_streamlit_command_proxy/
│   ├── phase15_stage14_gnn_advisory/
│   ├── phase15_stage15_end_to_end/
│   ├── phase15_stage16_regression_hard_gate/
│   └── phase15_stage17_performance_validation/
│
├── scripts/                     # Stage 3–17 validation / reproduction scripts
│
├── src/
│   ├── common/                  # Unit definitions
│   ├── controller/              # MPC, safety layer, downstream capacity guard, objective
│   ├── dashboard/               # FastAPI backend + Streamlit app + 3D twin component
│   ├── data_quality/            # Dataset construction, temporal splits, scaling
│   ├── graph/                   # Reservoir graph builders
│   ├── management/              # Risk engine and management data loader
│   ├── modeling/                # LSTM / GNN inference and training modules
│   ├── network_env/             # Authoritative network, adapters, topology config
│   ├── prediction/              # Baseline and feature-engineering experiments
│   ├── simulator/               # Offline research simulation engine
│   ├── rl/                      # Placeholder package (no implementation yet)
│   ├── hardware/                # Placeholder package (Stage 18)
│   └── continual_learning/      # Placeholder package (no implementation yet)
│
├── tests/                       # Unit / integration / stage test suite
├── LICENSE
└── README.md
```

---

## Installation

The project is developed against the Conda environment **`ai-lab`**.

```bash
conda activate ai-lab
pip install -r requirements.txt
```

`requirements.txt` covers the data-processing, visualisation, deep-learning
(`torch`, `torch-geometric`), backend (`fastapi`, `uvicorn`) and testing (`pytest`, `httpx`)
dependencies. The `python app.py` launcher can also install any missing runtime packages for
you (`python app.py --install` installs/verifies dependencies and exits).

> The FastAPI backend imports `torch_geometric` at module import time, so the GNN dependency must
> be installed for the backend to start.

---

## Running the System

### 3D Digital Twin + API (FastAPI + WebSocket)

From the project root:

```bash
python app.py                 # start on port 8000 and open the browser
python app.py --port 8080     # start on a chosen port
python app.py --no-browser    # start without opening a browser
python app.py --install       # only install/verify dependencies, then exit
```

This serves the 3D Digital Twin UI (`http://127.0.0.1:8000/`), the REST API
(`http://127.0.0.1:8000/api/...`) and the live WebSocket feed (`ws://127.0.0.1:8000/ws/state`).

### Streamlit operator dashboard

```bash
streamlit run src/dashboard/app.py
```

The Streamlit dashboard is a **read-only client** of the authoritative digital-twin state.

### Reproducibility / validation scripts

```bash
py scripts/stage3_phase15_3_reproduction.py        # canonical Phase 15.3 reproduction
py scripts/run_stage16_regression_hard_gate.py     # Phase 15.3 regression hard gate
py scripts/run_stage17_performance_validation.py   # Stage 17 performance evidence
```

---

## Testing

Tests are discovered from `tests/` (plus test modules inside `src/` and `results/`), and
`pyproject.toml` puts the repository root on `sys.path` so `src.*` imports resolve.

```bash
py -m pytest -q
# or
python -m pytest -q
```

**Last validated full-suite result: 790 passed, 0 failed, 0 errors, 0 skipped.** The Stage 17
test file alone (`tests/test_stage17_performance_validation.py`) recorded **38 passed**.

---

## Frozen Artifacts

The following LSTM V3 artifacts are **frozen** and must not be modified, retrained or
regenerated. Their hashes are recorded in the integrity manifest and re-verified by Stages 3,
16 and 17.

| Artifact | Path |
|---|---|
| Frozen model | `models/lstm_pytorch_v3_logtarget/best_model.pt` |
| Frozen scaler | `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` |
| Frozen predictions | `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` |

> ⚠️ **Contributors:** do not edit, overwrite, re-fit or regenerate these files. The Phase 15.3
> results and the Stage 16 regression hard gate depend on them being byte-identical.

The Stage 15 fixture / protected-output files in `results/phase15_stage*` are likewise
evidence artifacts and should be treated as read-only.

---

## Limitations / Current Scope

- **Four reservoirs** are currently validated end-to-end.
- The reservoir topology is a **prototype mapping** (`ASSUMED_FOR_PROTOTYPE` parameters), not a
  verified physical model of real reservoir connectivity.
- The **GNN is advisory** — it does not control gates or replace the LSTM.
- The **MPC is the current validated release planner**.
- **MARL / RL is not the current validated production control path.** The `src/rl/`,
  `src/hardware/` and `src/continual_learning/` packages are placeholders with no implementation.
- **Real physical hardware integration is not yet complete**; no sensors are connected and no
  physical gates are controlled.
- **Hardware-in-the-loop operation is future work** (Stage 18+).
- Results are **prototype / development validation** and are **not a claim of real dam
  operational safety**.

---

## Roadmap

This is the **current implementation roadmap** (not a quotation of an earlier planning document):

| Stage | Focus | Status |
|---|---|---|
| 17 | Performance Validation | Complete |
| 18 | Hardware Interface / ESP32 | **Next** |
| 19 | Physical Prototype / Hardware-in-the-Loop | Planned |
| 20 | Closed-Loop Demonstration / Final Validation | Planned |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements / References

- *Sutcliffe, J. V., & Parks, Y. P. (1999). The Hydrology of the Nile. IAHS Special Publication.*
- *Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.*
- [USGS Hydrological Data Archives](https://waterdata.usgs.gov/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [NetworkX documentation](https://networkx.org/documentation/stable/)
- [Plotly Python documentation](https://plotly.com/python/)

Additional project resources (literature, datasets and tooling links) are collected in
[`docs/Resources.md`](docs/Resources.md).
