# Phase 14.2: End-to-End System Architecture

## 1. Executive Summary

This document defines the complete system architecture for the **AI-Based Multi-Reservoir Water Management and Flood Prevention System** BTech project. The architecture integrates a frozen LSTM V3 inflow forecasting model, a validated Point-Forecast Warning Engine, a Streamlit operator dashboard, and a physical ESP32-based miniature reservoir prototype into a cohesive, scientifically defensible demonstration system.

The design explicitly separates AI forecasting from physical actuation through a deterministic, bounded control-policy layer. The AI never directly commands a servo. Instead, the risk engine produces a human-readable advisory, a deterministic control policy maps that advisory to a bounded actuator state, and a hardware safety layer on the ESP32 enforces physical limits independently of the server.

---

## 2. Current Repository Capabilities — Forensic Inventory

### EXISTS and VALIDATED
| Component | Location | Status |
|---|---|---|
| Raw reservoir data (KSEB + Irrigation) | `data/raw/reservoir/Kerala-Dam-Water-Levels/` | 18 historic JSON files, `live.json`, `irrigation_live.json` |
| Cleaned dataset | `data/processed/kerala_reservoir_clean.csv` | Validated |
| Supervised forecasting dataset | `data/processed/forecasting_supervised_7day.csv` | Validated |
| Train/val/test splits | `data/processed/model_splits/` | Validated |
| Historical inflow thresholds (training only) | `data/processed/historical_inflow_thresholds.json` | 16 reservoirs, 95th percentile |
| LSTM V3 model checkpoint | `models/lstm_pytorch_v3_logtarget/best_model.pt` | Frozen |
| V3 log-target scaler | `models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl` | Frozen |
| V3 test predictions | `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | 1100 rows, 16 reservoirs |
| V3 forensic audit | `results/v3_final_audit/V3_FINAL_AUDIT_REPORT.md` | GREEN |
| GNN experiment results | `results/gcn_lstm_v1/`, `v1_1_identity/`, `v1_2/`, `gated_v1/` | CLOSED |
| Thesis visualizations | `results/thesis_visualizations/` | 20 figures, 4 tables |
| Management data loader | `src/management/management_data_loader.py` | Validated |
| Risk engine (Point-Forecast) | `src/management/risk_engine.py` | Validated (Phase 14.1C) |
| Risk engine tests | `src/management/test_risk_engine.py` | 14 tests passing |
| Management reports | `results/phase14_management_engine/` | 3 reports |
| `requirements.txt` | Root | `streamlit`, `plotly` listed |

### EXISTS but EMPTY (placeholder `.gitkeep` only)
| Component | Location |
|---|---|
| Hardware module | `src/hardware/` |
| Dashboard module | `src/dashboard/` |
| Simulator module | `src/simulator/` |
| RL module | `src/rl/` |
| Images directory | `images/` |

### DOES NOT EXIST
| Component | Notes |
|---|---|
| ESP32 firmware / Arduino code | No `.ino`, `.cpp`, or PlatformIO files anywhere |
| Servo control code | None |
| Sensor reading code | None |
| MQTT / Serial communication code | None |
| Streamlit dashboard application | None |
| API / server code | None |
| Hardware wiring diagrams | None |
| Physical prototype documentation | None |

---

## 3. Final System Architecture — Layer Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA                                │
│  Historical reservoir telemetry, live.json metadata,            │
│  V3 test predictions, historical inflow thresholds              │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 2: AI FORECASTING                      │
│  Frozen LSTM V3 (best_model.pt)                                 │
│  Outputs: target_1d, target_3d, target_7d (point forecasts)    │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 3: RISK / DECISION SUPPORT             │
│  Point-Forecast Warning Engine (risk_engine.py)                 │
│  Channel 1: waterLevel vs blue/orange/red thresholds            │
│  Channel 2: forecast vs historical 95th percentile              │
│  Output: NORMAL / WATCH / ALERT / HIGH RISK + reason string     │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 4: OPERATOR DASHBOARD                  │
│  Streamlit application                                          │
│  Displays: reservoir status, forecasts, risk, prototype state   │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 5: CONTROL POLICY                      │
│  Deterministic, bounded mapping:                                │
│  Risk Status → Gate Position (0-100%)                           │
│  Configurable, not AI-driven                                    │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 6: COMMUNICATION                       │
│  USB Serial (pyserial ↔ ESP32)                                  │
│  JSON-line protocol                                             │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 7: HARDWARE SAFETY                     │
│  ESP32 firmware with independent safety constraints             │
│  Local watchdog, max-level emergency open, heartbeat timeout    │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 8: PHYSICAL PROTOTYPE                  │
│  Single miniature reservoir container                           │
│  Ultrasonic water-level sensor (HC-SR04)                        │
│  SG90 micro servo (outlet gate)                                 │
│  Manual inflow (pour/tube)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture

The key design principle is **defense in depth**. The AI model (Layer 2) never touches the servo. Its output flows through the Risk Engine (Layer 3), which produces a human-readable advisory. The Control Policy (Layer 5) translates that advisory into a bounded gate command. The ESP32 firmware (Layer 7) independently enforces physical safety limits regardless of what the server sends. If communication fails, the ESP32 defaults to a safe state (gate open).

---

## 4. AI → Risk → Control Separation

```
V3 Forecast ──→ Risk Engine ──→ Dashboard ──→ Control Policy ──→ ESP32
  (numbers)      (status +       (human       (bounded gate       (safety
                  reason)         review)      percentage)         enforced)
```

**Critical boundaries:**
1. V3 outputs raw inflow numbers. It has no concept of "risk" or "gate."
2. The Risk Engine outputs a categorical status with an explanation. It has no concept of "servo."
3. The Dashboard displays the status for human review. It can optionally forward the status to the control policy.
4. The Control Policy is a simple, deterministic lookup table (not ML). It maps `NORMAL → gate 0%`, `WATCH → gate 25%`, `ALERT → gate 50%`, `HIGH RISK → gate 100%`. These percentages are configurable parameters, not hardcoded magic numbers.
5. The ESP32 receives a gate-percentage command but independently validates it against its own sensor reading. If the local water level exceeds a hardware-defined maximum, the ESP32 opens the gate to 100% regardless of what the server says.

---

## 5. Physical Prototype Design

### Recommended Configuration: Single Reservoir + Software Multi-Reservoir

**Option C is recommended**: One physical miniature reservoir demonstrating the control pipeline, with the remaining 15 reservoirs displayed as software-only entries on the dashboard using historical data and V3 predictions.

| Option | Scientific Credibility | Hardware Complexity | Cost | Reliability | Demo Value | Time |
|---|---|---|---|---|---|---|
| A: One physical reservoir | Medium | Low | Low | High | Medium | Fast |
| B: Multiple physical reservoirs | High | Very High | High | Low | High | Slow |
| **C: One physical + 15 software** | **High** | **Low** | **Low** | **High** | **Very High** | **Fast** |
| D: Pure software simulation | Low | None | None | N/A | Low | Fastest |

**Rationale**: Option C demonstrates the full "multi-reservoir" concept on the dashboard (all 16 reservoirs visible with real data and forecasts) while proving the physical control loop works on one tangible prototype. Building 16 physical reservoirs is impractical for a BTech project and adds no scientific value — the AI and risk logic are identical regardless of whether the reservoir is physical or software-rendered.

### Hardware Components

| Component | Role | Estimated Category |
|---|---|---|
| ESP32 DevKit V1 (or NodeMCU-32S) | Microcontroller: reads sensor, drives servo, communicates via USB | NEEDS PURCHASE (unless already available) |
| HC-SR04 Ultrasonic Sensor | Measures water level (distance to surface) | NEEDS PURCHASE |
| SG90 Micro Servo (180°) | Controls miniature outlet gate | NEEDS PURCHASE |
| Plastic container (~30×20×15 cm) | Miniature reservoir body | ALREADY AVAILABLE (any waterproof container) |
| Flexible tubing + manual valve | Simulates controlled inflow | ALREADY AVAILABLE or minimal cost |
| USB cable (Micro-USB or USB-C) | ESP32 ↔ Laptop communication | ALREADY AVAILABLE |
| Breadboard + jumper wires | Prototyping connections | NEEDS PURCHASE (unless available) |
| 5V power supply (or USB power) | Powers ESP32 + servo | ALREADY AVAILABLE (USB from laptop) |
| Small water pump (optional) | Automated inflow simulation | OPTIONAL — manual pour is simpler and sufficient |

**No pump is strictly necessary.** For the demonstration, the operator can manually pour water into the container to simulate inflow. This is actually more realistic for a demo scenario because the operator can narrate: "I am now simulating a sudden monsoon inflow event" while pouring water, and the system responds in real time.

### Sensor Choice: HC-SR04 Ultrasonic

- Measures distance from the sensor (mounted at the top of the container) to the water surface.
- Water level = container height − measured distance.
- Range: 2–400 cm (more than sufficient for a 15 cm container).
- Accuracy: ~3 mm (adequate for prototype).
- Non-contact: does not corrode or require submersion.
- Alternative considered: capacitive/resistive water level strips — rejected because they are less precise and harder to calibrate.

### Servo Role

The SG90 servo rotates a small arm or flap that covers/uncovers a hole near the bottom of the container. This acts as the miniature "dam gate." At 0° (closed), water is retained. At 180° (fully open), water drains through the outlet. Intermediate positions (e.g., 45°, 90°) provide proportional drainage.

---

## 6. Dashboard Architecture

The dashboard will be built with **Streamlit** (already in `requirements.txt`). It will be located in `src/dashboard/app.py`.

### Panel 1: Multi-Reservoir Overview (Software)
For each of the 16 reservoirs:
- Reservoir name
- Current water level (from telemetry)
- Current risk status (from risk engine)
- Inflow forecast status
- Overall status with color coding (green/yellow/orange/red)

Data source: `live.json` metadata + `historical_inflow_thresholds.json` + V3 predictions CSV.

### Panel 2: Selected Reservoir Detail
When a reservoir is selected:
- Current water level vs. blue/orange/red thresholds (gauge chart)
- V3 1d/3d/7d point forecasts (bar chart)
- Historical 95th percentile threshold line
- Full risk engine output with human-readable reason
- Time-series of recent predictions (if available)

### Panel 3: Physical Prototype Monitor
- ESP32 connection status (Connected / Disconnected)
- Current mini-reservoir water level (from HC-SR04, in cm)
- Current servo/gate position (0–100%)
- Current control-policy state (NORMAL / WATCH / ALERT / HIGH RISK)
- Live sensor reading history (rolling chart, last 60 seconds)
- Manual override controls (for demonstration: buttons to set gate position manually)

### Panel 4: System Information
- Model information (LSTM V3, frozen)
- Threshold methodology (training-data 95th percentile)
- Scientific limitations disclaimer
- System mode (AUTO / MANUAL / DISCONNECTED)

### Data Sources — No Fabrication Rule
Every dashboard element must map to an actual data source. If telemetry is stale or unavailable, the dashboard must show "Data Unavailable" rather than interpolating or inventing values.

---

## 7. Communication Architecture: USB Serial

### Why USB Serial (not MQTT / WiFi / WebSocket)

| Method | Simplicity | Reliability | Offline | BTech Suitability | Complexity |
|---|---|---|---|---|---|
| **USB Serial** | **Very High** | **Very High** | **Yes** | **Excellent** | **Very Low** |
| WiFi HTTP | Medium | Medium | No | Good | Medium |
| MQTT | Medium | High | No | Over-engineered | High |
| WebSocket | Low | Medium | No | Over-engineered | High |

USB Serial is recommended because:
1. The ESP32 is physically tethered to the laptop during demonstration anyway.
2. No network configuration, no router dependency, no firewall issues.
3. `pyserial` on Python side + `Serial` on Arduino side — both are trivial.
4. Latency is sub-millisecond. Reliability is near-perfect.
5. The demonstration is a controlled lab/classroom environment, not a field deployment.

### Protocol: JSON-Line

**Server → ESP32 (command):**
```json
{"cmd": "SET_GATE", "position": 50, "ts": 1692612345}
```

**ESP32 → Server (telemetry):**
```json
{"type": "STATUS", "water_level_cm": 8.3, "gate_pct": 50, "mode": "AUTO", "ts": 1692612346}
```

- One JSON object per line, terminated by `\n`.
- ESP32 sends status at ~2 Hz (every 500 ms).
- Server sends commands only when the control policy state changes (event-driven, not polled).

---

## 8. Control Policy Design

The control policy is a **deterministic, bounded lookup table** — not a machine learning model.

| Risk Status | Gate Position | Rationale |
|---|---|---|
| `NORMAL` | 0% (closed) | No action needed, retain water |
| `WATCH` | 25% (partially open) | Precautionary drainage |
| `ALERT` | 50% (half open) | Active drainage to create buffer |
| `HIGH RISK` | 100% (fully open) | Maximum drainage |
| `UNKNOWN` | 0% (closed) | Safe default — do not drain without evidence |
| `DISCONNECTED` | Handled by ESP32 locally | See fail-safe design |

**These percentages are configurable parameters**, stored in a configuration dict/file that the operator can adjust before or during the demonstration. They are not derived from any ML model.

### What the Control Policy Does NOT Do
- It does NOT calculate optimal release volumes.
- It does NOT predict downstream effects.
- It does NOT attempt to minimize a cost function.
- It does NOT use reinforcement learning.

It is a simple, transparent, human-auditable mapping from risk status to gate position.

---

## 9. Fail-Safe Design

### ESP32-Side Safety (Independent of Server)

| Condition | ESP32 Behavior |
|---|---|
| Water level > hardware max (e.g., 12 cm) | Force gate to 100% immediately, regardless of server command |
| Water level > hardware warning (e.g., 10 cm) | Force gate to at least 50%, log warning |
| No server heartbeat for > 5 seconds | Enter `DISCONNECTED` mode, open gate to 100% |
| Servo command out of range (< 0 or > 100) | Clamp to [0, 100], log error |
| Sensor reading invalid (< 0 or > container height) | Ignore reading, maintain last valid state, log error |
| Sensor reading stuck (same value for > 10 seconds) | Flag sensor failure, open gate to 100% as precaution |

### Server-Side Safety

| Condition | Server Behavior |
|---|---|
| ESP32 disconnected | Dashboard shows "DISCONNECTED" prominently |
| V3 predictions unavailable | Risk engine returns `INSUFFICIENT_DATA`, gate stays at 0% |
| Stale telemetry data | Dashboard shows timestamp + "STALE" warning |
| Risk engine exception | Catch, log, return `UNKNOWN` status |

### Key Principle
The ESP32 must be independently safe. Even if the laptop crashes, the Python server dies, or the USB cable is yanked, the ESP32's local safety logic must protect the prototype. The ESP32's hardware max-level check is the **last line of defense** and operates entirely locally.

---

## 10. Real vs. Simulated Data Boundary

| Data Type | Source | Nature |
|---|---|---|
| Historical reservoir telemetry | `data/raw/reservoir/Kerala-Dam-Water-Levels/` | REAL historical data from Kerala dams |
| V3 inflow forecasts | `results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv` | AI-generated predictions from REAL historical inputs |
| Warning thresholds | `live.json` + `historical_inflow_thresholds.json` | REAL verified thresholds |
| Mini-reservoir water level | HC-SR04 sensor via ESP32 | REAL physical measurement |
| Mini-reservoir inflow | Manual pour or pump | SIMULATED — manually controlled |
| Gate position | SG90 servo | REAL physical actuation |

### What Must Be Stated in the Thesis

> "The miniature reservoir prototype demonstrates the end-to-end decision-support pipeline — from AI forecast to risk assessment to physical actuation — in a controlled laboratory environment. The prototype is NOT a hydrodynamically scaled model of a real Kerala dam. The physical dimensions, flow rates, and time scales of the prototype bear no calibrated relationship to the actual reservoirs. The prototype proves the feasibility of the AI-to-hardware control loop, not the hydraulic accuracy of the miniature system."

---

## 11. End-to-End Data Flow

```
┌──────────────────┐
│  Historical Data  │ ← data/raw/reservoir/
│  + V3 Predictions │ ← results/lstm_pytorch_v3_logtarget/
└────────┬─────────┘
         │ (CSV + JSON files)
         ▼
┌──────────────────┐
│   Risk Engine     │ ← src/management/risk_engine.py
│   (per reservoir) │
└────────┬─────────┘
         │ {overall_status, water_level_status, inflow_forecast_status, reason}
         ▼
┌──────────────────┐
│    Dashboard      │ ← src/dashboard/app.py (Streamlit)
│  16 reservoirs    │
│  + prototype panel│
└───┬──────────┬───┘
    │          │
    │ (display)│ (for selected physical reservoir)
    │          ▼
    │  ┌───────────────┐
    │  │ Control Policy │ ← Deterministic lookup
    │  │ Status → Gate% │
    │  └───────┬───────┘
    │          │ {"cmd": "SET_GATE", "position": 50}
    │          ▼
    │  ┌───────────────┐
    │  │  USB Serial    │ ← pyserial (Python) ↔ Serial (Arduino)
    │  │  JSON-line     │   ~2 Hz telemetry, event-driven commands
    │  └───────┬───────┘
    │          │
    │          ▼
    │  ┌───────────────┐
    │  │   ESP32        │
    │  │  + Safety Layer│ ← Independent max-level check
    │  └───┬───────┬───┘
    │      │       │
    │      ▼       ▼
    │  ┌──────┐ ┌──────┐
    │  │Servo │ │Sensor│ ← HC-SR04
    │  │(gate)│ │(level)│
    │  └──────┘ └──┬───┘
    │              │ water_level_cm
    │              ▼
    │      ┌───────────────┐
    │      │   ESP32        │ → {"type":"STATUS", "water_level_cm":8.3, ...}
    │      └───────┬───────┘
    │              │ (USB Serial back to server)
    │              ▼
    └──────── Dashboard ←── (live prototype panel updated)
```

### Connection Details

| Connection | Data | Direction | Format | Frequency | On Failure |
|---|---|---|---|---|---|
| Data → Risk Engine | Telemetry + forecasts + thresholds | Input | Python dicts | On dashboard load/refresh | Show "Data Unavailable" |
| Risk Engine → Dashboard | Status + reasons | Output | Python dict | Per refresh | Show "Engine Error" |
| Dashboard → Control Policy | Selected risk status | Internal | String | On status change | No command sent |
| Control Policy → Serial | Gate command | Server → ESP32 | JSON-line | Event-driven | ESP32 uses local safety |
| ESP32 → Serial | Sensor reading + state | ESP32 → Server | JSON-line | ~2 Hz (500 ms) | Dashboard shows "Disconnected" |
| Sensor → ESP32 | Distance measurement | Hardware | Analog/digital | ~10 Hz internally | ESP32 flags sensor error |
| ESP32 → Servo | PWM signal | Hardware | PWM | On command change | Servo holds last position |

---

## 12. Implementation Roadmap

| Phase | Name | Deliverable | Dependencies | Est. Time |
|---|---|---|---|---|
| **14.2** | System Architecture (this document) | Architecture report | Phase 14.1C complete | Done |
| **14.3** | Dashboard MVP | Streamlit app showing 16 reservoirs + risk status (software only, no hardware) | Risk engine, data files | 1 session |
| **14.4** | ESP32 Firmware | Arduino sketch: sensor reading, servo control, serial protocol, local safety | ESP32 + sensor + servo hardware | 1 session |
| **14.5** | Serial Bridge | Python `pyserial` module connecting dashboard to ESP32 | Phase 14.3 + 14.4 | 1 session |
| **14.6** | Dashboard Integration | Full dashboard with live prototype panel | Phase 14.3 + 14.5 | 1 session |
| **14.7** | Closed-Loop Demonstration | End-to-end demo: pour water → sensor reads → risk assessed → gate opens | All previous phases | 1 session |
| **14.8** | Final Validation & Documentation | Thesis-ready system documentation + demo recording | Phase 14.7 | 1 session |

### Critical Path
Phase 14.3 (dashboard) and Phase 14.4 (ESP32 firmware) are **independent** and can be developed in parallel. Phase 14.5 (serial bridge) requires both. Phase 14.6 and 14.7 are sequential.

---

## 13. Hardware Cost Estimate

| Component | Estimated Range (INR) | Category |
|---|---|---|
| ESP32 DevKit V1 | ₹300–500 | NEEDS PURCHASE |
| HC-SR04 Ultrasonic Sensor | ₹50–100 | NEEDS PURCHASE |
| SG90 Micro Servo | ₹50–100 | NEEDS PURCHASE |
| Breadboard + jumper wires | ₹100–200 | NEEDS PURCHASE (if not available) |
| Plastic container | ₹0–100 | ALREADY AVAILABLE |
| USB cable | ₹0 | ALREADY AVAILABLE |
| Flexible tubing | ₹0–50 | OPTIONAL |
| Small water pump | ₹200–400 | OPTIONAL (manual pour is sufficient) |
| **Total (minimum)** | **₹500–900** | |
| **Total (with pump)** | **₹700–1400** | |

---

## 14. Scientific Claim Boundaries

### The Project CAN Claim

1. "We developed an LSTM-based multi-horizon inflow forecasting model (V3) for 16 Kerala reservoirs achieving R² = 0.759 (1-day), 0.654 (3-day), and 0.501 (7-day)."
2. "We investigated spatial graph-based architectures (GCN-LSTM) and found that for this dataset, temporal features alone (LSTM) outperform spatiotemporal approaches."
3. "We designed a deterministic, dual-channel early-warning system that combines current water-level thresholds with historically-grounded high-inflow detection."
4. "We demonstrated the end-to-end AI-to-hardware pipeline through a physical ESP32-based miniature reservoir prototype."
5. "The system provides transparent, explainable decision support for human operators."

### The Project CANNOT Claim

1. ~~"The system can autonomously control a real dam."~~ — No downstream hydraulic model exists.
2. ~~"The system predicts floods."~~ — It detects elevated inflow conditions, not downstream flooding.
3. ~~"The miniature prototype is a scaled model of a real reservoir."~~ — It is a pipeline demonstration, not a hydraulic model.
4. ~~"The system optimizes water release volumes."~~ — No optimization or RL policy was validated.
5. ~~"The system predicts future reservoir storage."~~ — This was explicitly removed as scientifically invalid.
6. ~~"The GNN improved forecasting."~~ — It did not. The GNN experiments are documented as negative results.

---

## 15. Risks and Limitations

| Risk | Mitigation |
|---|---|
| HC-SR04 unreliable with splashing water | Mount sensor at sufficient height; use median filtering over 5 readings |
| Servo not waterproof | Keep servo above water line; use mechanical linkage to gate |
| USB cable constrains demo mobility | Acceptable for lab demo; WiFi upgrade is future work |
| V3 predictions are historical (Jan–Aug 2025 test set) | Clearly state this; the demo replays historical scenarios, not live forecasting |
| Single physical reservoir vs. "multi-reservoir" title | Dashboard shows all 16 with software data; physical prototype proves the control loop |
| Servo latency / hysteresis | SG90 response time (~100 ms) is negligible for this application |

---

## 16. Recommended Next Phase

**Phase 14.3: Dashboard MVP** should be implemented first.

Rationale:
- It requires only existing software artifacts (risk engine, data files, predictions).
- It does not require any hardware.
- It immediately demonstrates the "multi-reservoir" aspect of the project.
- It provides the user interface into which the physical prototype panel will later be integrated.
- It can be developed and tested independently of hardware procurement.

---

## 17. Architecture Classification

**CLASSIFICATION: A — READY FOR IMPLEMENTATION**

All data sources are verified. The risk engine is validated. The communication protocol is simple and well-understood. The hardware components are standard and inexpensive. The fail-safe design is comprehensive. The scientific claim boundaries are clearly defined.

No blocking information is missing.
