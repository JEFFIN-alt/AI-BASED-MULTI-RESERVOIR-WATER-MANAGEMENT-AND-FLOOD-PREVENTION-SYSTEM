# Phase 14.4 Step 2: Interactive Virtual Reservoir Testing Environment

## 1. What Was Implemented
We implemented a fully interactive software testing environment integrated directly into the existing Streamlit dashboard (`src/dashboard/simulation_page.py`). This environment serves as a virtual laboratory, enabling researchers to stress-test the project's downstream management logic under custom, hypothetical extreme weather conditions.

## 2. Interactive Environment
The environment leverages Streamlit to provide real-time configuration of four virtual reservoirs. The UI exposes:
- Primary inflow for each reservoir
- Rainfall, humidity, and storm multiplier modifiers
- Controller selection
- Dynamic visualizations of the cascade's behavior (storage, inflows, and gate decisions)

## 3. How Scenarios are Created
Scenarios are generated on the fly. The user defines a 30-day baseline context and can inject a localized storm event (e.g., "Day 15 for 5 days"). The simulator calculates the daily trajectory across the 30 days and feeds the states to the Risk Engine and Controllers exactly as if processing historical data.

## 4. Virtual Hydrology (Synthetic Stress Modifiers)
We explicitly implemented an isolated virtual hydrology layer (`src/simulator/hydrology.py`) to prevent scientific misrepresentation.
**Formula:**
`Synthetic Inflow = Base Inflow × Rainfall Multiplier × (1 + Humidity/100) × Storm Multiplier`
*This is explicitly labeled as a synthetic simulation assumption, not a calibrated hydrological model.*

## 5. Reservoir Interaction (Cascade)
The four reservoirs form a sequential cascade (A → B → C → D). The simulation enforces a strict 1-day routing delay between nodes. When A releases water, B sees that exact volume arrive in its mass balance the following day. If D's release exceeds 50 MCM/day, downstream flooding is recorded.

## 6. How V3 is Used
V3 is strictly protected. During a synthetic stress test, we cannot query V3 for predictions of hypothetical weather without modifying and retraining the model, which is forbidden. The Forecast-Aware controller degrades gracefully, falling back to reactive rules and explicitly recording that no forecast was available for the synthetic scenario. V3's real test artifacts are only read when replaying historical test-set scenarios.

## 7. How the Risk Engine is Used
The validated Risk Engine (`src/management/risk_engine.py`) remains unmodified. The simulator proxies a generic 0-100% water level reading to feed Channel 1 of the Risk Engine. This proves the Risk Engine's deterministic logic works regardless of the source data.

## 8. Management Decisions
Decisions are completely deterministic and explainable:
- **HIGH RISK**: 100% Gate
- **ALERT**: 50% Gate (25% if downstream is struggling)
- **WATCH**: 25% Gate (5% if downstream is struggling)
- **NORMAL**: 5% Minimum Environmental Flow

## 9. Controller Modes
The UI allows side-by-side comparison of:
- **Reactive Baseline**: Uses only current water level.
- **Forecast-Aware**: Uses V3 point-forecasts where available.

## 10. Simulation Assumptions
- Gate positions map linearly to volume released.
- Routing delay is discretized to exactly 1 day.
- Elevation-storage curves are simplified to a linear percentage.
- The downstream flood limit is an arbitrary 50 MCM/day stress bound.

## 11. Safety Constraints
- Gate commands are clamped between 0-100%.
- Negative storage mathematically throws an error (and is actively prevented).
- Inflow exceeding capacity forces an immediate spill (Overflow Event) and logs a warning.

## 12. Test Results
The automated test suite (`src/simulator/test_simulator.py`) confirms:
- **Mass conservation**: Tested by attempting to over-release an empty reservoir.
- **Absolute capacity limits**: Tested by forcing massive inflow on a full reservoir and verifying overflow.
- **Cascade routing**: Verified the exact 1-day lag between A's release and B's inflow.

## 13. V3 Integrity Verification
The implementation strictly isolates the simulator from the model pipeline. `models/` and `results/lstm_pytorch_v3_logtarget/` remain entirely unmodified.

## 14. Example Stress-Test Capabilities
A researcher can now define an "Extreme Monsoon Stress Test":
- Reservoir A base inflow = 50 MCM/day
- Rainfall = 3.0x
- Humidity = 90%
- Storm starts on day 10, lasts 5 days.
The UI immediately renders the resulting chaotic inflows, the cascading flood wave moving down the reservoirs, and the controller's desperate attempts to manage the gate positions.

## 15. Limitations
This is a software-only evaluation layer. It assumes perfect instantaneous control of gates and linear hydrological responses.

## 16. Files Created
- `src/dashboard/simulation_page.py`
- `src/simulator/hydrology.py`
- `src/simulator/test_simulator.py`
- `results/phase14_simulation/PHASE_14_4_STEP2_REPORT.md`

## 17. Files Modified
- `src/dashboard/app.py` (Added sidebar navigation)
- `src/simulator/engine.py` (Added synthetic generation logic)
- `src/simulator/environment.py` (Updated to consume dynamic config dictionaries)

## 18. Files Intentionally Untouched
- All LSTM V3 frozen model artifacts.
- GNN experiment outputs.
- `src/management/risk_engine.py`

## 19. How to Launch the Environment
```bash
streamlit run src/dashboard/app.py
```
*(Select "Interactive Simulator Laboratory" from the sidebar navigation)*

## 20. Recommended Next Phase
The simulator successfully visualizes the logic of the management controller. The next logical phase is **Phase 14.5: Hardware Integration (ESP32)**, where the 0-100% gate decisions produced by this exact controller are transmitted over USB Serial to physically move a servo.
