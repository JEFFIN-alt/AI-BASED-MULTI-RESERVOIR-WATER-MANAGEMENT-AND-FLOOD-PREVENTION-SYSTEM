# Phase 14.4 Step 0: Virtual Four-Reservoir Management Simulation Design

## 1. WHAT WE ALREADY HAVE
- **LSTM V3 Forecasting Model**: Frozen, produces point forecasts (`target_1d`, `target_3d`, `target_7d`) for 16 reservoirs.
- **Risk Engine (Point-Forecast Warning Engine)**: A deterministic logic component assessing risk based on current storage vs. thresholds, and forecast vs. 95th percentile historical inflow.
- **Historical Daily Telemetry**: Contains live storage, water level, inflow, total outflow, spillway release, and powerhouse discharge (`data/processed/kerala_reservoir_clean.csv`).
- **Metadata**: FRL, MWL, and Blue/Orange/Red warning levels for the reservoirs (`live.json`, `historical_inflow_thresholds.json`).

## 2. WHAT DATA IS AVAILABLE
From the existing CSVs and JSONs, we have:
- Verified maximum capacities (`liveStorageAtFRL`).
- Verified warning thresholds (Blue, Orange, Red levels).
- Historical daily inflows (MCM/day) for scenario generation.
- Historical total outflows and spillway releases.
- 95th percentile historical inflow limits.

## 3. WHAT DATA IS MISSING (Explicitly Unavailable)
- **Physical Cascade Topology**: The dataset does not define which reservoir flows into which downstream reservoir.
- **Routing Delay**: The time it takes for water released from an upstream dam to reach a downstream dam.
- **Gate Rating Curves**: The relationship between physical gate position (%) and volumetric discharge ($m^3/s$ or MCM/day).
- **Elevation-Volume Curves**: Area-capacity tables defining exact storage volume at every millimeter of water level.
- **Downstream Channel Capacity**: The maximum safe flow rate of the rivers between reservoirs before flooding occurs.

## 4. WHAT MUST BE SIMULATED (Explicit Assumptions)
Because the above data is missing, we must construct a **virtual** environment using explicit assumptions:
- **Virtual Topology**: We will assume a strict serial cascade (A → B → C → D → Downstream River).
- **Gate Abstraction**: Gate position (0-100%) will linearly map to a fraction of the maximum possible release capacity.
- **Virtual Routing**: Water released from Reservoir A arriving at Reservoir B will be modeled with a discrete fixed delay (e.g., 1 day).
- **Downstream Capacity**: We will invent a hypothetical safe maximum flow limit for the final downstream river to penalize excessive combined releases.
- **Level-Storage**: We will use a simplified linear or generic polynomial relationship between live storage and water level to translate simulated volumes back to levels for the Risk Engine.

## 5. PROPOSED FOUR-RESERVOIR STRUCTURE
We will select 4 distinct reservoirs from the V3 pool (e.g., Anayirankal [Small], Ponmudi [Medium], Idamalayar [Large], Idukki [Very Large]) to provide baseline capacities and historical inflows.
- **Reservoir A (Upstream)** → Releases to B
- **Reservoir B (Mid-Upper)** → Releases to C
- **Reservoir C (Mid-Lower)** → Releases to D
- **Reservoir D (Terminal)** → Releases to the final downstream river.

*Note: This is a virtual cascade. We do not claim this represents the true geographic hydrology of Kerala.*

## 6. PROPOSED STATE EQUATIONS
For a given Reservoir $i$ at discrete daily timestep $t$:
- $Storage_{i, t+1} = Storage_{i, t} + Inflow_{i, t} - Release_{i, t}$
- $Inflow_{i, t} = Local\_Catchment\_Inflow_{i, t} + Release_{i-1, t - delay}$
- $Level_{i, t} = f(Storage_{i, t})$ (using a simplified interpolation between 0 storage and `liveStorageAtFRL`)
- $Release_{i, t} = Gate\_Position_{i, t} \times Max\_Release\_Capacity_i$

*Constraints*: $Storage_{i, t+1}$ is bounded by 0 and $Maximum\_Capacity_i$. If calculated storage exceeds maximum capacity, the excess is forcibly routed to $Release_{i, t}$ as an **Overflow Event**.

## 7. PROPOSED MANAGEMENT POLICY
The controller is a deterministic policy evaluating risk and deciding gate position (0%, 25%, 50%, 100%).
- **Ruleset**: 
  - If Status = HIGH RISK → Gate = 100%
  - If Status = ALERT → Gate = 50%
  - If Status = WATCH → Gate = 25%
  - If Status = NORMAL → Gate = Minimum required environmental flow (e.g., 5%)
- **Cascade Coordination Penalty**: Before finalizing a release, the controller checks the downstream reservoir's status. If Reservoir B is in HIGH RISK, Reservoir A's gate is restricted to 0% (unless A is also overflowing) to prevent compounding floods.

## 8. HOW V3 WILL CONNECT TO THE SIMULATION
The V3 model outputs point forecasts (`target_1d`, `target_3d`, `target_7d`).
**Critically**, the simulation will **not** multiply these by 3 or 7.
At timestep $t$, the simulator will query the historical V3 predictions CSV for the simulated date $t$. The Risk Engine will process these point-forecasts. The controller will react to the Risk Engine's advisory. The simulation physically advances using the *actual* historical local catchment inflow for day $t$, while the controller operates based on the *forecasted* view of day $t+1$, $t+3$, and $t+7$.

## 9. HOW FLOOD RISK WILL BE CALCULATED
Flood risk is calculated using two mechanisms:
1. **Reservoir Risk**: Using the Phase 14.1C Risk Engine (evaluating simulated water level against Blue/Orange/Red thresholds, and V3 forecasts against historical 95th percentiles).
2. **Downstream Risk**: A metric checking if $Release_{D, t}$ exceeds the assumed downstream channel capacity.

## 10. BASELINE VS FORECAST-AWARE EXPERIMENT
We will test two controllers:
- **Reactive Baseline**: Ignores V3 forecasts entirely. Sets gate position based *only* on current water level (Channel 1 of the Risk Engine).
- **Forecast-Aware Policy**: Uses the full Risk Engine (Channel 1 + Channel 2). Opens gates preemptively when a 3-day or 7-day severe inflow alert is triggered, creating a buffer before the storm arrives.

## 11. METRICS
The simulation will compute the following at the end of a scenario run:
- Total Overflow Events (days where capacity was forcibly exceeded).
- Number of Red Threshold violations.
- Maximum Downstream Flow Peak (MCM/day).
- Number of Downstream Capacity Violations.
- Total Unnecessary Release (volume released when reservoir was below Blue level and no storm arrived).
- Minimum Storage (to ensure water wasn't needlessly emptied).

## 12. TEST SCENARIOS
1. **NORMAL INFLOW**: A 30-day historically average summer period.
2. **HEAVY INFLOW**: A 30-day monsoon period with regular heavy rains.
3. **EXTREME INFLOW**: A 30-day window containing a 1-in-100 year anomaly (e.g., August 2018 historical data).
4. **MULTI-RESERVOIR CASCADE EVENT**: An extreme storm hitting the upstream catchment (Reservoir A/B) but missing the lower catchment, testing the routing behavior.

## 13. PYTHON PROJECT STRUCTURE
```
src/simulator/
  ├── __init__.py
  ├── environment.py       # Defines the Reservoir and Cascade classes + mass balance
  ├── scenarios.py         # Loads/generates inflow timeseries
  ├── controllers.py       # Implements Baseline and Forecast-Aware logic
  ├── engine.py            # The main daily simulation loop
  └── metrics.py           # Calculates evaluation metrics
```
Configuration and scenario definitions will reside in a new `config/` or within the simulator module, and results will be saved to `results/phase14_simulation/`.

## 14. FUTURE DASHBOARD INTEGRATION
The Streamlit dashboard (`src/dashboard/app.py`) can be extended with a "Simulation" page. This page would load the results of a simulation run and plot:
- A 4-panel grid showing the storage trajectory of Reservoirs A, B, C, D over the 30-day scenario.
- A comparison bar chart showing the Metrics (Baseline vs Forecast-Aware).
- A timeline of Gate Position decisions.

## 15. FUTURE ESP32 INTEGRATION
Once the virtual simulation proves the controller logic is safe and effective, the output of the controller (e.g., Gate = 50%) can be piped out via PySerial. The ESP32 will receive this JSON payload and map 50% to a 90-degree servo angle, bridging the virtual policy to the physical prototype.

## 16. MATLAB/Simulink ROLE, IF ANY
No MATLAB dependency is required. The mass-balance routing equations (simple explicit Euler integration at daily timesteps) are trivial to implement in Python (`numpy`/`pandas`). Keeping it in Python ensures seamless integration with V3 and the Risk Engine. MATLAB could only be useful later if strict PID control-theory analysis of the gate actuators is requested, which is out of scope for this AI-focused project.

## 17. SCIENTIFIC LIMITATIONS
- **Not a real hydraulic model**: We assume instantaneous mixing and linear level-storage curves.
- **Fictitious Topology**: The 4-dam cascade is a logical testbed, not a geographical reality.
- **Fixed Routing Delay**: Real river routing is non-linear (Manning's equation, kinematic waves). We use a fixed discrete daily lag.
- **Point Forecasts**: V3 cannot perfectly predict the area under the hydrograph curve, so the controller is operating on sparse temporal glimpses.

## 18. EXACT IMPLEMENTATION PLAN FOR STEP 1
1. Create `src/simulator/environment.py` with `VirtualReservoir` and `VirtualCascade` classes.
2. Create `src/simulator/scenarios.py` to extract 30-day slices from `kerala_reservoir_clean.csv`.
3. Create `src/simulator/controllers.py` implementing the Reactive and Forecast-Aware strategies.
4. Create a main execution script `run_simulation.py` to run the 4 scenarios on both controllers and output CSV metrics.

## 19. SAFETY / FAILURE CONDITIONS
- If inflow + existing storage > capacity, the environment physically *forces* a spillway release (Overflow Event) to prevent mathematical >100% storage.
- If V3 predictions are missing for a date, the Forecast-Aware controller must gracefully degrade to the Reactive Baseline behavior.

## 20. FILES THAT WOULD EVENTUALLY BE CREATED
- `src/simulator/environment.py`
- `src/simulator/scenarios.py`
- `src/simulator/controllers.py`
- `src/simulator/engine.py`
- `src/simulator/run_simulation.py`
- `results/phase14_simulation/simulation_results.csv`
- `results/phase14_simulation/simulation_plots.png`

---

## FINAL DECISION
**CLASSIFICATION: A — READY TO IMPLEMENT**

The simulation design safely isolates the AI from physical assumptions by establishing an explicitly "virtual" testbed. It strictly adheres to V3's point-forecast limitations, prevents invalid storage math, and establishes a scientifically defensible method to prove whether forecast-awareness actually improves multi-reservoir decision-making compared to a reactive baseline.
