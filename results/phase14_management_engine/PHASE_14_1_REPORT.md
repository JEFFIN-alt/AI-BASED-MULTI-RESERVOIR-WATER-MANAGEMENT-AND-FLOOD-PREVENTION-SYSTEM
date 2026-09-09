# Phase 14.1: V3 Forecast → Flood-Risk Decision Engine

## 1. What We Built
We successfully built a deterministic Reservoir Management Risk Engine (`src/management/risk_engine.py`) alongside a metadata loader (`src/management/management_data_loader.py`). The engine is designed to translate current telemetry data and LSTM V3 inflow forecasts into an interpretable early-warning level (NORMAL, WATCH, ALERT, HIGH RISK) without relying on unverified control models or invented parameters.

## 2. Data Sources Used
- **Metadata**: Verified static fields parsed from `live.json` and `irrigation_live.json` via the data loader.
- **Current State**: Parsed from historical telemetry files (e.g., `historic_data/Idukki.json`).
- **Predictions**: V3 forecasted daily average inflow, passed from the `test_predictions_original_units.csv` artifacts.

## 3. Exact Fields Used
**From Telemetry:**
- `waterLevel` (meters)
- `liveStorage` (MCM - Million Cubic Meters)

**From Metadata:**
- `blueLevel` (meters)
- `orangeLevel` (meters)
- `redLevel` (meters)
- `liveStorageAtFRL` (MCM - live capacity at Full Reservoir Level)

**From V3 Forecasts:**
- `target_1d_prediction`
- `target_3d_prediction`
- `target_7d_prediction`

## 4. Risk Logic
The engine evaluates risk on two independent axes and takes the maximum severity across both:

**Axis 1: Current State (Water Level)**
- If `waterLevel` >= `redLevel` → HIGH RISK
- If `waterLevel` >= `orangeLevel` → ALERT
- If `waterLevel` >= `blueLevel` → WATCH

**Axis 2: Zero-Release Projection (Storage)**
Because we lack a verified outflow/release policy model, we project the worst-case future scenario assuming zero release.
The logic assumes the forecast is a daily average over the horizon.
- `projected_1d_storage` = `liveStorage` + (1 × `target_1d_prediction`)
- `projected_3d_storage` = `liveStorage` + (3 × `target_3d_prediction`)
- `projected_7d_storage` = `liveStorage` + (7 × `target_7d_prediction`)

If *any* projected storage exceeds the absolute physical capacity (`liveStorageAtFRL`), a risk is triggered.

## 5. Multi-Horizon Behavior
To assist operators with varying time sensitivities, projections trigger different severities based on lead time:
- A projected breach within **1 day** triggers **HIGH RISK** (immediate emergency).
- A projected breach within **3 days** triggers **ALERT** (short-term preparedness).
- A projected breach within **7 days** triggers **WATCH** (early planning).
The engine explicitly identifies the horizon that triggered the warning (e.g., `horizon_triggered: "3d"`).

## 6. Multi-Reservoir Behavior
The engine processes inputs immutably per reservoir. It accepts a reservoir's specific metadata dictionary, state dictionary, and forecast array, outputting a human-readable string suitable for a dashboard view.

## 7. Missing-Data Handling
The engine is engineered to fail safely. If `liveStorageAtFRL` or `liveStorage` is missing or null, the projection logic is bypassed without throwing an exception. If warning levels (`redLevel`, etc.) are missing, the state check is bypassed. If all checks are bypassed, the system defaults to "NORMAL" rather than hallucinating an invalid risk.

## 8. Safety Limitations
- **Zero-Release Assumption:** The projection is an absolute upper bound (worst-case scenario). It does not mean the dam *will* overtop, it means it *would* overtop if the operator took no action.
- **Volumetric Approximation:** Combining MCM storage with incoming volume is a first-order approximation. It ignores minor losses (evaporation, seepage).

## 9. Test Results
The unit tests (`src/management/test_risk_engine.py`) executed successfully, verifying:
- Normal conditions return safe status.
- State warnings are triggered precisely at thresholds.
- Projected volume breaches properly cascade up from 7d to 1d severity.
- Missing values do not crash the engine.

## 10. Example Output
```python
{
    'status': 'ALERT',
    'reason': '3-day zero-release projected storage (402.00) exceeds live capacity (400.0).',
    'horizon_triggered': '3d'
}
```

## 11. What the Engine Can and Cannot Claim
**CAN Claim:**
- It provides deterministic early-warning alerts for extreme inflow conditions based on physical limits.
- It operates strictly on verified telemetry and forecasting models.

**CANNOT Claim:**
- It is not an autonomous dam controller.
- It does not recommend exact release volumes, as downstream capacities are unknown.
- It does not predict exact future water levels, only worst-case upper bounds.

## What Should Be Done Next
With the backend logic established, the next logical step (Phase 14.2 / 14.3) is to build an interactive Operator Dashboard (e.g., using Streamlit) to visualize the telemetry, the V3 forecasts, and the Risk Engine status outputs simultaneously.
