# Phase 14.1C: Safe Point-Forecast Warning Engine

## 1. What Changed from Phase 14.1
The invalid Phase 14.1 implementation attempted to project future storage volumes by treating sparse V3 point-forecasts (`target_1d`, `target_3d`, `target_7d`) as multi-day cumulative constants. This resulted in catastrophic mathematical errors where a high peak arriving on day 7 would be incorrectly multiplied by 7, hallucinatory massive flooding, while peaks occurring in the un-forecasted gaps (days 2, 4, 5, 6) were missed entirely.

The new implementation (Design A) completely **removes** all future storage projections. Instead, it operates a dual-channel warning system:
- **Channel 1**: Evaluates the *current* verified water level against static threshold limits.
- **Channel 2**: Evaluates the *predicted* inflow rates directly against historical high-inflow thresholds.

## 2. Threshold Methodologies
### Water-Level Thresholds
- **Source**: `data/raw/reservoir/Kerala-Dam-Water-Levels/live.json` and `irrigation_live.json`.
- **Logic**: If `waterLevel` > `redLevel` -> HIGH RISK. If > `orangeLevel` -> ALERT. If > `blueLevel` -> WATCH.

### High-Inflow Thresholds
- **Source**: Computed dynamically from `data/processed/forecasting_supervised_7day.csv` restricting strictly to `date < 2025-01-01` (Training Data Only). 
- **Methodology**: For each reservoir, the 95th percentile of observed inflow was calculated from the training split. This prevents global thresholding (which fails since Idukki's 95th percentile is ~1581 MCM/day while Kundala's is ~7.6 MCM/day).
- **File**: Results are cached in `data/processed/historical_inflow_thresholds.json`.

## 3. Multi-Horizon Behavior
The engine evaluates the 1-day, 3-day, and 7-day point forecasts independently against the 95th percentile threshold. If *any* horizon exceeds the threshold, the `inflow_forecast_status` is elevated to `ALERT`. The engine clearly identifies which horizons triggered the alert (e.g., "at horizons: 7d"). It does not merge the horizons into a pseudo-cumulative score.

## 4. Combined Status Logic
The engine produces an `overall_status` by taking the highest severity between the water-level status and the inflow forecast status. The output explicitly combines the human-readable reasons from both channels, ensuring transparency.

Example:
```json
{
  "overall_status": "ALERT",
  "water_level_status": "NORMAL",
  "inflow_forecast_status": "ALERT",
  "reason": "Current water level is within safe thresholds. | Forecast inflow exceeds historical 95th percentile (1581.04) at horizons: 7d (1600.00)."
}
```

## 5. Missing-Data Handling
- If water-level thresholds (`redLevel`, etc.) are missing, the water-level status evaluates to `NORMAL` (assuming no evidence of breach).
- If the telemetry itself is missing (no `waterLevel`), the status evaluates to `UNKNOWN`.
- If the historical 95th percentile inflow threshold is missing for a reservoir, the inflow status evaluates to `INSUFFICIENT_DATA`.
- Negative forecasts trigger a localized warning reason but do not artificially inflate the risk level.

## 6. Real-Data Dry-Run Validation (Idukki)
Using actual Idukki telemetry and historical thresholds (95th percentile = ~1581 MCM/day):

**Scenario 1: Normal Conditions**
- Inputs: WL=709.09 (Safe), Forecasts=[50, 100, 500] (Safe)
- Output: `NORMAL` (Reason: Current water level is within safe thresholds. | Forecast inflow is within historical norms.)

**Scenario 2: Impending Storm (Day 7)**
- Inputs: WL=709.09 (Safe), Forecasts=[50, 100, 1600] (High)
- Output: `ALERT` (Reason: Current water level is within safe thresholds. | Forecast inflow exceeds historical 95th percentile (1581.04) at horizons: 7d (1600.00).)

**Scenario 3: Dam Currently Full**
- Inputs: WL=724.0 (Exceeds Red Level of 723.7), Forecasts=[50, 100, 500] (Safe)
- Output: `HIGH RISK` (Reason: Current water level (724.0) exceeds RED warning level (723.7). | Forecast inflow is within historical norms.)

## 7. What the System Can and Cannot Claim
- **CAN CLAIM**: Provides early-warning alerts for current static threshold breaches and predicts statistically extreme incoming hydrological volume based on validated historical norms.
- **CANNOT CLAIM**: Does NOT predict overtopping dates. Does NOT predict exact future reservoir levels. Does NOT provide automated dam release instructions or guarantee downstream safety.
