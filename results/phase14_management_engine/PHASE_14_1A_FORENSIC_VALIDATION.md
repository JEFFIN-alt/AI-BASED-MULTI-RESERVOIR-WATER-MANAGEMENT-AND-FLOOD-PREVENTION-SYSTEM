# Phase 14.1A: Risk Engine Forensic Validation

## 1. V3 Target Definitions
An inspection of `src/data_quality/build_supervised_dataset.py` (specifically lines 259-282) reveals that the V3 targets are constructed by strictly selecting the reservoir inflow on the exact target date:
```python
target_date = current_date + pd.Timedelta(days=horizon)
matches = group[group["date"] == target_date]
target_rows[target_name] = matches.iloc[0]
```
**Conclusion**: `target_1d`, `target_3d`, and `target_7d` represent the **exact inflow occurring on day $t+1$, $t+3$, and $t+7$ respectively**. They are **NOT** daily averages over the horizon, nor are they cumulative volumes.

## 2. Physical Units & Dimensional Analysis
- `waterLevel`, `redLevel`, `orangeLevel`, `blueLevel`, `FRL`: Measured in meters (elevation).
- `liveStorage`, `liveStorageAtFRL`: Measured in volumetric units (Million Cubic Meters / MCM).
- `inflow_day_x`, `target_xd`: Measured in volumetric rate (MCM/day).

**Dimensional Analysis**:
The equation used in the Phase 14.1 Risk Engine is:
`projected_3d_storage = liveStorage + (3 * target_3d)`

Dimensionally, this is:
$Volume + (Time \times \frac{Volume}{Time}) = Volume$
This is dimensionally valid. **However, it is mathematically and physically invalid.**

## 3. The Horizon Multiplication Error (Why it is wrong)
Because `target_7d` is the point-forecast of inflow *on day 7*, multiplying it by 7 assumes that the inflow on day 7 was constant for the entire preceding week. 
This is a catastrophic numerical integration error (equivalent to an Euler integration step of 7 days using the derivative at the *end* of the interval). If a sudden storm hits on day 7, the engine multiplies that storm by 7, producing a massive hallucinated cumulative volume. If a storm hits on day 4 but clears by day 7, the engine multiplies a low inflow by 7, entirely missing the flood volume.

## 4. Threshold Semantics
- The logic comparing `waterLevel` to `blueLevel`, `orangeLevel`, and `redLevel` is correct. These are water-level elevations, and comparing them directly to the current telemetry `waterLevel` is physically sound.
- The logic comparing `projected_storage` to `liveStorageAtFRL` is physically sound in principle (both are volumes).

## 5. Real-Data Examples
If Idukki has `liveStorage` = 300 MCM, and the V3 model predicts:
- `target_1d` = 10 MCM/day
- `target_3d` = 20 MCM/day
- `target_7d` = 100 MCM/day (a peak storm arriving exactly on day 7)

The Phase 14.1 Risk Engine calculates:
- 1-day projection = $300 + (1 \times 10) = 310$ MCM
- 3-day projection = $300 + (3 \times 20) = 360$ MCM
- 7-day projection = $300 + (7 \times 100) = 1000$ MCM

The 7-day projection hallucinates 700 MCM of incoming water (exceeding the FRL of 401.43 MCM by a massive margin), triggering a false HIGH RISK, simply because it assumes the day-7 storm existed for the entire week.

## 6. Problems Found
1. **Invalid Cumulative Projection**: Multiplying point-inflow forecasts by the horizon length creates mathematically invalid cumulative volume projections.
2. **Missing Intermediate Forecasts**: The V3 model does not predict days 2, 4, 5, and 6. Therefore, calculating a true 7-day cumulative inflow is impossible without interpolation.

## 7. Final Classification
**CLASSIFICATION: D (UNSAFE / INVALID)**

## 8. Recommended Correction
The risk engine must be corrected before proceeding to the dashboard. The zero-release projection is scientifically defensible **ONLY IF** it uses valid cumulative volumes. 

**Correction**:
Change the projection logic to use a simple linear interpolation (Trapezoidal rule) across the available point forecasts to estimate the cumulative volume:
- `volume_1d` = `target_1d`
- `volume_3d` = `target_1d` + 2 * `(target_1d + target_3d)/2` (interpolating day 2)
- `volume_7d` = `volume_3d` + 4 * `(target_3d + target_7d)/2` (interpolating days 4, 5, 6)

This mathematically grounds the volume estimation in the available predicted data points, eliminating the severe over/under-estimation caused by blind scalar multiplication.

## 9. Can We Proceed to the Dashboard?
**NO.** The engine in `src/management/risk_engine.py` must be rewritten to implement the trapezoidal volume estimation before any UI is built, otherwise the dashboard will display dangerously inaccurate risk alerts to operators.
