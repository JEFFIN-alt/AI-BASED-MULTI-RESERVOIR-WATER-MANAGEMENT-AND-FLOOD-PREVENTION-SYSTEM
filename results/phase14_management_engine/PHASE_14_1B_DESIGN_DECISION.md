# Phase 14.1B: Safe Correction Design Decision

## 1. Mathematical and Physical Analysis
The V3 model outputs point forecasts of daily inflow volume (MCM/day) at $t+1$, $t+3$, and $t+7$.
To construct a cumulative volume ($V$), the exact integral required is:
$$ V = \int_{0}^{7} \text{Inflow}(t) dt $$

Because we only have sparse discrete points ($t=1, 3, 7$), numerical integration (e.g., Trapezoidal rule) would require linear interpolation across multi-day gaps (days 4, 5, 6). 
In hydrology, storm events are highly non-linear and bursty. A flash flood event could easily occur entirely on day 5, peaking and subsiding between the day 3 and day 7 forecasts. Any linear interpolation across this gap is fundamentally blind to such events, leading to either severe underestimation (missing the day 5 storm) or overestimation (stretching a day 7 peak backwards across the entire week).

Furthermore, projecting future storage requires the equation:
$S_{future} = S_{current} + V_{inflow} - V_{outflow}$
Assuming zero release ($V_{outflow} = 0$) for an entire 7-day period is an extreme and physically unrealistic assumption for an active reservoir, producing bounds so wide they become operationally useless.

## 2. Design Options Evaluated

### Design A: POINT-FORECAST WARNING ENGINE
- Evaluates the verified current `waterLevel` against the verified static thresholds (`blueLevel`, `orangeLevel`, `redLevel`).
- Evaluates the V3 point forecasts (F1, F3, F7) directly as inflow-rate anomalies.
- **No cumulative storage projection is calculated.**

### Design B: ESTIMATED ZERO-RELEASE TRAJECTORY ENGINE
- Explicitly interpolates daily inflow for the un-forecasted intermediate days.
- Integrates the trajectory over 7 days.
- Projects worst-case storage using a zero-release assumption.

## 3. Comparison
| Criterion | Design A (Point-Forecast) | Design B (Interpolated Zero-Release) |
| :--- | :--- | :--- |
| **Scientific Defensibility** | **High.** Makes no hidden assumptions. Analyzes exactly what the model predicts. | **Low.** Relies on physically invalid hydrological interpolation and unrealistic zero-release assumptions. |
| **Interpretability** | **High.** "High inflow expected on Day 7." | **Low.** "Projected storage assumes no gates are opened for 7 days and interpolation holds." |
| **False Alarm Risk** | **Low.** Direct mapping of model confidence. | **High.** Interpolation stretches isolated peaks across multiple days. |
| **False Reassurance Risk**| **Low.** | **High.** Misses storms that fall entirely within the unforecasted gaps. |
| **Implementation** | **Simple.** Clean, deterministic threshold logic. | **Complex.** Requires numerical integration and extensive caveat documentation. |

## 4. Recommendation: DESIGN A
**Design A (Point-Forecast Warning Engine)** is strictly recommended. 
It prioritizes scientific transparency, physical validity, and safety. Attempting to force a volumetric projection out of three sparse data points provides a dangerous illusion of precision. An operator decision-support system must be honest about what it knows and what it does not know.

## 5. Next Implementation Steps (Phase 14.1C)
1. **Remove** the `projected_storage` logic entirely from `risk_engine.py`.
2. **Implement** an alternative inflow-risk metric for the V3 forecasts (e.g., comparing the predicted F1/F3/F7 inflow against historical 95th/99th percentile inflow values to trigger high-inflow warnings independently of current storage).
3. **Retain** the valid current-state `waterLevel` vs warning threshold logic.
