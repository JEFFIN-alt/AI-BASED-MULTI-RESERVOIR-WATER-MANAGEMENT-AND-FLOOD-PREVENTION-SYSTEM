# Operator Decision Support Dashboard (Phase 14.3 MVP)

This directory contains the Streamlit dashboard for the AI-Based Multi-Reservoir Water Management System.

## Architecture

This dashboard acts as **Layer 4 (Operator UI)** of the system architecture.
It integrates:
- Real historic reservoir telemetry (`data/raw/reservoir/`)
- Frozen LSTM V3 inflow point-forecasts (`results/lstm_pytorch_v3_logtarget/`)
- The validated Point-Forecast Warning Engine (`src/management/risk_engine.py`)

## Running the Dashboard

Ensure you are in the project root directory and your Conda environment (or venv) is activated.

Run the following command:

```bash
streamlit run src/dashboard/app.py
```

## Scientific Constraints

- The dashboard displays point forecasts (1d/3d/7d), not cumulative weekly inflow.
- It displays historical test predictions; it is not hooked up to a live meteorological feed.
- It assesses risk based on static water-level thresholds and historically derived 95th-percentile extreme-inflow thresholds.
- Hardware telemetry (ESP32) is simulated as "Not Connected" in this MVP and will be integrated in Phase 14.4.
