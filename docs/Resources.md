# Project Resources & Literature Repository

This document serves as a centralized hub for all scholarly literature, datasets, reference implementations, and documentation related to the AI-Based Multi-Reservoir Water Management and Flood Prevention System.

## Research Papers

### Inflow & Flood Forecasting (Predictive Modeling)
*   *LSTM-Based Streamflow Prediction*: Reference studies on using Long Short-Term Memory (LSTM) and GRU networks for rainfall-runoff modeling.
*   *Spatiotemporal Forecasting*: Graph Neural Networks (GNNs) for river network routing forecasts.

### Multi-Reservoir Operation & Control (Reinforcement Learning)
*   *Deep Reinforcement Learning for Water Resources*: Policy gradient methods (PPO, DDPG, SAC) applied to reservoir release optimization.
*   *Multi-Agent Reinforcement Learning (MARL)*: Coordinated release policies for multi-reservoir cascades.

### Continual & Adaptive Learning
*   *Concept Drift in Hydrology*: Methods for handling seasonal variations and climate change shifts in predictive models without retraining from scratch.

---

## Datasets

### Hydrological & Meteorological Data
*   **USGS National Water Information System**: [USGS Water Data](https://waterdata.usgs.gov/nwis) - Real-time and historical streamflow data.
*   **NASA POWER**: [NASA POWER Portal](https://power.larc.nasa.gov/) - Solar, meteorology, and precipitation data.
*   **Global Runoff Data Centre (GRDC)**: River discharge data from around the globe.
*   **Local Watershed Datasets**: [Provide links or local paths to specific local reservoir and basin records here].

---

## GitHub Repositories

### Hydrological Modeling & RL Environments
*   **PySheds**: Simple, fast watershed delineation in Python.
*   **Gymnasium / Stable-Baselines3**: Standard RL environments and algorithms.
*   **Continual Learning Benchmarks**: Repositories containing implementations of EWC, GEM, or replay buffer strategies for non-stationary environments.

---

## Documentation

### Frameworks & Tools
*   [Streamlit Docs](https://docs.streamlit.io/) — Reference guide for constructing the operator dashboard.
*   [NetworkX Docs](https://networkx.org/documentation/stable/) — Guide for graph representation of reservoir networks.
*   [Plotly Python Graphing Library](https://plotly.com/python/) — For interactive charting and geographic visualization.

---

## Notes

### Critical Considerations
*   *Safety Constraints*: The system must operate under strict safety boundaries (e.g., maximum water level thresholds to prevent overtopping, minimum water level for irrigation/power generation).
*   *Data Quality*: Inflow data is often noisy, missing, or subject to sensor calibration errors. Quality assurance/quality control (QA/QC) pipelines must be built in `data/processed/`.
