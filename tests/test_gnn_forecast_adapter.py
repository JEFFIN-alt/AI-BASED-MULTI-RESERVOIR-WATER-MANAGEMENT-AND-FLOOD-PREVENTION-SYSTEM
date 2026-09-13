"""
Test Suite: GNN Forecast Adapter
=================================

Tests:
  1. Adapter initialization
  2. Forecast snapshot creation
  3. Combined forecast dict structure (LSTM + GNN separate)
  4. Risk envelope computation
  5. Control policy: lstm_primary
  6. Control policy: gnn_primary (fallback to LSTM)
  7. Status validation (NaN, Inf, None)
  8. GNN forecasts never overwrite LSTM forecasts
"""
import sys
import math
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.network_env.gnn_forecast_adapter import (
    GNNForecastAdapter,
    GNNReservoirForecast,
    GNNNetworkForecastSnapshot,
    GNNForecastStatus,
)


@pytest.fixture
def mapping():
    return {
        "Reservoir_A": "Anayirankal",
        "Reservoir_B": "Idukki",
        "Reservoir_C": "Idamalayar",
    }


@pytest.fixture
def adapter(mapping):
    return GNNForecastAdapter(mapping)


@pytest.fixture
def sample_gnn_result():
    return {
        "Anayirankal": {
            "target_1d": 5.0, "target_3d": 6.0, "target_7d": 7.0, "available": True,
        },
        "Idukki": {
            "target_1d": 10.0, "target_3d": 12.0, "target_7d": 14.0, "available": True,
        },
        "Idamalayar": {
            "target_1d": 3.0, "target_3d": 4.0, "target_7d": 5.0, "available": True,
        },
    }


@pytest.fixture
def sample_lstm_forecasts():
    return {
        "Reservoir_A": {"forecast_1d": 4.0, "forecast_3d": 5.5, "forecast_7d": 6.0},
        "Reservoir_B": {"forecast_1d": 8.0, "forecast_3d": 11.0, "forecast_7d": 13.0},
        "Reservoir_C": {"forecast_1d": 2.5, "forecast_3d": 3.5, "forecast_7d": 4.5},
    }


# ── 1. Adapter initialization ──────────────────────────────────────────────

def test_adapter_init(adapter, mapping):
    assert adapter._mapping == mapping
    assert adapter.latest_snapshot is None


# ── 2. Forecast snapshot creation ───────────────────────────────────────────

def test_update_from_inference(adapter, sample_gnn_result):
    snapshot = adapter.update_from_inference(sample_gnn_result, inference_time_ms=5.0, gate_value=0.015)
    
    assert isinstance(snapshot, GNNNetworkForecastSnapshot)
    assert snapshot.inference_time_ms == 5.0
    assert snapshot.gate_value == 0.015
    assert len(snapshot.forecasts) == 3
    
    f = snapshot.get("Anayirankal")
    assert f is not None
    assert f.target_1d == 5.0
    assert f.status_1d == GNNForecastStatus.AVAILABLE


# ── 3. Combined forecast dict ──────────────────────────────────────────────

def test_combined_forecast_keeps_separate(adapter, sample_gnn_result, sample_lstm_forecasts):
    adapter.update_from_inference(sample_gnn_result)
    combined = adapter.get_combined_forecast_dict(sample_lstm_forecasts)
    
    # LSTM and GNN must be in separate sub-keys
    for name in sample_lstm_forecasts:
        assert "lstm" in combined[name]
        assert "gnn" in combined[name]
        
        # LSTM values are preserved
        assert combined[name]["lstm"]["1d"] == sample_lstm_forecasts[name]["forecast_1d"]
        assert combined[name]["lstm"]["3d"] == sample_lstm_forecasts[name]["forecast_3d"]
        assert combined[name]["lstm"]["7d"] == sample_lstm_forecasts[name]["forecast_7d"]


# ── 4. Risk envelope ───────────────────────────────────────────────────────

def test_risk_envelope(adapter, sample_gnn_result, sample_lstm_forecasts):
    adapter.update_from_inference(sample_gnn_result)
    envelope = adapter.get_risk_envelope_forecasts(sample_lstm_forecasts)
    
    # Reservoir_A: LSTM 1d=4.0, GNN 1d=5.0 → max=5.0
    assert envelope["Reservoir_A"]["forecast_1d"] == 5.0
    # Reservoir_B: LSTM 1d=8.0, GNN 1d=10.0 → max=10.0
    assert envelope["Reservoir_B"]["forecast_1d"] == 10.0
    # Reservoir_C: LSTM 1d=2.5, GNN 1d=3.0 → max=3.0
    assert envelope["Reservoir_C"]["forecast_1d"] == 3.0


# ── 5. Control policy: lstm_primary ─────────────────────────────────────────

def test_control_lstm_primary(adapter, sample_gnn_result, sample_lstm_forecasts):
    adapter.update_from_inference(sample_gnn_result)
    ctrl = adapter.get_control_forecasts(sample_lstm_forecasts, policy="lstm_primary")
    
    # Must use LSTM values exactly
    for name in sample_lstm_forecasts:
        assert ctrl[name]["forecast_1d"] == sample_lstm_forecasts[name]["forecast_1d"]
        assert ctrl[name]["source"] == "LSTM_V3"


# ── 6. Control policy: gnn_primary with fallback ───────────────────────────

def test_control_gnn_primary(adapter, sample_gnn_result, sample_lstm_forecasts):
    adapter.update_from_inference(sample_gnn_result)
    ctrl = adapter.get_control_forecasts(sample_lstm_forecasts, policy="gnn_primary")
    
    # GNN values should be used where available
    assert ctrl["Reservoir_A"]["forecast_1d"] == 5.0
    assert "GCN_LSTM" in ctrl["Reservoir_A"]["source"]


# ── 7. Status validation ───────────────────────────────────────────────────

def test_nan_marked_invalid(adapter):
    gnn_result = {
        "Anayirankal": {
            "target_1d": float('nan'), "target_3d": 6.0, "target_7d": 7.0,
            "available": True,
        },
    }
    snapshot = adapter.update_from_inference(gnn_result)
    f = snapshot.get("Anayirankal")
    assert f.status_1d == GNNForecastStatus.INVALID
    assert f.status_3d == GNNForecastStatus.AVAILABLE


def test_inf_marked_invalid(adapter):
    gnn_result = {
        "Anayirankal": {
            "target_1d": float('inf'), "target_3d": 6.0, "target_7d": 7.0,
            "available": True,
        },
    }
    snapshot = adapter.update_from_inference(gnn_result)
    f = snapshot.get("Anayirankal")
    assert f.status_1d == GNNForecastStatus.INVALID


def test_missing_node(adapter):
    gnn_result = {
        "Anayirankal": {
            "target_1d": 5.0, "target_3d": 6.0, "target_7d": 7.0,
            "available": False,
        },
    }
    snapshot = adapter.update_from_inference(gnn_result)
    f = snapshot.get("Anayirankal")
    assert f.status_1d == GNNForecastStatus.NODE_MISSING


# ── 8. GNN never overwrites LSTM ───────────────────────────────────────────

def test_lstm_values_unchanged(adapter, sample_gnn_result, sample_lstm_forecasts):
    """Updating GNN forecasts must never modify the original LSTM dict."""
    import copy
    lstm_copy = copy.deepcopy(sample_lstm_forecasts)
    
    adapter.update_from_inference(sample_gnn_result)
    adapter.get_combined_forecast_dict(sample_lstm_forecasts)
    adapter.get_risk_envelope_forecasts(sample_lstm_forecasts)
    
    assert sample_lstm_forecasts == lstm_copy


# ── 9. Invalid policy raises ───────────────────────────────────────────────

def test_invalid_policy_raises(adapter, sample_lstm_forecasts):
    with pytest.raises(ValueError, match="Unknown control policy"):
        adapter.get_control_forecasts(sample_lstm_forecasts, policy="invalid")
