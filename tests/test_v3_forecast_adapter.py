"""
Phase 15.2 — V3 Forecast Adapter Tests

Tests 1-16 plus safety/integrity checks as specified in Phase 15.2.
"""
import sys
import os
import hashlib
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.v3_forecast_adapter import (
    V3ForecastAdapter, ForecastStatus, ReservoirForecast, NetworkForecastSnapshot
)
from src.network_env.reservoir_network import ReservoirNetwork

# Project root for adapter
PROJECT_ROOT = str(_PROJECT_ROOT)

V3_ARTIFACT = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv"
V3_MODEL_DIR = _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget"


def _hash_dir(dirpath: Path) -> dict:
    """Compute SHA256 hash of every file in a directory."""
    hashes = {}
    if dirpath.exists():
        for f in sorted(dirpath.rglob("*")):
            if f.is_file():
                hashes[str(f.relative_to(dirpath))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# TEST 1 — Load real V3 prediction record
# ---------------------------------------------------------------------------
def test_01_load_v3_predictions():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    assert len(adapter.available_dates) > 0, "No dates loaded"
    assert len(adapter.available_reservoirs) > 0, "No reservoirs loaded"
    print(f"TEST 1 PASSED — Loaded {len(adapter.available_dates)} dates, "
          f"{len(adapter.available_reservoirs)} reservoirs from V3 artifact")


# ---------------------------------------------------------------------------
# TEST 2 — Correctly identify reservoir
# ---------------------------------------------------------------------------
def test_02_identify_reservoir():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.v3_reservoir_name == "Anayirankal", \
        f"Expected Anayirankal, got {fc.v3_reservoir_name}"
    fc_d = adapter.get_forecast("Reservoir_D", "2025-01-01")
    assert fc_d.v3_reservoir_name == "Idukki", \
        f"Expected Idukki, got {fc_d.v3_reservoir_name}"
    print(f"TEST 2 PASSED — Reservoir_A → Anayirankal, Reservoir_D → Idukki")


# ---------------------------------------------------------------------------
# TEST 3 — Correctly identify prediction date
# ---------------------------------------------------------------------------
def test_03_identify_date():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.forecast_date == "2025-01-01"
    fc2 = adapter.get_forecast("Reservoir_A", "2025-01-05")
    assert fc2.forecast_date == "2025-01-05"
    print(f"TEST 3 PASSED — Forecast date correctly identified")


# ---------------------------------------------------------------------------
# TEST 4 — Correctly expose target_1d
# ---------------------------------------------------------------------------
def test_04_target_1d():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.status_1d == ForecastStatus.AVAILABLE
    assert fc.target_1d is not None
    assert isinstance(fc.target_1d, float)
    assert fc.target_1d >= 0
    print(f"TEST 4 PASSED — target_1d = {fc.target_1d:.4f} MCM/day (Reservoir_A, 2025-01-01)")


# ---------------------------------------------------------------------------
# TEST 5 — Correctly expose target_3d
# ---------------------------------------------------------------------------
def test_05_target_3d():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.status_3d == ForecastStatus.AVAILABLE
    assert fc.target_3d is not None
    assert isinstance(fc.target_3d, float)
    print(f"TEST 5 PASSED — target_3d = {fc.target_3d:.4f} MCM/day")


# ---------------------------------------------------------------------------
# TEST 6 — Correctly expose target_7d
# ---------------------------------------------------------------------------
def test_06_target_7d():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.status_7d == ForecastStatus.AVAILABLE
    assert fc.target_7d is not None
    assert isinstance(fc.target_7d, float)
    print(f"TEST 6 PASSED — target_7d = {fc.target_7d:.4f} MCM/day")


# ---------------------------------------------------------------------------
# TEST 7 — target_3d is point forecast, NOT cumulative
# ---------------------------------------------------------------------------
def test_07_target_3d_is_point():
    """
    Verify target_3d is treated as a point forecast for day+3,
    not as (target_1d × 3) or any cumulative interpretation.
    """
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    # A cumulative interpretation would give target_3d ≈ target_1d * 3
    # A point forecast gives independent values
    # We verify the adapter does NOT scale by horizon
    assert fc.target_3d != fc.target_1d * 3 or fc.target_1d == 0, \
        "target_3d appears to be target_1d * 3 — may be cumulative"
    # Also verify the raw value matches the CSV directly
    import pandas as pd
    df = pd.read_csv(str(V3_ARTIFACT))
    row = df[(df["date"] == "2025-01-01") & (df["reservoir"] == "Anayirankal")].iloc[0]
    assert abs(fc.target_3d - row["target_3d_prediction"]) < 1e-10, \
        "Adapter modified the raw prediction value"
    print(f"TEST 7 PASSED — target_3d is raw point forecast: {fc.target_3d:.4f} "
          f"(verified against CSV: {row['target_3d_prediction']:.4f})")


# ---------------------------------------------------------------------------
# TEST 8 — target_7d is point forecast, NOT cumulative
# ---------------------------------------------------------------------------
def test_08_target_7d_is_point():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_D", "2025-01-01")
    import pandas as pd
    df = pd.read_csv(str(V3_ARTIFACT))
    row = df[(df["date"] == "2025-01-01") & (df["reservoir"] == "Idukki")].iloc[0]
    assert abs(fc.target_7d - row["target_7d_prediction"]) < 1e-10, \
        "Adapter modified the raw prediction value"
    print(f"TEST 8 PASSED — target_7d is raw point forecast: {fc.target_7d:.4f} "
          f"(verified against CSV: {row['target_7d_prediction']:.4f})")


# ---------------------------------------------------------------------------
# TEST 9 — Missing forecast fails safely
# ---------------------------------------------------------------------------
def test_09_missing_forecast():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    # Query a date that doesn't exist
    fc = adapter.get_forecast("Reservoir_A", "2099-01-01")
    assert fc.status_1d == ForecastStatus.UNAVAILABLE
    assert fc.status_3d == ForecastStatus.UNAVAILABLE
    assert fc.status_7d == ForecastStatus.UNAVAILABLE
    assert fc.target_1d is None
    assert not fc.is_available("1d")
    print(f"TEST 9 PASSED — Missing forecast returns UNAVAILABLE status")


# ---------------------------------------------------------------------------
# TEST 10 — NaN forecast fails safely
# ---------------------------------------------------------------------------
def test_10_nan_handling():
    """Test that NaN values are detected as INVALID."""
    import math
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    # Create a mock forecast with NaN
    fc = ReservoirForecast(
        reservoir_id="test",
        v3_reservoir_name="test",
        forecast_date="2025-01-01",
        target_1d=float('nan'),
        status_1d=adapter._validate_value(float('nan')),
    )
    assert fc.status_1d == ForecastStatus.INVALID
    print(f"TEST 10 PASSED — NaN detected as FORECAST_INVALID")


# ---------------------------------------------------------------------------
# TEST 11 — Unknown reservoir fails safely
# ---------------------------------------------------------------------------
def test_11_unknown_reservoir():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("NonexistentReservoir", "2025-01-01")
    assert fc.v3_reservoir_name == "UNKNOWN"
    assert fc.status_1d == ForecastStatus.UNAVAILABLE
    assert not fc.is_available("1d")
    print(f"TEST 11 PASSED — Unknown reservoir returns UNAVAILABLE")


# ---------------------------------------------------------------------------
# TEST 12 — Date mismatch fails safely
# ---------------------------------------------------------------------------
def test_12_date_mismatch():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    # Valid reservoir but non-existent date
    fc = adapter.get_forecast("Reservoir_A", "1900-01-01")
    assert fc.status_1d == ForecastStatus.UNAVAILABLE
    assert fc.target_1d is None
    print(f"TEST 12 PASSED — Date mismatch returns UNAVAILABLE")


# ---------------------------------------------------------------------------
# TEST 13 — Forecast provenance is MODEL_PREDICTION
# ---------------------------------------------------------------------------
def test_13_provenance():
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    fc = adapter.get_forecast("Reservoir_A", "2025-01-01")
    assert fc.provenance == "MODEL_PREDICTION"
    assert fc.model == "LSTM_V3"
    info = adapter.get_provenance_info()
    assert info["provenance_level"] == "MODEL_PREDICTION"
    assert "POINT FORECAST" in info["forecast_semantics"]["target_1d"]
    print(f"TEST 13 PASSED — Provenance: {fc.provenance}, model: {fc.model}")


# ---------------------------------------------------------------------------
# TEST 14 — Network state NOT overwritten by forecast
# ---------------------------------------------------------------------------
def test_14_forecast_does_not_overwrite_state():
    """
    Prove that loading forecasts does not modify the network state.
    The forecast metadata exists separately from the network's actual inflow.
    """
    config_path = str(_PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml")
    net = ReservoirNetwork(config_path=config_path)

    # Record initial state
    initial_storages = {nid: node.state.storage for nid, node in net.nodes.items()}

    # Load forecasts
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    snapshot = adapter.get_network_snapshot("2025-01-01")

    # Verify network state unchanged
    for nid, node in net.nodes.items():
        assert node.state.storage == initial_storages[nid], \
            f"{nid} storage changed after loading forecast!"
        assert node.state.inflow_local == 0.0, \
            f"{nid} inflow_local was modified by forecast loading!"

    # The forecast is metadata, not an action
    fc_a = snapshot.get("Reservoir_A")
    assert fc_a.target_1d is not None  # forecast exists
    assert net.nodes["Reservoir_A"].state.inflow_local == 0.0  # but inflow unchanged

    print(f"TEST 14 PASSED — Network state unchanged after loading forecasts")


# ---------------------------------------------------------------------------
# TEST 15 — Deterministic adapter output
# ---------------------------------------------------------------------------
def test_15_deterministic():
    adapter1 = V3ForecastAdapter(PROJECT_ROOT)
    adapter2 = V3ForecastAdapter(PROJECT_ROOT)

    fc1 = adapter1.get_forecast("Reservoir_C", "2025-01-03")
    fc2 = adapter2.get_forecast("Reservoir_C", "2025-01-03")

    assert fc1.target_1d == fc2.target_1d
    assert fc1.target_3d == fc2.target_3d
    assert fc1.target_7d == fc2.target_7d
    print(f"TEST 15 PASSED — Deterministic: two adapter instances produce identical output")


# ---------------------------------------------------------------------------
# TEST 16 — V3 artifact integrity
# ---------------------------------------------------------------------------
def test_16_v3_integrity():
    """
    Prove that loading forecasts does NOT modify V3 artifacts.
    Hash all files before and after adapter creation.
    """
    # Hash before
    artifact_hash_before = hashlib.sha256(V3_ARTIFACT.read_bytes()).hexdigest()
    model_hashes_before = _hash_dir(V3_MODEL_DIR)

    # Create adapter and exercise it
    adapter = V3ForecastAdapter(PROJECT_ROOT)
    for date in adapter.available_dates[:10]:
        _ = adapter.get_network_snapshot(date)

    # Hash after
    artifact_hash_after = hashlib.sha256(V3_ARTIFACT.read_bytes()).hexdigest()
    model_hashes_after = _hash_dir(V3_MODEL_DIR)

    assert artifact_hash_before == artifact_hash_after, \
        "V3 prediction artifact was modified!"
    assert model_hashes_before == model_hashes_after, \
        "V3 model files were modified!"

    print(f"TEST 16 PASSED — V3 artifacts verified byte-for-byte unchanged")
    print(f"  Prediction SHA256: {artifact_hash_before[:16]}...")
    print(f"  Model files checked: {len(model_hashes_before)}")


# ---------------------------------------------------------------------------
# BONUS — Network snapshot demo
# ---------------------------------------------------------------------------
def test_bonus_network_snapshot():
    """
    Demonstrate a complete network forecast snapshot.
    This is the integration demo showing forecast metadata alongside
    network state WITHOUT making gate decisions.
    """
    config_path = str(_PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml")
    net = ReservoirNetwork(config_path=config_path)
    adapter = V3ForecastAdapter(PROJECT_ROOT)

    demo_date = "2025-01-01"
    snapshot = adapter.get_network_snapshot(demo_date)

    print(f"\n{'='*70}")
    print(f"INTEGRATION DEMO — Simulation date: {demo_date}")
    print(f"{'='*70}")

    for nid in net.processing_order:
        node = net.nodes[nid]
        fc = snapshot.get(nid)

        print(f"\n  {nid} (V3 source: {fc.v3_reservoir_name})")
        print(f"    NETWORK STATE (from network_env):")
        print(f"      storage   = {node.state.storage:.2f} MCM ({node.storage_fraction*100:.1f}%)")
        print(f"      inflow    = {node.state.inflow_local:.2f} MCM/day")
        print(f"      gate      = {node.state.gate_position:.2f}")
        print(f"    V3 FORECAST METADATA (MODEL_PREDICTION, not applied):")
        print(f"      1-day     = {fc.target_1d:.4f} MCM/day  [{fc.status_1d.value}]")
        print(f"      3-day     = {fc.target_3d:.4f} MCM/day  [{fc.status_3d.value}]")
        print(f"      7-day     = {fc.target_7d:.4f} MCM/day  [{fc.status_7d.value}]")
        print(f"    PROVENANCE  = {fc.provenance}")

    print(f"\n  NOTE: Forecasts are METADATA only. No gate decisions made.")
    print(f"  The actual network inflow remains SEPARATE from the forecast.")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 15.2 — V3 Forecast Adapter Tests")
    print("=" * 60)

    test_01_load_v3_predictions()
    test_02_identify_reservoir()
    test_03_identify_date()
    test_04_target_1d()
    test_05_target_3d()
    test_06_target_7d()
    test_07_target_3d_is_point()
    test_08_target_7d_is_point()
    test_09_missing_forecast()
    test_10_nan_handling()
    test_11_unknown_reservoir()
    test_12_date_mismatch()
    test_13_provenance()
    test_14_forecast_does_not_overwrite_state()
    test_15_deterministic()
    test_16_v3_integrity()
    test_bonus_network_snapshot()

    print("\n" + "=" * 60)
    print("ALL 16 TESTS + INTEGRATION DEMO PASSED.")
    print("=" * 60)
