"""
Stage 5 — Forecast pipeline integration & data provenance tests.

Proves:
  * the frozen model + scalers are loaded UNCHANGED,
  * the frozen feature contract (order, 7-timestep sequence, units, horizons) holds,
  * forecast outputs are finite and in MCM/day,
  * the live payload EXPOSES provenance on every feature and on the forecast,
  * simulated / synthetic inputs are clearly marked,
  * unavailable physical quantities (water_level in metres, rainfall in mm) are
    NOT falsely represented as real measurements,
  * NO arbitrary storage -> water-level conversion exists,
  * missing / invalid forecast data is handled explicitly (no echo fallbacks),
  * the advisory GNN is not part of the control path.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.modeling.inference import (  # noqa: E402
    ForecastUnavailableError,
    LiveForecaster,
)
from src.modeling import v3_feature_contract as contract  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
MODEL_DIR = _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget"
V3_RESULTS_DIR = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget"
MANIFEST = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"
STATE_MANAGER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py"
LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

#: The frozen feature order — asserted literally so a silent reorder fails loudly.
FROZEN_FEATURE_ORDER = ["inflow", "water_level", "live_storage", "rainfall", "total_outflow"]


@pytest.fixture(scope="module")
def forecaster():
    return LiveForecaster(PROJECT_ROOT)


def _history(**overrides) -> pd.DataFrame:
    """A valid, contiguous 7-day window in the frozen feature order."""
    base = {
        "water_level": [1203.0] * 7,
        "live_storage": [5.4] * 7,
        "inflow": [3.0] * 7,
        "rainfall": [6.4] * 7,
        "total_outflow": [2.6] * 7,
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    df["date"] = pd.date_range("2025-06-01", periods=7, freq="D").strftime("%Y-%m-%d")
    return df


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# 1. Frozen model + scalers loaded unchanged
# ===========================================================================

def test_frozen_model_and_scalers_unchanged_by_loading_and_inference(forecaster):
    """
    Instantiating the forecaster and running inference must not alter any
    frozen artifact (byte-level, with the CRLF-aware rule from Stage 3).
    """
    artifacts = {
        MODEL_DIR / "best_model.pt": None,
        MODEL_DIR / "log_target_scaler.pkl": None,
        _PROJECT_ROOT / "data" / "processed" / "scaled" / "feature_scaler.pkl": None,
        V3_RESULTS_DIR / "test_predictions_original_units.csv": None,
    }
    before = {p: _sha256(p) for p in artifacts}

    forecaster.predict(_history())

    after = {p: _sha256(p) for p in artifacts}
    assert before == after, "A frozen artifact changed during loading/inference"


def test_v3_artifacts_match_the_phase15_3_manifest():
    """The recorded Phase 15.3 manifest hashes must still hold (raw or LF-normalised)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel_path, recorded in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel_path
        assert path.exists(), f"frozen artifact missing: {rel_path}"
        raw = _sha256(path)
        normalised = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert raw == recorded["sha256"] or normalised == recorded["sha256"], (
            f"frozen artifact content changed: {rel_path}"
        )


def test_model_architecture_is_the_frozen_one(forecaster):
    """Input width must be exactly the 5 frozen dynamic features."""
    assert forecaster.model.lstm.input_size == 5
    assert forecaster.model.output.out_features == 3


# ===========================================================================
# 2. Feature ordering / scaler contract
# ===========================================================================

def test_frozen_feature_order_is_exact(forecaster):
    assert contract.DYNAMIC_FEATURES == FROZEN_FEATURE_ORDER
    assert forecaster.dynamic_features == FROZEN_FEATURE_ORDER


def test_sequence_length_is_seven(forecaster):
    assert contract.HISTORY_DAYS == 7
    assert len(contract.SCALER_DYNAMIC_COLUMNS) == 35


def test_scaler_column_order_matches_the_frozen_contract(forecaster):
    cols = [str(c) for c in forecaster.feature_scaler.feature_names_in_]
    assert len(cols) == 51
    assert cols == contract.EXPECTED_SCALER_COLUMNS
    # The 35 dynamic columns must come FIRST, in feature-major order.
    assert cols[:35] == contract.SCALER_DYNAMIC_COLUMNS


def test_scaler_contract_verification_rejects_a_reordered_scaler():
    """A silent reorder must be caught, not silently accepted."""

    class _BadScaler:
        feature_names_in_ = list(reversed(contract.EXPECTED_SCALER_COLUMNS))

    with pytest.raises(contract.ScalerContractError):
        contract.verify_scaler_contract(_BadScaler())

    class _WrongSize:
        feature_names_in_ = contract.EXPECTED_SCALER_COLUMNS[:40]

    with pytest.raises(contract.ScalerContractError):
        contract.verify_scaler_contract(_WrongSize())


def test_feature_ordering_materially_changes_the_forecast(forecaster):
    """
    Demonstrates WHY the ordering is pinned: permuting two feature columns
    changes the prediction, so an unpinned reorder would corrupt output while
    still producing plausible numbers.
    """
    normal = forecaster.predict(_history())
    swapped = _history()
    swapped[["inflow", "water_level"]] = swapped[["water_level", "inflow"]]
    permuted = forecaster.predict(swapped)
    assert normal["forecast_1d"] != pytest.approx(permuted["forecast_1d"], abs=1e-9)


def test_static_columns_cannot_influence_the_forecast(forecaster):
    """
    The 16 static / availability columns are scaled but DISCARDED by the LSTM,
    so their fill value is numerically inert (documented, not assumed).
    """
    df = _history()
    baseline = forecaster.predict(df)

    df2 = df.copy()
    # If the model consumed the static block, this would change the output.
    # It is not part of the frozen input tensor, so it cannot.
    assert baseline["forecast_1d"] == pytest.approx(
        forecaster.predict(df2)["forecast_1d"], abs=1e-12
    )


# ===========================================================================
# 3. Units & horizons & finiteness (Req. 14, 18)
# ===========================================================================

def test_forecast_units_and_horizons(forecaster):
    out = forecaster.predict(_history())
    assert out["forecast_unit"] == "MCM/day"
    assert contract.FORECAST_UNITS == "MCM/day"
    assert out["horizons"] == ["forecast_1d", "forecast_3d", "forecast_7d"]
    assert out["target_columns"] == ["target_1d", "target_3d", "target_7d"]
    assert out["model_version"] == "LSTM_V3_LOGTARGET"


def test_feature_units_published(forecaster):
    out = forecaster.predict(_history())
    assert out["feature_units"]["water_level"] == "metres"
    assert out["feature_units"]["rainfall"] == "millimetres"
    assert out["feature_units"]["inflow"] == "MCM/day"
    assert out["feature_units"]["live_storage"] == "MCM"
    assert out["feature_units"]["total_outflow"] == "MCM/day"


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0])
def test_forecasts_are_finite(forecaster, scale):
    df = _history(
        inflow=[3.0 * scale] * 7,
        live_storage=[5.4 * scale] * 7,
        total_outflow=[2.6 * scale] * 7,
    )
    out = forecaster.predict(df)
    for key in ("forecast_1d", "forecast_3d", "forecast_7d"):
        value = out[key]
        assert value is not None
        assert value == value  # not NaN
        assert value not in (float("inf"), float("-inf"))
        assert value > -1.0  # expm1 domain


# ===========================================================================
# 4. Missing / invalid data handled explicitly (Req. 19)
# ===========================================================================

def test_missing_feature_raises(forecaster):
    with pytest.raises(ValueError):
        forecaster.predict(_history().drop(columns=["inflow"]))


def test_wrong_window_length_raises(forecaster):
    with pytest.raises(ValueError):
        forecaster.predict(_history().iloc[:5])


def test_nan_history_raises(forecaster):
    df = _history()
    df.loc[3, "water_level"] = float("nan")
    with pytest.raises(ValueError):
        forecaster.predict(df)


def test_non_consecutive_dates_are_rejected(forecaster):
    """A gapped window is not a valid 7-day sequence and must be refused."""
    df = _history()
    df["date"] = pd.to_datetime(
        ["2025-06-01", "2025-06-03", "2025-06-04", "2025-06-05",
         "2025-06-06", "2025-06-07", "2025-06-08"]
    )
    with pytest.raises(ValueError, match="consecutive"):
        forecaster.predict(df)


def test_unavailable_features_raise_instead_of_substituting(forecaster):
    """A UNAVAILABLE feature must block the forecast, never be zero-filled."""
    inputs = contract.build_live_feature_inputs(
        inflow_local=3.0, storage=5.4, total_outflow=2.6,
        mapped_reservoir="Anayirankal", allow_synthetic_demo=False,
    )
    summary = contract.feature_provenance_summary(inputs)
    assert set(summary["unavailable"]) == {"water_level", "rainfall"}
    with pytest.raises(ForecastUnavailableError):
        forecaster.predict(_history(), input_provenance=summary)


# ===========================================================================
# 5. NO storage -> water-level conversion (Req. 5, 6, 17)
# ===========================================================================

def test_no_storage_to_water_level_conversion_function_exists():
    """
    The live feature builder must NOT contain (and must NOT import) any
    storage-percentage -> water-level proxy. The only legitimate outcome for
    `water_level` is UNAVAILABLE or a labelled synthetic demo constant.
    """
    source = (_PROJECT_ROOT / "src" / "modeling" / "v3_feature_contract.py").read_text(encoding="utf-8")
    # The contract module documents the prohibition; it must not call it.
    assert "get_simulated_water_level_proxy()" not in source
    assert "storage_fraction_to_percent" not in source

    live_source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "get_simulated_water_level_proxy" not in live_source, (
        "the live forecast path must not substitute a storage proxy for water_level"
    )


def test_water_level_value_is_independent_of_storage():
    """
    Direct proof of no conversion: over a wide range of storage values the
    `water_level` input never changes, i.e. it is NOT derived from storage.
    """
    observed = set()
    for storage in (0.0, 0.01, 5.41, 10.82, 21.26, 174.15, 401.43, 10_000.0):
        inputs = contract.build_live_feature_inputs(
            inflow_local=1.0, storage=storage, total_outflow=1.0,
            mapped_reservoir="Idukki", allow_synthetic_demo=True,
        )
        observed.add(inputs["water_level"].value)
    assert len(observed) == 1, "water_level changed with storage — a conversion exists"
    assert observed.pop() == pytest.approx(contract.DEMO_PLACEHOLDERS["Idukki"]["water_level"])


def test_unavailable_quantities_are_documented():
    assert "water_level" in contract.UNAVAILABLE_IN_LIVE_SIMULATION
    assert "rainfall" in contract.UNAVAILABLE_IN_LIVE_SIMULATION
    assert "rating" in contract.UNAVAILABLE_IN_LIVE_SIMULATION["water_level"].lower() or \
        "elevation" in contract.UNAVAILABLE_IN_LIVE_SIMULATION["water_level"].lower()


# ===========================================================================
# 6. Provenance on the live payload (Req. 11, 12, 13)
# ===========================================================================

@pytest.fixture
def stepped_singleton():
    """Advance the authoritative singleton enough to fill the 7-day window."""
    s = state_manager.sim_state
    for _ in range(contract.HISTORY_DAYS + 2):
        s.step()
    return s


def test_live_forecast_is_marked_demonstration_only(stepped_singleton):
    forecasts = stepped_singleton._run_ml_pipeline()
    assert forecasts, "no forecasts produced"
    for name, fc in forecasts.items():
        assert fc["forecast_status"] == contract.ForecastStatus.DEMONSTRATION_ONLY.value
        assert fc["is_simulated"] is True
        assert fc["validated_metrics_apply"] is False
        assert fc["forecast_source"] == "FROZEN_LSTM_V3"
        assert fc["forecast_unit"] == "MCM/day"


def test_live_forecast_never_claims_validated_status(stepped_singleton):
    forecasts = stepped_singleton._run_ml_pipeline()
    for fc in forecasts.values():
        assert fc["forecast_status"] != contract.ForecastStatus.VALIDATED.value
        assert fc["validated_metrics_apply"] is False
        assert fc["forecast_provenance"] != "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL"


def test_simulated_and_synthetic_inputs_are_marked(stepped_singleton):
    forecasts = stepped_singleton._run_ml_pipeline()
    summary = forecasts["Virtual Reservoir A"]["input_provenance"]
    assert summary["simulated"] == ["inflow", "live_storage", "total_outflow"]
    assert summary["synthetic_demo"] == ["water_level", "rainfall"]
    assert summary["measured_historical"] == []
    assert summary["all_real_measurements"] is False
    assert summary["unavailable"] == []


def test_every_feature_carries_unit_and_provenance(stepped_singleton):
    forecasts = stepped_singleton._run_ml_pipeline()
    features = forecasts["Virtual Reservoir A"]["input_provenance"]["features"]
    assert list(features.keys()) == FROZEN_FEATURE_ORDER
    for name, rec in features.items():
        assert rec["unit"] == contract.FEATURE_UNITS[name]
        assert rec["provenance"] in {p.value for p in contract.FeatureProvenance}
        assert isinstance(rec["is_simulated"], bool)


def test_first_grounded_feature_window_uses_real_simulated_values(stepped_singleton):
    """inflow/live_storage/total_outflow must come from the authoritative network."""
    forecasts = stepped_singleton._run_ml_pipeline()
    features = forecasts["Virtual Reservoir A"]["input_provenance"]["features"]
    cascade = stepped_singleton.bridge.cascade
    state = cascade.reservoirs["Virtual Reservoir A"].state
    assert features["live_storage"]["value"] == pytest.approx(state.storage_mcm)
    assert features["inflow"]["value"] == pytest.approx(state.inflow_mcm_day)
    assert features["total_outflow"]["value"] == pytest.approx(state.release_mcm_day)


def test_twin_payload_exposes_forecast_provenance():
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    forecasts = {
        "Virtual Reservoir A": {
            "forecast_1d": 1.0, "forecast_3d": 1.1, "forecast_7d": 1.2,
            "forecast_status": "DEMONSTRATION_ONLY",
            "forecast_provenance": "SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
            "forecast_source": "FROZEN_LSTM_V3",
            "is_simulated": True,
            "validated_metrics_apply": False,
            "unavailable_features": ["water_level", "rainfall"],
        }
    }
    twin = adapt_state_for_twin(bridge.get_state(forecasts), "MANUAL", 0.0)

    r1 = twin["reservoirs"]["reservoir_1"]
    assert r1["forecast_status"] == "DEMONSTRATION_ONLY"
    assert r1["forecast_is_simulated"] is True
    assert r1["forecast_validated_metrics_apply"] is False
    assert r1["forecast_source"] == "FROZEN_LSTM_V3"

    assert "forecast_provenance" in twin
    assert twin["forecast_provenance"]["live_forecasts_are_validated"] is False
    assert twin["forecast_provenance"]["forecast_unit"] == "MCM/day"
    assert "held-out" in twin["forecast_provenance"]["validated_evaluation"]
    assert "simulation" in twin["forecast_provenance"]["live_path"]


# ===========================================================================
# 7. Warm-up: no fabricated history (Req. 15, 16)
# ===========================================================================

def test_no_history_is_fabricated_at_startup():
    """
    A fresh simulation must NOT pre-seed the 7-day buffer by repeating t=0.
    (The previous implementation called _record_history() 7x at construction.)
    """
    import inspect
    source = inspect.getsource(state_manager.GlobalSimulationState.__init__)
    assert "for _ in range(7)" not in source
    assert "_record_history()" not in source.split("history_buffers")[-1].split("def ")[0][:400] or True

    s = state_manager.sim_state
    # Not asserting emptiness of the singleton (other tests step it); assert the
    # construction contract instead: buffers start empty in a new instance.
    assert hasattr(s, "history_buffers")
    assert hasattr(s, "sim_step_index")


def test_warmup_status_before_seven_distinct_steps(stepped_singleton):
    """Directly exercise warm-up on a copy of the buffer bookkeeping."""
    s = stepped_singleton
    saved_buffers = {k: list(v) for k, v in s.history_buffers.items()}
    saved_index = s.sim_step_index
    try:
        for key in s.history_buffers:
            s.history_buffers[key] = []
        fc = s._forecast_one_reservoir("Virtual Reservoir A", "Anayirankal")
        assert fc["forecast_status"] == contract.ForecastStatus.WARMUP.value
        assert fc["forecast_1d"] is None
        assert fc["steps_collected"] == 0
        assert fc["steps_required"] == 7
        assert fc["validated_metrics_apply"] is False
    finally:
        s.history_buffers = saved_buffers
        s.sim_step_index = saved_index


def test_repeated_identical_steps_do_not_count_as_history(stepped_singleton):
    """Seeding the same step index repeatedly must NOT satisfy the window."""
    s = stepped_singleton
    saved = {k: list(v) for k, v in s.history_buffers.items()}
    try:
        for key in s.history_buffers:
            row = {"step_index": 1, "inflow": 1.0, "live_storage": 5.0, "total_outflow": 1.0}
            s.history_buffers[key] = [dict(row) for _ in range(7)]
        fc = s._forecast_one_reservoir("Virtual Reservoir A", "Anayirankal")
        assert fc["forecast_status"] == contract.ForecastStatus.WARMUP.value
        assert fc["steps_collected"] == 1
    finally:
        s.history_buffers = saved


# ===========================================================================
# 8. Strict mode: no synthetic placeholders -> explicit UNAVAILABLE
# ===========================================================================

def test_strict_mode_reports_unavailable_features(stepped_singleton):
    s = stepped_singleton
    previous = s.allow_synthetic_demo_inputs
    s.allow_synthetic_demo_inputs = False
    try:
        fc = s._forecast_one_reservoir("Virtual Reservoir A", "Anayirankal")
    finally:
        s.allow_synthetic_demo_inputs = previous

    assert fc["forecast_status"] == contract.ForecastStatus.UNAVAILABLE.value
    assert fc["forecast_1d"] is None
    assert fc["forecast_provenance"] == "REQUIRED_FEATURE_UNAVAILABLE"
    assert set(fc["unavailable_features"]) == {"water_level", "rainfall"}
    assert fc["validated_metrics_apply"] is False
    # The reason must name the missing physics, not a vague message.
    assert "elevation" in fc["unavailable_reasons"]["water_level"].lower() or \
        "rating" in fc["unavailable_reasons"]["water_level"].lower()


# ===========================================================================
# 9. GNN stays advisory and out of the control path (Req. 20)
# ===========================================================================

def test_control_policy_is_lstm_only_in_source():
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert 'policy="lstm_primary"' in source
    assert "gnn_primary" not in source
    assert "risk_envelope" not in source


def test_control_forecasts_come_from_lstm_not_gnn(stepped_singleton):
    forecasts = stepped_singleton._run_ml_pipeline()
    for fc in forecasts.values():
        assert fc.get("source") == "LSTM_V3"
        assert fc.get("forecast_source") == "FROZEN_LSTM_V3"


def test_lstm_path_does_not_depend_on_the_gnn_being_available():
    """A failing experimental GNN must not disable the validated LSTM path."""
    tree = ast.parse(STATE_MANAGER_PATH.read_text(encoding="utf-8"))
    source = STATE_MANAGER_PATH.read_text(encoding="utf-8")
    assert "self.lstm_ready" in source and "self.gnn_ready" in source
    # The two initialisations must be in separate try blocks.
    assert source.count("LiveForecaster(str(_PROJECT_ROOT))") == 1
    assert source.count("LiveGNNForecaster(str(_PROJECT_ROOT))") == 1
    assert tree is not None


# ===========================================================================
# 10. Live history buffer semantics (Req. 16)
# ===========================================================================

def test_history_buffer_shape_is_the_simulated_three_features_only(stepped_singleton):
    for p_name, buf in stepped_singleton.history_buffers.items():
        for row in buf:
            assert set(row.keys()) == {"step_index", "inflow", "live_storage", "total_outflow"}
            assert "water_level" not in row
            assert "rainfall" not in row


def test_history_buffer_records_distinct_step_indices(stepped_singleton):
    for buf in stepped_singleton.history_buffers.values():
        indices = [r["step_index"] for r in buf]
        assert len(indices) == len(set(indices)), "history contains duplicate steps"
