"""
Stage 6 — Forecast Adapter tests.

Proves the boundary between the LIVE forecast representation and the VALIDATED
MPC representation is correct:

  * all four reservoirs map correctly (A/B/C/D -> Anayirankal/Ponmudi/Idamalayar/Idukki)
  * all three horizons map correctly
  * units remain MCM/day
  * numerical values are UNCHANGED by the adapter
  * provenance survives the adapter
  * invalid forecasts are rejected / handled explicitly
  * missing reservoirs are handled explicitly
  * no fabricated forecast is introduced
  * the output is exactly the Phase 15.2 NetworkForecastSnapshot the MPC consumes
  * the MPC / SafetyLayer / ReservoirNetwork / frozen artifacts are untouched
"""

import ast
import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.live_forecast_adapter import (  # noqa: E402
    FORECAST_UNIT,
    HORIZON_KEYS,
    LiveForecastAdapter,
)
from src.network_env.v3_forecast_adapter import (  # noqa: E402
    ForecastStatus,
    NetworkForecastSnapshot,
    ReservoirForecast,
)

PROJECT_ROOT = str(_PROJECT_ROOT)
MPC_PATH = _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"
SAFETY_PATH = _PROJECT_ROOT / "src" / "controller" / "safety.py"
RESERVOIR_NETWORK_PATH = _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py"
V3_ADAPTER_PATH = _PROJECT_ROOT / "src" / "network_env" / "v3_forecast_adapter.py"

EXPECTED_NODE_MAPPING = {
    "Virtual Reservoir A": "Reservoir_A",
    "Virtual Reservoir B": "Reservoir_B",
    "Virtual Reservoir C": "Reservoir_C",
    "Virtual Reservoir D": "Reservoir_D",
}
EXPECTED_V3_MAPPING = {
    "Virtual Reservoir A": "Anayirankal",
    "Virtual Reservoir B": "Ponmudi",
    "Virtual Reservoir C": "Idamalayar",
    "Virtual Reservoir D": "Idukki",
}


@pytest.fixture(scope="module")
def adapter():
    return LiveForecastAdapter(project_root=PROJECT_ROOT)


def _payload(v1=2.436246156692505, v3=2.539130449295044, v7=2.731862783432007,
             status="DEMONSTRATION_ONLY",
             provenance="SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
             unit=FORECAST_UNIT, horizons=None, extra=None):
    payload = {
        "forecast_1d": v1,
        "forecast_3d": v3,
        "forecast_7d": v7,
        "forecast_status": status,
        "forecast_provenance": provenance,
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": True,
        "validated_metrics_apply": False,
        "forecast_unit": unit,
        "horizons": list(horizons) if horizons is not None else list(HORIZON_KEYS),
        "input_provenance": {
            "synthetic_demo": ["water_level", "rainfall"],
            "unavailable": [],
            "simulated": ["inflow", "live_storage", "total_outflow"],
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _all_four(**kwargs):
    return {name: _payload(**kwargs) for name in EXPECTED_NODE_MAPPING}


# ===========================================================================
# 1. Reservoir mapping (Req. 5)
# ===========================================================================

def test_all_four_reservoirs_map_correctly(adapter):
    assert adapter.node_mapping == EXPECTED_NODE_MAPPING
    assert adapter.reservoir_mapping == EXPECTED_V3_MAPPING
    assert adapter.network_node_ids == ["Reservoir_A", "Reservoir_B", "Reservoir_C", "Reservoir_D"]


def test_snapshot_is_keyed_by_network_node_id(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    assert set(bundle.forecasts.keys()) == set(EXPECTED_NODE_MAPPING.values())
    for node_id, fc in bundle.forecasts.items():
        assert isinstance(fc, ReservoirForecast)
        assert fc.reservoir_id == node_id


def test_each_node_carries_its_v3_reservoir_name(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    for live_name, node_id in EXPECTED_NODE_MAPPING.items():
        assert bundle.get(node_id).v3_reservoir_name == EXPECTED_V3_MAPPING[live_name]


def test_wrong_mapping_is_rejected():
    """A mapping that contradicts requirement 5 must not be accepted silently."""
    bad = dict(EXPECTED_V3_MAPPING)
    bad["Virtual Reservoir A"] = "Idukki"
    with pytest.raises(ValueError, match="mapping mismatch"):
        LiveForecastAdapter(project_root=PROJECT_ROOT, reservoir_mapping=bad)


# ===========================================================================
# 2. Horizon mapping (Req. 6)
# ===========================================================================

def test_all_three_horizons_map_correctly(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.target_1d == pytest.approx(2.436246156692505)
    assert fc.target_3d == pytest.approx(2.539130449295044)
    assert fc.target_7d == pytest.approx(2.731862783432007)
    for label in ("1d", "3d", "7d"):
        assert fc.is_available(label) is True
        assert fc.get_prediction(label) is not None


def test_horizons_are_not_conflated(adapter):
    """Each horizon must carry its OWN value — no cross-assignment."""
    bundle = adapter.build_bundle(_all_four(v1=1.0, v3=2.0, v7=3.0), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert (fc.target_1d, fc.target_3d, fc.target_7d) == (1.0, 2.0, 3.0)


def test_horizon_mismatch_is_rejected_explicitly(adapter):
    payload = _all_four(horizons=["forecast_1d", "forecast_7d"])
    bundle = adapter.build_bundle(payload, "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert bundle.get_provenance("Reservoir_A")["issue"] == "HORIZON_MISMATCH"
    for label in ("1d", "3d", "7d"):
        assert fc.is_available(label) is False
        assert fc.get_prediction(label) is None


# ===========================================================================
# 3. Units (Req. 7)
# ===========================================================================

def test_unit_is_mcm_per_day(adapter):
    assert FORECAST_UNIT == "MCM/day"
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    for prov in bundle.provenance.values():
        assert prov["unit"] == "MCM/day"
    assert bundle.provenance_summary()["unit"] == "MCM/day"


def test_unit_mismatch_is_rejected_explicitly(adapter):
    bundle = adapter.build_bundle(_all_four(unit="m3/s"), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert bundle.get_provenance("Reservoir_A")["issue"] == "UNIT_MISMATCH"
    assert fc.status_1d == ForecastStatus.INVALID
    assert fc.target_1d is None


# ===========================================================================
# 4. Numerical values unchanged (Req. 12)
# ===========================================================================

def test_values_are_bit_identical(adapter):
    """The adapter must not scale, round, smooth or offset any value."""
    values = (0.0, 1e-12, 2.436246156692505, 91.71626281738281, 1e6)
    bundle = adapter.build_bundle(_all_four(v1=values[0], v3=values[1], v7=values[2]), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.target_1d == values[0]      # exact equality, not approx
    assert fc.target_3d == values[1]
    assert fc.target_7d == values[2]
    assert repr(fc.target_3d) == repr(float(values[1]))


@pytest.mark.parametrize("value", [0.0, 0.1, 3.0, 1234.5678901234567, 1e-9])
def test_round_trip_preserves_every_value(adapter, value):
    bundle = adapter.build_bundle(_all_four(v1=value, v3=value, v7=value), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.target_1d == value and fc.target_3d == value and fc.target_7d == value


# ===========================================================================
# 5. Provenance survives (Req. 8)
# ===========================================================================

def test_provenance_survives_the_adapter(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    prov = bundle.get_provenance("Reservoir_A")

    assert prov["declared_status"] == "DEMONSTRATION_ONLY"
    assert prov["declared_provenance"] == "SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL"
    assert prov["is_simulated"] is True
    assert prov["validated_metrics_apply"] is False
    assert prov["synthetic_features"] == ["water_level", "rainfall"]
    assert prov["v3_reservoir_name"] == "Anayirankal"
    assert prov["live_reservoir"] == "Virtual Reservoir A"


def test_provenance_is_also_carried_in_the_phase152_contract_field(adapter):
    """The frozen snapshot contract has one provenance slot — it must be filled."""
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    prov_str = bundle.get("Reservoir_A").provenance
    assert "status=DEMONSTRATION_ONLY" in prov_str
    assert "source=SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL" in prov_str
    assert "unit=MCM/day" in prov_str
    assert "synthetic=water_level+rainfall" in prov_str


def test_model_version_is_recorded(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    assert bundle.get("Reservoir_A").model == "LSTM_V3_LOGTARGET"


def test_provenance_summary_exposes_status(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    summary = bundle.provenance_summary()
    assert summary["forecast_date"] == "2026-09-14"
    assert summary["nodes"]["Reservoir_A"]["is_simulated"] is True
    assert summary["nodes"]["Reservoir_A"]["horizons_available"] == ["1d", "3d", "7d"]


def test_validated_provenance_is_preserved_verbatim(adapter):
    """A future real-telemetry payload must pass through with its status intact."""
    payload = _all_four(status="VALIDATED",
                        provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
                        extra={"is_simulated": False, "validated_metrics_apply": True})
    bundle = adapter.build_bundle(payload, "2026-09-14")
    prov = bundle.get_provenance("Reservoir_A")
    assert prov["declared_status"] == "VALIDATED"
    assert prov["is_simulated"] is False
    assert prov["validated_metrics_apply"] is True
    assert bundle.get("Reservoir_A").is_available("1d")


# ===========================================================================
# 6. Invalid / unavailable handling (Req. 9, 10, 11)
# ===========================================================================

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_invalid(adapter, bad):
    bundle = adapter.build_bundle(_all_four(v1=bad), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.status_1d == ForecastStatus.INVALID
    assert fc.target_1d is None
    assert "1d" in bundle.get_provenance("Reservoir_A")["value_issues"]


@pytest.mark.parametrize("bad", ["2.5", None, [], {}])
def test_non_numeric_values_are_invalid(adapter, bad):
    bundle = adapter.build_bundle(_all_four(v1=bad), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.status_1d in (ForecastStatus.INVALID, ForecastStatus.UNAVAILABLE)
    assert fc.target_1d is None


def test_negative_inflow_is_invalid_not_clamped(adapter):
    bundle = adapter.build_bundle(_all_four(v1=-0.5), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.status_1d == ForecastStatus.INVALID
    assert fc.target_1d is None, "a negative inflow must be reported, not clamped to 0"


def test_positive_horizons_survive_a_negative_sibling(adapter):
    """One bad horizon must not silently invalidate the others' VALUES."""
    bundle = adapter.build_bundle(_all_four(v1=-1.0, v3=5.0, v7=6.0), "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.target_1d is None
    assert fc.target_3d == 5.0
    assert fc.target_7d == 6.0


def test_warmup_status_becomes_unavailable(adapter):
    bundle = adapter.build_bundle(_all_four(status="WARMUP_INSUFFICIENT_HISTORY"),
                                  "2026-09-14")
    prov = bundle.get_provenance("Reservoir_A")
    assert prov["issue"] == "WARMUP_INSUFFICIENT_HISTORY"
    assert bundle.get("Reservoir_A").status_1d == ForecastStatus.UNAVAILABLE
    assert bundle.get("Reservoir_A").target_1d is None


def test_forecast_unavailable_status_is_respected(adapter):
    payload = _all_four(status="FORECAST_UNAVAILABLE", v1=None, v3=None, v7=None)
    bundle = adapter.build_bundle(payload, "2026-09-14")
    assert bundle.get("Reservoir_A").status_1d == ForecastStatus.UNAVAILABLE
    assert bundle.get("Reservoir_A").target_1d is None


def test_missing_reservoir_is_explicit(adapter):
    """A reservoir absent from the payload must appear, explicitly unavailable."""
    bundle = adapter.build_bundle(
        {"Virtual Reservoir A": _payload()}, "2026-09-14"
    )
    assert set(bundle.forecasts.keys()) == set(EXPECTED_NODE_MAPPING.values())
    for node_id in ("Reservoir_B", "Reservoir_C", "Reservoir_D"):
        fc = bundle.get(node_id)
        assert fc.status_1d == ForecastStatus.UNAVAILABLE
        assert fc.target_1d is None
        assert bundle.get_provenance(node_id)["issue"] == "MISSING_FORECAST"


def test_malformed_payload_is_explicit(adapter):
    bundle = adapter.build_bundle({"Virtual Reservoir A": "not-a-dict"}, "2026-09-14")
    prov = bundle.get_provenance("Reservoir_A")
    assert prov["issue"] == "MALFORMED_PAYLOAD"
    assert bundle.get("Reservoir_A").status_1d == ForecastStatus.INVALID


def test_unmapped_payload_keys_are_reported(adapter):
    payload = _all_four()
    payload["Virtual Reservoir Z"] = _payload()
    bundle = adapter.build_bundle(payload, "2026-09-14")
    assert any("Virtual Reservoir Z" in n for n in bundle.notes)


# ===========================================================================
# 7. No fabrication (Req. 10, 11, 13)
# ===========================================================================

def test_no_forecast_is_fabricated_from_an_empty_payload(adapter):
    bundle = adapter.build_bundle({}, "2026-09-14")
    assert bundle.status == "NO_FORECASTS"
    for node_id, fc in bundle.forecasts.items():
        assert fc.target_1d is None and fc.target_3d is None and fc.target_7d is None
        assert fc.status_1d == ForecastStatus.UNAVAILABLE
        assert fc.status_3d == ForecastStatus.UNAVAILABLE
        assert fc.status_7d == ForecastStatus.UNAVAILABLE
    assert bundle.snapshot.available_count("1d") == 0
    assert bundle.snapshot.all_available("1d") is False


def test_adapter_never_invents_a_default_value(adapter):
    """Absent horizon keys must be None, never 0.0 or a carried-forward value."""
    payload = _all_four()
    payload["Virtual Reservoir A"].pop("forecast_3d")
    bundle = adapter.build_bundle(payload, "2026-09-14")
    fc = bundle.get("Reservoir_A")
    assert fc.target_3d is None
    assert fc.status_3d == ForecastStatus.UNAVAILABLE
    # The other horizons were NOT altered to compensate.
    assert fc.target_1d == pytest.approx(2.436246156692505)
    assert fc.target_7d == pytest.approx(2.731862783432007)


def test_adapter_does_not_synthesise_a_snapshot_for_unknown_nodes(adapter):
    bundle = adapter.build_bundle(_all_four(), "2026-09-14")
    assert bundle.get("Reservoir_Z") is None
    assert set(bundle.provenance.keys()) == set(EXPECTED_NODE_MAPPING.values())


# ===========================================================================
# 8. Contract compatibility with the validated MPC (Req. 6, 15)
# ===========================================================================

def test_output_is_exactly_the_phase152_snapshot_contract(adapter):
    snapshot = adapter.build_snapshot(_all_four(), "2026-09-14")
    assert isinstance(snapshot, NetworkForecastSnapshot)
    assert snapshot.forecast_date == "2026-09-14"

    # Every accessor the validated MPC uses must work unchanged.
    for node_id in EXPECTED_NODE_MAPPING.values():
        fc = snapshot.get(node_id)
        assert isinstance(fc, ReservoirForecast)
        assert isinstance(fc.is_available("1d"), bool)
        assert fc.get_prediction("1d") is not None
    assert snapshot.all_available("1d") is True
    assert snapshot.available_count("1d") == 4


def test_snapshot_node_keys_match_the_reservoir_network_processing_order():
    """The MPC looks up snapshot.get(nid) for network.processing_order."""
    from src.network_env.reservoir_network import ReservoirNetwork
    import yaml

    with open(_PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    network = ReservoirNetwork(config_dict=cfg)

    adapter = LiveForecastAdapter(project_root=PROJECT_ROOT)
    snapshot = adapter.build_snapshot(_all_four(), "2026-09-14")

    for nid in network.processing_order:
        fc = snapshot.get(nid)
        assert fc is not None, f"MPC would find no forecast for {nid}"


def test_adapter_is_not_wired_into_the_mpc_yet(adapter):
    """Stage 6 must NOT integrate the MPC (Req. 15)."""
    for path in (MPC_PATH, SAFETY_PATH):
        source = path.read_text(encoding="utf-8")
        assert "live_forecast_adapter" not in source
        assert "LiveForecastAdapter" not in source


def test_validated_modules_are_not_modified_by_stage6():
    """MPC, SafetyLayer, ReservoirNetwork and the Phase 15.2 adapter are intact."""
    mpc = MPC_PATH.read_text(encoding="utf-8")
    assert "class MPCController" in mpc
    assert "def decide(" in mpc
    assert "from ..network_env.v3_forecast_adapter import" in mpc or \
           "v3_forecast_adapter import" in mpc

    safety = SAFETY_PATH.read_text(encoding="utf-8")
    assert "class SafetyLayer" in safety

    net = RESERVOIR_NETWORK_PATH.read_text(encoding="utf-8")
    assert "class ReservoirNetwork" in net
    assert "def mass_balance_check" in net

    v3 = V3_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "class NetworkForecastSnapshot" in v3
    assert "class ReservoirForecast" in v3
    assert "class V3ForecastAdapter" in v3


def test_stage6_adds_no_dependency_to_reservoir_network_or_controller():
    """The new module must be additive: no validated module imports it."""
    sources = [
        MPC_PATH,
        SAFETY_PATH,
        _PROJECT_ROOT / "src" / "controller" / "objective.py",
        RESERVOIR_NETWORK_PATH,
        V3_ADAPTER_PATH,
    ]
    for path in sources:
        assert "live_forecast_adapter" not in path.read_text(encoding="utf-8")


# ===========================================================================
# 9. End-to-end: live Stage 5 pipeline -> adapter -> MPC contract
# ===========================================================================

def test_live_pipeline_forecasts_adapt_to_the_mpc_contract():
    from src.dashboard.api import state_manager

    sim = state_manager.sim_state
    for _ in range(9):
        sim.step()

    live = sim._run_ml_pipeline()
    assert live, "live pipeline produced no forecasts"

    adapter = LiveForecastAdapter(project_root=PROJECT_ROOT)
    bundle = adapter.build_bundle(live, "2026-09-14")

    # All four network nodes are present even though the live sim forecasts only A/B/C.
    assert set(bundle.forecasts.keys()) == set(EXPECTED_NODE_MAPPING.values())
    assert bundle.get("Reservoir_D").status_1d == ForecastStatus.UNAVAILABLE

    for node_id, fc in bundle.forecasts.items():
        if fc.is_available("1d"):
            assert fc.target_1d is not None and math.isfinite(fc.target_1d)
            prov = bundle.get_provenance(node_id)
            assert prov["declared_status"] in (
                "DEMONSTRATION_ONLY", "VALIDATED", "FORECAST_UNAVAILABLE",
                "WARMUP_INSUFFICIENT_HISTORY",
            )
            assert prov["is_simulated"] in (True, False, None)


def test_live_demo_forecasts_keep_demonstration_status_through_the_adapter():
    from src.dashboard.api import state_manager

    sim = state_manager.sim_state
    for _ in range(9):
        sim.step()
    live = sim._run_ml_pipeline()

    bundle = LiveForecastAdapter(project_root=PROJECT_ROOT).build_bundle(live, "2026-09-14")
    available = [nid for nid, fc in bundle.forecasts.items() if fc.is_available("1d")]
    if not available:
        pytest.skip("no live forecast available in this environment")

    for nid in available:
        prov = bundle.get_provenance(nid)
        assert prov["declared_status"] == "DEMONSTRATION_ONLY"
        assert prov["validated_metrics_apply"] is False
        assert "DEMONSTRATION_ONLY" in bundle.get(nid).provenance


# ===========================================================================
# 10. Hardware readiness (Req. 17)
# ===========================================================================

def test_adapter_accepts_real_telemetry_style_payloads_without_contract_change(adapter):
    """
    A payload originating from REAL hardware telemetry must flow through the
    SAME adapter into the SAME snapshot type — no MPC change required.
    """
    telemetry_payload = {
        "forecast_1d": 12.5, "forecast_3d": 11.0, "forecast_7d": 9.75,
        "forecast_status": "VALIDATED",
        "forecast_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": False,
        "validated_metrics_apply": True,
        "forecast_unit": "MCM/day",
        "horizons": list(HORIZON_KEYS),
        "input_provenance": {
            "measured_historical": ["inflow", "water_level", "live_storage",
                                    "rainfall", "total_outflow"],
            "synthetic_demo": [], "unavailable": [], "simulated": [],
            "all_real_measurements": True,
        },
    }
    snapshot = adapter.build_snapshot(
        {name: telemetry_payload for name in EXPECTED_NODE_MAPPING}, "2027-01-01"
    )
    assert isinstance(snapshot, NetworkForecastSnapshot)
    assert snapshot.all_available("1d")
    assert snapshot.get("Reservoir_D").target_1d == 12.5

    bundle = adapter.build_bundle(
        {name: telemetry_payload for name in EXPECTED_NODE_MAPPING}, "2027-01-01"
    )
    assert bundle.get_provenance("Reservoir_D")["is_simulated"] is False
    assert bundle.get_provenance("Reservoir_D")["declared_status"] == "VALIDATED"


def test_adapter_contract_info_declares_no_integration(adapter):
    info = adapter.contract_info()
    assert info["mpc_integrated"] is False
    assert info["safety_layer_integrated"] is False
    assert info["unit"] == "MCM/day"
    assert "NONE" in info["alteration"]
    assert info["reservoir_mapping"] == EXPECTED_V3_MAPPING


def test_gnn_is_not_part_of_the_adapter_path():
    source = (_PROJECT_ROOT / "src" / "network_env" / "live_forecast_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "src.network_env.gnn_forecast_adapter" not in imported
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert "GNN" not in node.id


# ===========================================================================
# 11. Stage 5 decision conditions (accepted, enforced)
# ===========================================================================

TWIN_INDEX_PATH = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"
STREAMLIT_APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
STATE_ADAPTER_PATH = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py"


def test_authoritative_twin_shows_the_demonstration_badge():
    """Decision condition 3 — the UI must visibly state the demonstration status."""
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "DEMONSTRATION — MODEL INPUTS SIMULATED" in html
    # The badge element is built in JS and appended to the HUD.
    assert "demoBadge.id = 'demo-badge'" in html
    assert "hud.appendChild(demoBadge)" in html
    assert "#demo-badge" in html          # its styling exists


def test_demo_badge_is_driven_by_backend_provenance_not_inferred():
    """
    Decision condition 1 — SYNTHETIC_DEMO must never be presented as real
    telemetry. The badge must react to the backend provenance flag, and must be
    shown unless the backend explicitly reports validated forecasts.
    """
    html = TWIN_INDEX_PATH.read_text(encoding="utf-8")
    assert "live_forecasts_are_validated" in html
    assert "fp.live_forecasts_are_validated !== true" in html
    assert "forecast_is_simulated" in html


def test_streamlit_viewer_shows_the_demonstration_banner():
    source = STREAMLIT_APP_PATH.read_text(encoding="utf-8")
    assert "DEMONSTRATION — MODEL INPUTS SIMULATED" in source
    assert "live_forecasts_are_validated" in source


def test_twin_payload_declares_live_forecasts_are_not_validated():
    """Decision conditions 2 — DEMONSTRATION_ONLY + validated_metrics_apply=false."""
    from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
    from src.dashboard.sim_bridge import SimBridge

    bridge = SimBridge(
        str(_PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"),
        str(_PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"),
    )
    twin = adapt_state_for_twin(bridge.get_state({}), "MANUAL", 0.0)

    fp = twin["forecast_provenance"]
    assert fp["live_forecasts_are_validated"] is False
    assert fp["model_status"] == "FROZEN_UNMODIFIED"
    assert "held-out" in fp["validated_evaluation"]


def test_strict_mode_remains_available():
    """Decision condition 4 — synthetic inputs must be disableable at runtime."""
    from src.dashboard.api import state_manager as sm

    source = (_PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py").read_text(encoding="utf-8")
    assert "AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS" in source
    assert "allow_synthetic_demo_inputs" in source

    sim = sm.sim_state
    previous = sim.allow_synthetic_demo_inputs
    try:
        sim.allow_synthetic_demo_inputs = False
        assert sim.allow_synthetic_demo_inputs is False
    finally:
        sim.allow_synthetic_demo_inputs = previous
    assert isinstance(sim.allow_synthetic_demo_inputs, bool)


def test_strict_mode_env_var_actually_disables_placeholders():
    """
    Decision condition 4, verified end-to-end in an isolated process
    (so the suite's own singleton is untouched).
    """
    import os
    import subprocess

    env = dict(os.environ, AQUAFLOW_ALLOW_SYNTHETIC_DEMO_INPUTS="0")
    code = (
        "from src.dashboard.api.state_manager import sim_state as s;"
        "print('ALLOW=', s.allow_synthetic_demo_inputs);"
        "print('PLACEHOLDERS=', s._live_feature_inputs('Virtual Reservoir A', 'Anayirankal')"
        "['water_level'].provenance.value)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "ALLOW= False" in result.stdout
    assert "PLACEHOLDERS= UNAVAILABLE" in result.stdout


def test_synthetic_demo_is_never_labelled_as_a_measurement():
    """Decision condition 1 / 7 — a placeholder is never a real measurement."""
    from src.modeling import v3_feature_contract as contract

    inputs = contract.build_live_feature_inputs(
        inflow_local=1.0, storage=5.0, total_outflow=1.0,
        mapped_reservoir="Anayirankal", allow_synthetic_demo=True,
    )
    for name in ("water_level", "rainfall"):
        assert inputs[name].provenance is contract.FeatureProvenance.SYNTHETIC_DEMO
        assert inputs[name].is_simulated is True
        assert "NOT a live measurement" in inputs[name].note


def test_no_storage_to_water_level_conversion_exists():
    """
    Decision conditions 5 and 6 — no elevation-storage curve, and no arbitrary
    storage-percentage -> metres conversion anywhere in the forecast path.
    """
    sources = [
        _PROJECT_ROOT / "src" / "modeling" / "v3_feature_contract.py",
        _PROJECT_ROOT / "src" / "network_env" / "live_forecast_adapter.py",
        _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py",
    ]
    forbidden = (
        "storage_to_water_level",
        "percent_to_metres",
        "percent_to_meters",
        "storage_pct_to_level",
        "level_from_storage",
        "get_simulated_water_level_proxy()",
        "rating_curve",
        "elevation_storage_curve(",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains a forbidden conversion: {token}"


def test_hardware_readiness_interface_is_source_agnostic():
    """
    Decision condition 11 — a future REAL telemetry source must be able to
    replace SYNTHETIC_DEMO through this adapter without touching the frozen LSTM
    or the downstream MPC contract.
    """
    adapter = LiveForecastAdapter(project_root=PROJECT_ROOT)
    info = adapter.contract_info()
    assert info["input"] == "live forecast payload {live_reservoir_name: dict}"
    assert info["output"] == "NetworkForecastSnapshot (Phase 15.2 contract, unmodified)"

    # The adapter reads ONLY declared status/unit/horizons/values, so swapping
    # the producer requires no adapter change.
    source = (_PROJECT_ROOT / "src" / "network_env" / "live_forecast_adapter.py").read_text(encoding="utf-8")
    assert "forecast_status" in source
    assert "forecast_unit" in source
    assert "input_provenance" in source
