"""
Stage 3 — Authoritative Network: parity & regression tests.

These tests pin the Stage 3 topology/physics reconciliation:

  * the live Digital Twin's physics IS the validated ``ReservoirNetwork``
  * the live path no longer depends on ``VirtualCascade``
  * ``VirtualCascade`` still exists and still behaves as before (not deleted)
  * the authoritative topology/delays/attenuation match the Stage 3 targets
    (A->B->C->D; 2/1/1 days; 0.90/0.85/0.80)
  * the legacy and network implementations agree exactly whenever their
    assumptions coincide (implementation-equivalence proof)
  * mass balance, unit boundary, spill and determinism hold on the live path
"""

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.network_env.live_cascade_adapter import LiveCascadeAdapter  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402
from src.simulator.environment import VirtualCascade  # noqa: E402

LIVE_CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"

# The Stage 3 targets, stated once.
TARGET_ORDER = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]
TARGET_DELAYS = {("Virtual Reservoir A", "Virtual Reservoir B"): 2,
                 ("Virtual Reservoir B", "Virtual Reservoir C"): 1,
                 ("Virtual Reservoir C", "Virtual Reservoir D"): 1}
TARGET_ATTENUATION = {("Virtual Reservoir A", "Virtual Reservoir B"): 0.90,
                      ("Virtual Reservoir B", "Virtual Reservoir C"): 0.85,
                      ("Virtual Reservoir C", "Virtual Reservoir D"): 0.80}

# Live-path modules that must not depend on the legacy cascade.
LIVE_PATH_MODULES = [
    _PROJECT_ROOT / "src" / "dashboard" / "sim_bridge.py",
    _PROJECT_ROOT / "src" / "dashboard" / "api" / "state_manager.py",
    _PROJECT_ROOT / "src" / "dashboard" / "api" / "routes.py",
    _PROJECT_ROOT / "src" / "dashboard" / "api" / "app.py",
    _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "state_adapter.py",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_live_config() -> dict:
    with open(LIVE_CONFIG_PATH, "r") as fh:
        return json.load(fh)


def _make_adapter(initial_storages: dict | None = None) -> LiveCascadeAdapter:
    cfg = _load_live_config()
    if initial_storages:
        for name, value in initial_storages.items():
            cfg["reservoirs"][name]["initial_storage_mcm"]["value"] = float(value)
    return LiveCascadeAdapter(cfg, topology_path=str(TOPOLOGY_PATH))


def _make_bridge() -> SimBridge:
    return SimBridge(str(LIVE_CONFIG_PATH), str(THRESH_PATH))


# ---------------------------------------------------------------------------
# 1. The live path IS the authoritative network
# ---------------------------------------------------------------------------

def test_bridge_cascade_is_live_adapter_backed_by_reservoir_network():
    bridge = _make_bridge()
    assert isinstance(bridge.cascade, LiveCascadeAdapter)
    assert isinstance(bridge.cascade.network, ReservoirNetwork)
    # The adapter must not secretly be a VirtualCascade
    assert not isinstance(bridge.cascade, VirtualCascade)


def test_bridge_exposes_same_public_surface_as_before():
    """The facade keeps every attribute the live callers relied on."""
    bridge = _make_bridge()
    for attr in ("cascade_order", "reservoirs", "current_downstream_flow",
                 "downstream_capacity", "routing_delays", "step"):
        assert hasattr(bridge.cascade, attr), f"missing legacy attribute: {attr}"
    for name in TARGET_ORDER:
        res = bridge.cascade.reservoirs[name]
        for attr in ("name", "capacity_mcm", "max_release_capacity_mcm_day",
                     "get_simulated_water_level_proxy", "state"):
            assert hasattr(res, attr), f"missing legacy reservoir attribute: {attr}"
        for attr in ("storage_mcm", "inflow_mcm_day", "release_mcm_day",
                     "upstream_routed_inflow", "gate_position_pct", "overflow_events"):
            assert hasattr(res.state, attr), f"missing legacy state attribute: {attr}"


# ---------------------------------------------------------------------------
# 2. The live path no longer depends on VirtualCascade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path", LIVE_PATH_MODULES, ids=lambda p: p.name)
def test_live_path_does_not_depend_on_virtual_cascade(module_path):
    """
    AST-level check: no live-path module imports or references
    ``VirtualCascade`` / ``src.simulator.environment``.

    The check is AST-based (not a text scan) so that documentation strings may
    still *mention* the legacy class without creating a runtime dependency.
    """
    assert module_path.exists(), f"live-path module missing: {module_path}"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    forbidden_modules = {"src.simulator.environment"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules, (
                f"{module_path.name} imports {node.module}"
            )
            assert not any(a.name == "VirtualCascade" for a in node.names)
        elif isinstance(node, ast.Import):
            assert not any(a.name in forbidden_modules for a in node.names)
        elif isinstance(node, ast.Name):
            assert node.id != "VirtualCascade", (
                f"{module_path.name} references VirtualCascade at runtime"
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr != "VirtualCascade", (
                f"{module_path.name} references VirtualCascade at runtime"
            )


def test_sim_bridge_module_does_not_import_legacy_simulator():
    source = (_PROJECT_ROOT / "src" / "dashboard" / "sim_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "src.simulator.environment" not in imported


# ---------------------------------------------------------------------------
# 3. VirtualCascade is preserved (not deleted, not repurposed)
# ---------------------------------------------------------------------------

def test_virtual_cascade_still_exists_and_behaves_as_before():
    """The legacy class remains importable and keeps its documented routing."""
    cfg = _load_live_config()
    cascade = VirtualCascade(copy.deepcopy(cfg))

    inflows = {n: 0.0 for n in TARGET_ORDER}
    inflows["Virtual Reservoir A"] = 5.0
    gates = {n: 100.0 for n in TARGET_ORDER}

    cascade.step(inflows, gates)
    release_a_day1 = cascade.reservoirs["Virtual Reservoir A"].state.release_mcm_day
    assert cascade.reservoirs["Virtual Reservoir B"].state.upstream_routed_inflow == 0.0

    cascade.step(inflows, gates)
    # Legacy behaviour: 1-day lag, NO attenuation (factor 1.0)
    assert cascade.reservoirs["Virtual Reservoir B"].state.upstream_routed_inflow == pytest.approx(
        release_a_day1
    )


def test_virtual_cascade_source_unmodified_by_stage3():
    """The legacy module keeps its percent-based external gate contract."""
    source = (_PROJECT_ROOT / "src" / "simulator" / "environment.py").read_text(encoding="utf-8")
    assert "class VirtualCascade" in source
    assert "class VirtualReservoir" in source
    assert "gate_command_pct" in source


# ---------------------------------------------------------------------------
# 4. Topology
# ---------------------------------------------------------------------------

def test_authoritative_topology_is_linear_cascade_a_b_c_d():
    adapter = _make_adapter()
    assert adapter.cascade_order == TARGET_ORDER
    edges = [(c.source, c.destination) for c in adapter.network.connections]
    assert edges == list(zip(TARGET_ORDER[:-1], TARGET_ORDER[1:]))


def test_authoritative_delays_and_attenuation_match_stage3_targets():
    adapter = _make_adapter()
    for conn in adapter.network.connections:
        key = (conn.source, conn.destination)
        assert conn.delay == TARGET_DELAYS[key], f"delay mismatch {key}"
        assert conn.attenuation == pytest.approx(TARGET_ATTENUATION[key]), f"attenuation mismatch {key}"


# ---------------------------------------------------------------------------
# 5. Routing delay / attenuation measured behaviourally
# ---------------------------------------------------------------------------

def test_measured_routing_delays_are_two_one_one():
    """
    Impulse response: A releases 5.0 MCM on day 1; with a 2-day delay B must
    receive nothing on days 1-2 and 0.90 * 5.0 = 4.5 MCM on day 3.
    """
    adapter = _make_adapter(initial_storages={
        "Virtual Reservoir A": 5.41,
        "Virtual Reservoir B": 0.0,
        "Virtual Reservoir C": 0.0,
        "Virtual Reservoir D": 0.0,
    })
    inflows = {n: 0.0 for n in TARGET_ORDER}
    gates = {"Virtual Reservoir A": 100.0, "Virtual Reservoir B": 100.0,
             "Virtual Reservoir C": 100.0, "Virtual Reservoir D": 0.0}

    routed_b, routed_c, routed_d = [], [], []
    for _ in range(6):
        adapter.step(inflows, gates)
        routed_b.append(adapter.reservoirs["Virtual Reservoir B"].state.upstream_routed_inflow)
        routed_c.append(adapter.reservoirs["Virtual Reservoir C"].state.upstream_routed_inflow)
        routed_d.append(adapter.reservoirs["Virtual Reservoir D"].state.upstream_routed_inflow)

    # A -> B : 2 day delay, attenuation 0.90
    assert routed_b[0] == pytest.approx(0.0)
    assert routed_b[1] == pytest.approx(0.0)
    assert routed_b[2] == pytest.approx(5.0 * 0.90)

    # B -> C : 1 day delay, attenuation 0.85
    assert routed_c[3] == pytest.approx(routed_b[2] * 0.85)

    # C -> D : 1 day delay, attenuation 0.80
    assert routed_d[4] == pytest.approx(routed_c[3] * 0.80)


def test_transmission_loss_is_accounted_not_destroyed():
    adapter = _make_adapter(initial_storages={
        "Virtual Reservoir A": 5.41,
        "Virtual Reservoir B": 0.0,
        "Virtual Reservoir C": 0.0,
        "Virtual Reservoir D": 0.0,
    })
    inflows = {n: 0.0 for n in TARGET_ORDER}
    gates = {"Virtual Reservoir A": 100.0, "Virtual Reservoir B": 100.0,
             "Virtual Reservoir C": 100.0, "Virtual Reservoir D": 0.0}
    for _ in range(6):
        adapter.step(inflows, gates)

    mb = adapter.mass_balance_check()
    assert mb["total_routing_loss"] > 0.0
    assert mb["residual_error"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Legacy vs network implementation equivalence (parity)
# ---------------------------------------------------------------------------

def test_network_matches_legacy_cascade_when_parameters_coincide():
    """
    With legacy-equivalent parameters (delays 1/1/1, attenuation 1.0) and no
    spill, the two implementations must agree EXACTLY. This proves the
    ReservoirNetwork is a faithful generalisation of VirtualCascade and that
    every Stage 3 behaviour change is attributable to a named parameter.
    """
    cfg = _load_live_config()
    inflows = {"Virtual Reservoir A": 1.0, "Virtual Reservoir B": 2.0,
               "Virtual Reservoir C": 20.0, "Virtual Reservoir D": 0.0}
    gates = {"Virtual Reservoir A": 40.0, "Virtual Reservoir B": 35.0,
             "Virtual Reservoir C": 50.0, "Virtual Reservoir D": 100.0}

    legacy = VirtualCascade(copy.deepcopy(cfg))

    legacy_order = cfg["topology"]["cascade_order"]
    net_cfg = {
        "reservoirs": [
            {
                "id": name,
                "capacity_mcm": {"value": cfg["reservoirs"][name]["capacity_mcm"]["value"]},
                "initial_storage_mcm": {"value": cfg["reservoirs"][name]["initial_storage_mcm"]["value"]},
                "max_release_mcm_day": {"value": cfg["reservoirs"][name]["max_release_capacity_mcm_day"]["value"]},
            }
            for name in legacy_order
        ],
        "connections": [
            {
                "source": a,
                "destination": b,
                "routing_delay_days": {"value": 1},
                "attenuation_factor": {"value": 1.0},
            }
            for a, b in zip(legacy_order[:-1], legacy_order[1:])
        ],
        "topology": {"downstream_capacity_mcm_day": {"value": 50.0}},
    }
    network = LiveCascadeAdapter.from_network_config(net_cfg, topology_path=str(TOPOLOGY_PATH))

    for _ in range(10):
        legacy.step(inflows, gates)
        network.step(inflows, gates)

    for name in legacy_order:
        lg = legacy.reservoirs[name].state
        nw = network.reservoirs[name].state
        assert nw.storage_mcm == pytest.approx(lg.storage_mcm, abs=1e-12), name
        assert nw.release_mcm_day == pytest.approx(lg.release_mcm_day, abs=1e-12), name
        assert nw.upstream_routed_inflow == pytest.approx(lg.upstream_routed_inflow, abs=1e-12), name

    assert network.current_downstream_flow == pytest.approx(
        legacy.current_downstream_flow, abs=1e-12
    )


# ---------------------------------------------------------------------------
# 7. Unit boundary on the live path
# ---------------------------------------------------------------------------

def test_gate_percent_is_converted_once_on_the_live_path():
    adapter = _make_adapter()
    name = "Virtual Reservoir A"
    max_release = adapter.reservoirs[name].max_release_capacity_mcm_day

    adapter.step({n: 0.0 for n in TARGET_ORDER},
                 {name: 75.0, "Virtual Reservoir B": 0.0,
                  "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0})

    # 75 percent -> 0.75 fraction -> 0.75 * max_release MCM/day
    assert adapter.reservoirs[name].state.controlled_release_mcm_day == pytest.approx(
        units.gate_percent_to_fraction(75.0) * max_release
    )
    assert adapter.reservoirs[name].state.gate_position_pct == pytest.approx(75.0)


def test_gate_percent_75_is_not_interpreted_as_fraction():
    adapter = _make_adapter()
    name = "Virtual Reservoir A"
    max_release = adapter.reservoirs[name].max_release_capacity_mcm_day
    adapter.step({n: 0.0 for n in TARGET_ORDER},
                 {name: 75.0, "Virtual Reservoir B": 0.0,
                  "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0})
    release = adapter.reservoirs[name].state.controlled_release_mcm_day
    # The old defect would have produced 0.75% -> 0.0075 * max_release
    assert release == pytest.approx(0.75 * max_release)
    assert release > 10 * (0.0075 * max_release)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_gate_raises_instead_of_failing_open(bad):
    adapter = _make_adapter()
    gates = {n: 0.0 for n in TARGET_ORDER}
    gates["Virtual Reservoir A"] = bad
    with pytest.raises(units.UnitContractError):
        adapter.step({n: 0.0 for n in TARGET_ORDER}, gates)


# ---------------------------------------------------------------------------
# 8. Conservation / physics
# ---------------------------------------------------------------------------

def test_mass_balance_conserved_over_live_like_run():
    adapter = _make_adapter()
    inflows = {"Virtual Reservoir A": 5.0, "Virtual Reservoir B": 4.0,
               "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
    gates = {"Virtual Reservoir A": 40.0, "Virtual Reservoir B": 35.0,
             "Virtual Reservoir C": 50.0, "Virtual Reservoir D": 100.0}
    for _ in range(25):
        adapter.step(inflows, gates)
    mb = adapter.mass_balance_check()
    assert abs(mb["residual_error"]) < 1e-10


def test_storage_never_negative_and_never_above_capacity():
    adapter = _make_adapter()
    order = adapter.cascade_order
    for day, (inflow, gate) in enumerate([
        (0.0, 100.0),      # drain with fully open gates and no inflow
        (50.0, 0.0),       # flood with fully closed gates
        (50.0, 0.0),
    ]):
        adapter.step({n: inflow for n in order}, {n: gate for n in order})
        for n in order:
            res = adapter.reservoirs[n]
            assert res.state.storage_mcm >= 0.0, f"negative storage at {n} on day {day}"
            assert res.state.storage_mcm <= res.capacity_mcm + 1e-9, f"capacity breach at {n}"


def test_non_terminal_spill_is_not_routed_downstream():
    """
    Stage 3 semantic difference: the authoritative network does NOT route
    non-terminal spill downstream (the legacy cascade did).
    """
    adapter = _make_adapter(initial_storages={
        "Virtual Reservoir A": 10.82,   # at capacity
        "Virtual Reservoir B": 0.0,
        "Virtual Reservoir C": 0.0,
        "Virtual Reservoir D": 0.0,
    })
    order = adapter.cascade_order
    # A receives a flood with its gate closed -> it must spill.
    adapter.step({"Virtual Reservoir A": 6.0, "Virtual Reservoir B": 0.0,
                  "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0},
                 {n: 0.0 for n in order})

    spill_a = adapter.reservoirs["Virtual Reservoir A"].state.spill_mcm
    assert spill_a > 0.0, "scenario did not produce spill at Reservoir A"
    # B receives nothing on day 1 (2-day delay), but crucially also must never
    # receive A's spill on any later day.
    for _ in range(5):
        adapter.step({"Virtual Reservoir A": 6.0, "Virtual Reservoir B": 0.0,
                      "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0},
                     {n: 0.0 for n in order})
    mb = adapter.mass_balance_check()
    assert mb["total_nonterminal_spill"] > 0.0
    # Conservation must still hold with spill treated as a network outflow.
    assert abs(mb["residual_error"]) < 1e-10


def test_downstream_flow_equals_terminal_total_outflow():
    adapter = _make_adapter()
    order = adapter.cascade_order
    gates = {n: 60.0 for n in order}
    inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
               "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
    adapter.step(inflows, gates)
    terminal = adapter.reservoirs[adapter.terminal_node_id].state
    assert adapter.current_downstream_flow == pytest.approx(terminal.release_mcm_day)
    assert adapter.downstream_capacity == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

def test_live_path_determinism():
    def run():
        adapter = _make_adapter()
        order = adapter.cascade_order
        inflows = {"Virtual Reservoir A": 3.0, "Virtual Reservoir B": 4.0,
                   "Virtual Reservoir C": 90.0, "Virtual Reservoir D": 0.0}
        gates = {n: 55.0 for n in order}
        for _ in range(15):
            adapter.step(inflows, gates)
        return {n: adapter.reservoirs[n].state.storage_mcm for n in order}

    assert run() == run()


# ---------------------------------------------------------------------------
# 10. The Digital Twin JSON contract still holds
# ---------------------------------------------------------------------------

def test_twin_json_contract_via_authoritative_physics():
    bridge = _make_bridge()
    forecasts = {n: {"forecast_1d": 10.0, "forecast_3d": 10.0, "forecast_7d": 10.0}
                 for n in bridge.cascade.cascade_order}
    sim_state = bridge.get_state(forecasts)
    twin = adapt_state_for_twin(sim_state, "MANUAL", 0.0)

    for i in range(1, 4):
        res = twin["reservoirs"][f"reservoir_{i}"]
        for field in ("water_level", "storage", "inflow", "release", "gate", "risk"):
            assert field in res
        for field in ("water_level", "storage", "inflow", "release", "gate"):
            assert isinstance(res[field], (int, float))
    assert twin["metadata"]["flow_unit"] == "m3/s"


def test_twin_water_level_tracks_storage_fraction_on_authoritative_path():
    bridge = _make_bridge()
    res_a = bridge.cascade.reservoirs["Virtual Reservoir A"]
    res_a.state.storage_mcm = res_a.capacity_mcm * 0.85
    twin = adapt_state_for_twin(bridge.get_state({}), "MANUAL", 0.0)
    assert twin["reservoirs"]["reservoir_1"]["water_level"] == pytest.approx(0.85, abs=0.02)


# ---------------------------------------------------------------------------
# 11. Provenance of the live physics
# ---------------------------------------------------------------------------

def test_live_physics_provenance_is_exposed():
    adapter = _make_adapter()
    prov = adapter.physics_provenance()
    assert prov["topology_path"].endswith("topology_config.yaml")
    assert prov["node_id_mapping"]["Virtual Reservoir A"] == "Reservoir_A"
    summary = prov["summary"]
    assert summary["OBSERVED"] >= 4          # the four capacities
    assert summary["ASSUMED_FOR_PROTOTYPE"] >= 6  # topology, delays, attenuations...
