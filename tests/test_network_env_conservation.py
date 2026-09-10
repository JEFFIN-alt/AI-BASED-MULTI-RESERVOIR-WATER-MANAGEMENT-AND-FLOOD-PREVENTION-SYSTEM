"""
Phase 15.1 — Conservation, Physics, and Routing Tests
for the Deterministic Interconnected Reservoir Network Environment.

Tests 1-8 as specified in the Phase 15.1 requirements.
"""
import sys
from pathlib import Path
from copy import deepcopy

# Add project root to path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.reservoir_network import ReservoirNetwork, ReservoirNode


# ---------------------------------------------------------------------------
# Helper: minimal inline config builder
# ---------------------------------------------------------------------------
def _make_config(
    n_reservoirs=2,
    capacities=None,
    initial_storages=None,
    max_releases=None,
    connections=None,
    downstream_cap=50.0,
):
    """Build a minimal in-memory config dict for testing."""
    if capacities is None:
        capacities = [100.0] * n_reservoirs
    if initial_storages is None:
        initial_storages = [50.0] * n_reservoirs
    if max_releases is None:
        max_releases = [20.0] * n_reservoirs

    reservoirs = []
    for i in range(n_reservoirs):
        rid = f"R{i}"
        reservoirs.append({
            "id": rid,
            "capacity_mcm": {"value": capacities[i], "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
            "initial_storage_mcm": {"value": initial_storages[i], "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
            "max_release_mcm_day": {"value": max_releases[i], "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
        })

    if connections is None:
        connections = []
        for i in range(n_reservoirs - 1):
            connections.append({
                "source": f"R{i}",
                "destination": f"R{i+1}",
                "routing_delay_days": {"value": 1, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
                "attenuation_factor": {"value": 1.0, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
            })

    return {
        "reservoirs": reservoirs,
        "connections": connections,
        "topology": {
            "downstream_capacity_mcm_day": {"value": downstream_cap, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
            "topology_provenance": {"provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
        },
    }


# ---------------------------------------------------------------------------
# TEST 1 — MASS CONSERVATION
# ---------------------------------------------------------------------------
def test_mass_conservation():
    """
    For a closed network with known inflows/releases, prove:
    total_water_entering = total_retained + total_released + total_spilled
    within numerical tolerance.
    """
    cfg = _make_config(n_reservoirs=4, capacities=[100]*4,
                       initial_storages=[50]*4, max_releases=[20]*4)
    # Chain: R0 -> R1 -> R2 -> R3, delay=1, attenuation=1.0
    net = ReservoirNetwork(config_dict=cfg)

    inflows = {"R0": 10.0, "R1": 5.0, "R2": 3.0, "R3": 1.0}
    gates = {"R0": 0.5, "R1": 0.3, "R2": 0.2, "R3": 0.1}

    for _ in range(50):
        net.step(inflows, gates)

    mb = net.mass_balance_check()
    assert abs(mb["residual_error"]) < 1e-8, \
        f"Mass balance violated: residual = {mb['residual_error']}"
    print(f"TEST 1 PASSED — Mass conservation residual: {mb['residual_error']:.2e}")


# ---------------------------------------------------------------------------
# TEST 2 — ROUTING DELAY
# ---------------------------------------------------------------------------
def test_routing_delay():
    """
    If R0 releases water at t=0 and R0->R1 has delay=2:
    R1 must NOT receive that water at t=0 or t=1.
    It must arrive at t=2.
    """
    cfg = _make_config(n_reservoirs=2, capacities=[100, 100],
                       initial_storages=[50, 0], max_releases=[50, 50],
                       connections=[{
                           "source": "R0", "destination": "R1",
                           "routing_delay_days": {"value": 2, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
                           "attenuation_factor": {"value": 1.0, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
                       }])

    net = ReservoirNetwork(config_dict=cfg)

    # t=1: inject inflow into R0 and fully open gate
    states = net.step({"R0": 10.0, "R1": 0.0}, {"R0": 1.0, "R1": 0.0})
    release_r0_t1 = states["R0"].controlled_release
    assert release_r0_t1 > 0, "R0 should release water"
    assert states["R1"].inflow_routed == 0.0, "R1 should NOT receive water at t=1"

    # t=2: no new inflow
    states = net.step({"R0": 0.0, "R1": 0.0}, {"R0": 1.0, "R1": 0.0})
    assert states["R1"].inflow_routed == 0.0, "R1 should NOT receive water at t=2 (delay=2)"

    # t=3: the water from t=1 should arrive now
    states = net.step({"R0": 0.0, "R1": 0.0}, {"R0": 1.0, "R1": 0.0})
    assert states["R1"].inflow_routed == release_r0_t1, \
        f"R1 should receive {release_r0_t1} at t=3, got {states['R1'].inflow_routed}"

    print(f"TEST 2 PASSED — Routing delay: released {release_r0_t1:.2f} at t=1, arrived at R1 at t=3 (delay=2)")


# ---------------------------------------------------------------------------
# TEST 3 — ATTENUATION
# ---------------------------------------------------------------------------
def test_attenuation():
    """
    If routing factor is 0.8 and R0 sends 100 units:
    R1 should receive 80 units after the configured delay.
    """
    cfg = _make_config(n_reservoirs=2, capacities=[200, 200],
                       initial_storages=[100, 0], max_releases=[100, 100],
                       connections=[{
                           "source": "R0", "destination": "R1",
                           "routing_delay_days": {"value": 1, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
                           "attenuation_factor": {"value": 0.8, "provenance": "ASSUMED_FOR_PROTOTYPE", "source": "test"},
                       }])

    net = ReservoirNetwork(config_dict=cfg)

    # t=1: R0 releases water
    states = net.step({"R0": 0.0, "R1": 0.0}, {"R0": 1.0, "R1": 0.0})
    released = states["R0"].controlled_release  # should be 100

    # t=2: water arrives at R1, attenuated
    states = net.step({"R0": 0.0, "R1": 0.0}, {"R0": 0.0, "R1": 0.0})
    expected = released * 0.8
    assert abs(states["R1"].inflow_routed - expected) < 1e-8, \
        f"Expected {expected}, got {states['R1'].inflow_routed}"

    # Check that routing loss is accounted for
    mb = net.mass_balance_check()
    assert mb["total_routing_loss"] > 0, "Routing loss should be positive"
    assert abs(mb["residual_error"]) < 1e-8, "Mass balance should hold even with attenuation"

    print(f"TEST 3 PASSED — Attenuation: sent {released:.2f}, received {states['R1'].inflow_routed:.2f} "
          f"(factor=0.8), routing loss={mb['total_routing_loss']:.2f}")


# ---------------------------------------------------------------------------
# TEST 4 — CAPACITY
# ---------------------------------------------------------------------------
def test_capacity_overflow():
    """
    If storage exceeds capacity, storage must be capped at capacity
    and the excess must be classified as spill.
    """
    cfg = _make_config(n_reservoirs=1, capacities=[10.0],
                       initial_storages=[9.0], max_releases=[1.0],
                       connections=[])

    net = ReservoirNetwork(config_dict=cfg)

    # Inject 5 MCM with gate closed → storage would be 14, capacity=10
    states = net.step({"R0": 5.0}, {"R0": 0.0})

    assert states["R0"].storage == 10.0, f"Storage should be capped at 10.0, got {states['R0'].storage}"
    assert states["R0"].spill == 4.0, f"Spill should be 4.0, got {states['R0'].spill}"

    print(f"TEST 4 PASSED — Capacity: storage capped at {states['R0'].storage}, spill={states['R0'].spill}")


# ---------------------------------------------------------------------------
# TEST 5 — NEGATIVE STORAGE
# ---------------------------------------------------------------------------
def test_no_negative_storage():
    """A reservoir must never have negative storage."""
    cfg = _make_config(n_reservoirs=1, capacities=[10.0],
                       initial_storages=[2.0], max_releases=[100.0],
                       connections=[])

    net = ReservoirNetwork(config_dict=cfg)

    # Gate fully open, max_release=100, but only 2 + 1 = 3 MCM available
    states = net.step({"R0": 1.0}, {"R0": 1.0})

    assert states["R0"].storage >= 0.0, f"Negative storage: {states['R0'].storage}"
    assert states["R0"].controlled_release == 3.0, \
        f"Release should be 3.0 (all available), got {states['R0'].controlled_release}"

    print(f"TEST 5 PASSED — No negative storage: storage={states['R0'].storage}, release={states['R0'].controlled_release}")


# ---------------------------------------------------------------------------
# TEST 6 — GATE LIMITS
# ---------------------------------------------------------------------------
def test_gate_limits():
    """Gate positions outside [0,1] must be safely clipped."""
    cfg = _make_config(n_reservoirs=1, capacities=[100.0],
                       initial_storages=[50.0], max_releases=[20.0],
                       connections=[])

    net = ReservoirNetwork(config_dict=cfg)

    # Over-range gate
    states = net.step({"R0": 10.0}, {"R0": 5.0})  # 5.0 > 1.0
    assert states["R0"].gate_position == 1.0, f"Gate should clip to 1.0, got {states['R0'].gate_position}"

    # Under-range gate
    net.reset()
    states = net.step({"R0": 10.0}, {"R0": -2.0})  # -2.0 < 0.0
    assert states["R0"].gate_position == 0.0, f"Gate should clip to 0.0, got {states['R0'].gate_position}"

    print(f"TEST 6 PASSED — Gate limits: clipped to [0.0, 1.0]")


# ---------------------------------------------------------------------------
# TEST 7 — RELEASE LIMIT
# ---------------------------------------------------------------------------
def test_release_limit():
    """Release cannot exceed maximum gate release or available water."""
    cfg = _make_config(n_reservoirs=1, capacities=[100.0],
                       initial_storages=[50.0], max_releases=[20.0],
                       connections=[])

    net = ReservoirNetwork(config_dict=cfg)

    # Fully open gate: release = 1.0 * 20 = 20 MCM
    states = net.step({"R0": 10.0}, {"R0": 1.0})
    assert states["R0"].controlled_release == 20.0, \
        f"Release should be 20.0 (max), got {states['R0'].controlled_release}"

    # Now drain: storage ≈ 40, inflow = 0, gate open → release = min(20, 40) = 20
    states = net.step({"R0": 0.0}, {"R0": 1.0})
    assert states["R0"].controlled_release == 20.0

    # Keep draining
    states = net.step({"R0": 0.0}, {"R0": 1.0})  # 20 left
    assert states["R0"].controlled_release == 20.0

    states = net.step({"R0": 0.0}, {"R0": 1.0})  # 0 left, try to release 20
    assert states["R0"].controlled_release == 0.0, \
        f"Should release 0 when empty, got {states['R0'].controlled_release}"

    print(f"TEST 7 PASSED — Release limit: capped at max_release and available water")


# ---------------------------------------------------------------------------
# TEST 8 — CASCADE PROPAGATION
# ---------------------------------------------------------------------------
def test_cascade_propagation():
    """
    Demonstrate: A release → delayed B inflow → B release → delayed C inflow → ...
    """
    cfg = _make_config(n_reservoirs=4, capacities=[100]*4,
                       initial_storages=[50, 0, 0, 0], max_releases=[50]*4)
    # Default connections: R0->R1->R2->R3, delay=1, attenuation=1.0

    net = ReservoirNetwork(config_dict=cfg)

    # t=1: Inject into R0, open R0 gate
    s = net.step({"R0": 10.0, "R1": 0.0, "R2": 0.0, "R3": 0.0},
                 {"R0": 1.0, "R1": 1.0, "R2": 1.0, "R3": 1.0})
    r0_release = s["R0"].controlled_release
    assert r0_release > 0, "R0 should release"
    assert s["R1"].inflow_routed == 0.0, "R1 shouldn't receive yet (delay=1)"

    # t=2: R0's release arrives at R1
    s = net.step({"R0": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0},
                 {"R0": 0.0, "R1": 1.0, "R2": 1.0, "R3": 1.0})
    assert s["R1"].inflow_routed == r0_release, \
        f"R1 should receive {r0_release}, got {s['R1'].inflow_routed}"
    r1_release = s["R1"].controlled_release
    assert r1_release > 0, "R1 should release the received water"
    assert s["R2"].inflow_routed == 0.0, "R2 shouldn't receive yet"

    # t=3: R1's release arrives at R2
    s = net.step({"R0": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0},
                 {"R0": 0.0, "R1": 0.0, "R2": 1.0, "R3": 1.0})
    assert s["R2"].inflow_routed == r1_release, \
        f"R2 should receive {r1_release}, got {s['R2'].inflow_routed}"
    r2_release = s["R2"].controlled_release
    assert s["R3"].inflow_routed == 0.0, "R3 shouldn't receive yet"

    # t=4: R2's release arrives at R3
    s = net.step({"R0": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0},
                 {"R0": 0.0, "R1": 0.0, "R2": 0.0, "R3": 1.0})
    assert s["R3"].inflow_routed == r2_release, \
        f"R3 should receive {r2_release}, got {s['R3'].inflow_routed}"

    print(f"TEST 8 PASSED — Cascade: R0 released {r0_release:.2f} → R1 → R2 → R3 propagated correctly")


# ---------------------------------------------------------------------------
# TEST 10 — REPRODUCIBILITY
# ---------------------------------------------------------------------------
def test_reproducibility():
    """Same config and inputs must produce identical results."""
    cfg = _make_config(n_reservoirs=4, capacities=[100]*4,
                       initial_storages=[50]*4, max_releases=[20]*4)

    inflows = {"R0": 10.0, "R1": 5.0, "R2": 3.0, "R3": 1.0}
    gates = {"R0": 0.5, "R1": 0.3, "R2": 0.2, "R3": 0.1}

    results_a = []
    net_a = ReservoirNetwork(config_dict=deepcopy(cfg))
    for _ in range(20):
        s = net_a.step(inflows, gates)
        results_a.append({nid: st.storage for nid, st in s.items()})

    results_b = []
    net_b = ReservoirNetwork(config_dict=deepcopy(cfg))
    for _ in range(20):
        s = net_b.step(inflows, gates)
        results_b.append({nid: st.storage for nid, st in s.items()})

    for t in range(20):
        for nid in results_a[t]:
            assert abs(results_a[t][nid] - results_b[t][nid]) < 1e-12, \
                f"Reproducibility failed at t={t}, {nid}"

    print(f"TEST 10 PASSED — Reproducibility: 20 timesteps, 4 reservoirs, identical")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 15.1 — Conservation & Physics Tests")
    print("=" * 60)
    test_mass_conservation()
    test_routing_delay()
    test_attenuation()
    test_capacity_overflow()
    test_no_negative_storage()
    test_gate_limits()
    test_release_limit()
    test_cascade_propagation()
    test_reproducibility()
    print("=" * 60)
    print("ALL 9 TESTS PASSED.")
    print("=" * 60)
