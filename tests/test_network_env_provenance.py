"""
Phase 15.1 — Provenance Tests
for the Deterministic Interconnected Reservoir Network Environment.

Test 9 as specified in the Phase 15.1 requirements.
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.reservoir_network import ReservoirNetwork
from src.network_env.provenance import ProvenanceLevel

# Path to the canonical config
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"


def test_provenance_all_parameters_classified():
    """
    TEST 9 — Every configured physical/network parameter must have
    a provenance classification.
    """
    net = ReservoirNetwork(config_path=str(TOPOLOGY_PATH))
    registry = net.provenance

    all_entries = registry.all_entries()

    # Must have entries for every reservoir and connection
    assert len(all_entries) > 0, "Provenance registry is empty"

    # Check that every entry has a valid ProvenanceLevel
    for key, prov in all_entries.items():
        assert prov.level in ProvenanceLevel, \
            f"Parameter '{key}' has invalid provenance level: {prov.level}"
        assert prov.source != "", f"Parameter '{key}' has empty source"
        assert prov.unit != "", f"Parameter '{key}' has empty unit"

    # Expect at least:
    #   4 reservoirs × 3 params (capacity, initial_storage, max_release) = 12
    #   3 connections × 2 params (delay, attenuation)                    = 6
    #   1 downstream_capacity                                             = 1
    #   1 network_topology                                                = 1
    # Total: 20
    assert len(all_entries) >= 20, \
        f"Expected ≥20 provenance entries, got {len(all_entries)}"

    print(f"TEST 9 PASSED — Provenance: {len(all_entries)} parameters classified")

    # Print summary
    summary = registry.summary()
    print(f"  Provenance summary: {summary}")

    assumed = registry.all_assumed()
    verified = registry.all_verified()
    print(f"  Assumed: {len(assumed)} parameters")
    print(f"  Observed/Verified: {len(verified)} parameters")

    # Verify that the topology itself is marked ASSUMED_FOR_PROTOTYPE
    topo_prov = registry.get("network_topology")
    assert topo_prov.level == ProvenanceLevel.ASSUMED_FOR_PROTOTYPE, \
        f"Network topology should be ASSUMED_FOR_PROTOTYPE, got {topo_prov.level}"
    print(f"  Network topology provenance: {topo_prov.level.value} ✓")

    # Verify that capacities are marked as OBSERVED
    cap_a = registry.get("Reservoir_A.capacity")
    assert cap_a.level == ProvenanceLevel.OBSERVED, \
        f"Reservoir_A capacity should be OBSERVED, got {cap_a.level}"
    print(f"  Reservoir_A capacity provenance: {cap_a.level.value} ✓")


def test_provenance_distinguishes_observed_from_assumed():
    """
    Verify that the configuration correctly marks some parameters
    as OBSERVED and others as ASSUMED_FOR_PROTOTYPE.
    """
    net = ReservoirNetwork(config_path=str(TOPOLOGY_PATH))
    registry = net.provenance

    # Capacities should be OBSERVED (derived from dataset)
    for res_id in ["Reservoir_A", "Reservoir_B", "Reservoir_C", "Reservoir_D"]:
        cap = registry.get(f"{res_id}.capacity")
        assert cap.level == ProvenanceLevel.OBSERVED, \
            f"{res_id} capacity should be OBSERVED"

    # Routing delays should be ASSUMED_FOR_PROTOTYPE
    for conn in ["Reservoir_A->Reservoir_B", "Reservoir_B->Reservoir_C", "Reservoir_C->Reservoir_D"]:
        delay = registry.get(f"{conn}.routing_delay")
        assert delay.level == ProvenanceLevel.ASSUMED_FOR_PROTOTYPE, \
            f"{conn} routing delay should be ASSUMED_FOR_PROTOTYPE"

    print("TEST 9b PASSED — Provenance correctly distinguishes OBSERVED from ASSUMED_FOR_PROTOTYPE")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 15.1 — Provenance Tests")
    print("=" * 60)
    test_provenance_all_parameters_classified()
    test_provenance_distinguishes_observed_from_assumed()
    print("=" * 60)
    print("ALL PROVENANCE TESTS PASSED.")
    print("=" * 60)
