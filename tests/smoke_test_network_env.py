"""
Phase 15.1 — Deterministic Smoke Test

Demonstrates water propagation through A → B → C → D
with exact timestep-by-timestep storage/inflow/release values.

This is NOT a scientific claim about real Kerala dams.
It is a mathematical verification of the network environment.
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.reservoir_network import ReservoirNetwork

# Use the canonical four-reservoir config
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"


def run_smoke_test():
    net = ReservoirNetwork(config_path=str(TOPOLOGY_PATH))

    print("=" * 80)
    print("Phase 15.1 — Deterministic Smoke Test")
    print("Network: Reservoir_A → Reservoir_B → Reservoir_C → Reservoir_D")
    print(f"Processing order: {net.processing_order}")
    print("=" * 80)

    # Print initial state
    print(f"\n{'='*80}")
    print("INITIAL STATE")
    print(f"{'='*80}")
    for nid in net.processing_order:
        node = net.nodes[nid]
        print(f"  {nid}: storage={node.state.storage:.2f} MCM, "
              f"capacity={node.capacity:.2f} MCM "
              f"({node.storage_fraction*100:.1f}%)")

    # Scenario: Inject 5 MCM/day into A for 10 days, gate A at 80%, others at 20%
    inflows = {"Reservoir_A": 5.0, "Reservoir_B": 0.0,
               "Reservoir_C": 0.0, "Reservoir_D": 0.0}
    gates = {"Reservoir_A": 0.8, "Reservoir_B": 0.5,
             "Reservoir_C": 0.3, "Reservoir_D": 0.1}

    print(f"\n{'='*80}")
    print("SCENARIO: 5 MCM/day external inflow into A only")
    print(f"Gates: A=80%, B=50%, C=30%, D=10%")
    print(f"Routing: A→B delay=2 atten=0.90, B→C delay=1 atten=0.85, C→D delay=1 atten=0.80")
    print(f"{'='*80}")

    header = (f"{'t':>3} | "
              f"{'A_stor':>8} {'A_in':>6} {'A_rel':>6} {'A_spl':>6} | "
              f"{'B_stor':>8} {'B_rte':>6} {'B_rel':>6} | "
              f"{'C_stor':>8} {'C_rte':>6} {'C_rel':>6} | "
              f"{'D_stor':>8} {'D_rte':>6} {'D_rel':>6} {'D_spl':>6} | "
              f"{'mb_err':>10}")
    print(header)
    print("-" * len(header))

    for t in range(1, 16):
        states = net.step(inflows, gates)
        mb = net.mass_balance_check()

        sa = states["Reservoir_A"]
        sb = states["Reservoir_B"]
        sc = states["Reservoir_C"]
        sd = states["Reservoir_D"]

        print(f"{t:3d} | "
              f"{sa.storage:8.2f} {sa.inflow_local:6.2f} {sa.controlled_release:6.2f} {sa.spill:6.2f} | "
              f"{sb.storage:8.2f} {sb.inflow_routed:6.2f} {sb.controlled_release:6.2f} | "
              f"{sc.storage:8.2f} {sc.inflow_routed:6.2f} {sc.controlled_release:6.2f} | "
              f"{sd.storage:8.2f} {sd.inflow_routed:6.2f} {sd.controlled_release:6.2f} {sd.spill:6.2f} | "
              f"{mb['residual_error']:10.2e}")

    # Final mass balance
    mb = net.mass_balance_check()
    print(f"\n{'='*80}")
    print("FINAL MASS BALANCE")
    print(f"{'='*80}")
    for k, v in mb.items():
        print(f"  {k:30s}: {v:12.4f}")

    print(f"\n{'='*80}")
    print("PROVENANCE SUMMARY")
    print(f"{'='*80}")
    summary = net.provenance.summary()
    for level, count in summary.items():
        print(f"  {level}: {count} parameters")

    assumed = net.provenance.all_assumed()
    print(f"\n  ASSUMED_FOR_PROTOTYPE parameters:")
    for key, prov in assumed.items():
        print(f"    {key}: {prov.value} {prov.unit} — {prov.source}")

    print(f"\n{'='*80}")
    assert abs(mb["residual_error"]) < 1e-8, f"MASS BALANCE FAILED: {mb['residual_error']}"
    print("SMOKE TEST PASSED — Mass balance verified.")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_smoke_test()
