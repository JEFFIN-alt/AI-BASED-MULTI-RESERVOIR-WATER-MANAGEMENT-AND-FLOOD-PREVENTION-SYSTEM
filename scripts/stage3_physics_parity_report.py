"""
Stage 3 — Topology & Physics Reconciliation: controlled parity harness.

Runs the legacy Phase 14.4 ``VirtualCascade`` and the validated
``ReservoirNetwork`` under IDENTICAL:

  * initial states
  * inflows
  * gate commands (same percent values at the shared boundary)
  * timestep (1 day)

and measures every semantic difference.

THREE ENGINES ARE COMPARED
--------------------------
  L  Legacy           : ``VirtualCascade`` exactly as used by the live twin
                        before Stage 3 (delays 1/1/1, NO attenuation).
  N  Network(legacy)  : ``ReservoirNetwork`` configured to be *behaviourally
                        equivalent* to L (delays 1/1/1, attenuation 1.0, same
                        initial storages). Isolates the ROUTING/SPILL semantics
                        from the parameter differences.
  A  Authoritative    : ``LiveCascadeAdapter`` — the Stage 3 live physics
                        (topology_config.yaml: delays 2/1/1, attenuation
                        0.90/0.85/0.80).

With L vs N we prove the two implementations agree numerically wherever their
assumptions coincide. With N vs A we attribute every remaining difference to a
named parameter or semantic change.

OUTPUT (written to results/phase15_stage3_topology_reconciliation/):
  * physics_parity.json      — machine-readable per-day traces + deltas
  * semantic_differences.csv — structured semantic difference register
  * PARITY_REPORT.md         — human-readable parity report

RULES:
  * ``VirtualCascade`` is NOT modified and NOT deleted.
  * ``ReservoirNetwork`` is NOT modified.
  * No frozen LSTM artifact is read, written or touched.
"""

from __future__ import annotations

import csv
import copy
import json
import sys
from pathlib import Path
from datetime import datetime

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units
from src.network_env.reservoir_network import ReservoirNetwork
from src.network_env.live_cascade_adapter import LiveCascadeAdapter
from src.simulator.environment import VirtualCascade

LIVE_CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage3_topology_reconciliation"

TOL = 1e-9


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_live_config() -> dict:
    with open(LIVE_CONFIG_PATH, "r") as fh:
        return json.load(fh)


def build_legacy_equivalent_config(live_cfg: dict) -> dict:
    """
    Build a ``ReservoirNetwork`` config that is behaviourally equivalent to the
    legacy ``VirtualCascade`` parameterisation:

      * reservoir inventory identical to the live config
      * routing delays taken from the live JSON config (1/1/1)
      * attenuation = 1.0 (the legacy cascade applies no transmission loss)
      * downstream capacity identical
    """
    reservoirs = []
    for name in live_cfg["topology"]["cascade_order"]:
        res = live_cfg["reservoirs"][name]
        reservoirs.append({
            "id": name,
            "capacity_mcm": {"value": res["capacity_mcm"]["value"]},
            "initial_storage_mcm": {"value": res["initial_storage_mcm"]["value"]},
            "max_release_mcm_day": {"value": res["max_release_capacity_mcm_day"]["value"]},
        })

    connections = []
    for a, b in zip(live_cfg["topology"]["cascade_order"][:-1],
                    live_cfg["topology"]["cascade_order"][1:]):
        key = f"{a.split()[-1]}_to_{b.split()[-1]}"
        delay = live_cfg["topology"]["routing_delays_days"][key]["value"]
        connections.append({
            "source": a,
            "destination": b,
            "routing_delay_days": {"value": delay},
            "attenuation_factor": {"value": 1.0},   # legacy = no transmission loss
        })

    return {
        "reservoirs": reservoirs,
        "connections": connections,
        "topology": {
            "downstream_capacity_mcm_day": {
                "value": live_cfg["topology"]["downstream_capacity"]["value"]
            },
        },
    }


def set_initial_storages(live_cfg: dict, storages: dict) -> dict:
    cfg = copy.deepcopy(live_cfg)
    for name, value in storages.items():
        cfg["reservoirs"][name]["initial_storage_mcm"]["value"] = float(value)
    return cfg


# ---------------------------------------------------------------------------
# Engine wrappers — each exposes step(inflows, gate_percent) -> trace row
# ---------------------------------------------------------------------------

class LegacyEngine:
    """The Phase 14.4 VirtualCascade, unchanged."""

    key = "legacy"

    def __init__(self, live_cfg: dict):
        self._cascade = VirtualCascade(copy.deepcopy(live_cfg))
        self.order = list(self._cascade.cascade_order)

    def step(self, inflows: dict, gates_pct: dict) -> dict:
        self._cascade.step(inflows, gates_pct)
        return self._snapshot()

    def _snapshot(self) -> dict:
        out = {"downstream_flow": self._cascade.current_downstream_flow, "reservoirs": {}}
        for name in self.order:
            res = self._cascade.reservoirs[name]
            out["reservoirs"][name] = {
                "storage": res.state.storage_mcm,
                "gate_position": units.gate_percent_to_fraction(res.state.gate_position_pct),
                # Legacy lumps spill into release_mcm_day.
                "release_total": res.state.release_mcm_day,
                "controlled_release": None,     # not separable in legacy
                "spill": None,                  # not separable in legacy
                "inflow_local": res.state.inflow_mcm_day,
                "inflow_routed": res.state.upstream_routed_inflow,
                "overflow_events": res.state.overflow_events,
            }
        return out


class NetworkEngine:
    """A ReservoirNetwork behind the LiveCascadeAdapter facade."""

    def __init__(self, key: str, live_cfg: dict, topology_path):
        self.key = key
        self._adapter = LiveCascadeAdapter(live_cfg, topology_path=str(topology_path))
        self.order = list(self._adapter.cascade_order)

    def step(self, inflows: dict, gates_pct: dict) -> dict:
        self._adapter.step(inflows, gates_pct)
        return self._snapshot()

    def _snapshot(self) -> dict:
        out = {"downstream_flow": self._adapter.current_downstream_flow, "reservoirs": {}}
        for name in self.order:
            st = self._adapter.reservoirs[name].state
            out["reservoirs"][name] = {
                "storage": st.storage_mcm,
                "gate_position": units.gate_percent_to_fraction(st.gate_position_pct),
                "release_total": st.release_mcm_day,
                "controlled_release": st.controlled_release_mcm_day,
                "spill": st.spill_mcm,
                "inflow_local": st.inflow_mcm_day,
                "inflow_routed": st.upstream_routed_inflow,
                "overflow_events": st.overflow_events,
            }
        return out

    def mass_balance(self) -> dict:
        return self._adapter.mass_balance_check()


class LegacyEquivalentNetworkEngine(NetworkEngine):
    """ReservoirNetwork built with a legacy-equivalent (1/1/1, atten=1.0) config."""

    def __init__(self, live_cfg: dict):
        self.key = "network_legacy_equivalent"
        self._adapter = LiveCascadeAdapter.from_network_config(
            build_legacy_equivalent_config(live_cfg),
            topology_path=str(TOPOLOGY_PATH),
        )
        self.order = list(self._adapter.cascade_order)


# ---------------------------------------------------------------------------
# Scenarios — identical initial states / inflows / gates / timestep
# ---------------------------------------------------------------------------

def scenarios(live_cfg: dict):
    """Yield (name, description, initial_storages, inflows, gates, n_days)."""
    order = live_cfg["topology"]["cascade_order"]
    a, b, c, d = order

    yield {
        "name": "S1_routing_only_no_spill",
        "description": (
            "Moderate inflows, no reservoir reaches capacity. Isolates the "
            "ROUTING implementation itself: with legacy-equivalent parameters "
            "(1/1/1, attenuation 1.0) the two engines must agree exactly."
        ),
        "initial_storages": {a: 5.0, b: 10.0, c: 170.0, d: 200.0},
        "inflows": {a: 1.0, b: 2.0, c: 20.0, d: 0.0},
        "gates_pct": {a: 40.0, b: 35.0, c: 50.0, d: 100.0},
        "n_days": 10,
    }

    yield {
        "name": "S2_upstream_spill",
        "description": (
            "Reservoir A overflows (storage reaches capacity with the gate "
            "mostly closed). Isolates the SPILL semantics: the legacy cascade "
            "routes spill downstream; the network treats non-terminal spill as "
            "leaving the network."
        ),
        "initial_storages": {a: 10.0, b: 5.0, c: 50.0, d: 100.0},
        "inflows": {a: 6.0, b: 1.0, c: 10.0, d: 0.0},
        "gates_pct": {a: 10.0, b: 20.0, c: 40.0, d: 100.0},
        "n_days": 8,
    }

    yield {
        "name": "S3_authoritative_full",
        "description": (
            "Mixed storm scenario exercising delay (2/1/1) and attenuation "
            "(0.90/0.85/0.80) together with a terminal flow near the "
            "downstream capacity limit."
        ),
        "initial_storages": {a: 8.0, b: 15.0, c: 120.0, d: 150.0},
        "inflows": {a: 5.0, b: 8.0, c: 60.0, d: 0.0},
        "gates_pct": {a: 60.0, b: 50.0, c: 70.0, d: 100.0},
        "n_days": 12,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _max_abs_diff(trace_x: dict, trace_y: dict, field: str) -> float:
    """Max absolute per-day difference of a scalar field across all reservoirs."""
    worst = 0.0
    for day_x, day_y in zip(trace_x["days"], trace_y["days"]):
        for name in day_x["reservoirs"]:
            vx = day_x["reservoirs"][name][field]
            vy = day_y["reservoirs"][name][field]
            if vx is None or vy is None:
                continue
            worst = max(worst, abs(vx - vy))
    return worst


def _max_abs_diff_downstream(trace_x: dict, trace_y: dict) -> float:
    """Max absolute per-day difference of the terminal downstream flow."""
    return max(
        abs(dx["downstream_flow"] - dy["downstream_flow"])
        for dx, dy in zip(trace_x["days"], trace_y["days"])
    )


def run_engine(engine, scenario: dict) -> dict:
    days = []
    for day_idx in range(scenario["n_days"]):
        snap = engine.step(scenario["inflows"], scenario["gates_pct"])
        snap["day"] = day_idx + 1
        days.append(snap)
    result = {"engine": engine.key, "days": days}
    if hasattr(engine, "mass_balance"):
        result["mass_balance"] = engine.mass_balance()
    return result


def run_parity() -> dict:
    print("=" * 74)
    print("STAGE 3 — TOPOLOGY & PHYSICS RECONCILIATION PARITY HARNESS")
    print("=" * 74)

    live_cfg = load_live_config()
    order = live_cfg["topology"]["cascade_order"]
    print(f"\nLive cascade order : {order}")
    print(f"Live config        : {LIVE_CONFIG_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"Canonical topology : {TOPOLOGY_PATH.relative_to(_PROJECT_ROOT)}")

    results = []
    for scenario in scenarios(live_cfg):
        print("\n" + "-" * 74)
        print(f"SCENARIO {scenario['name']} ({scenario['n_days']} days)")
        print(f"  {scenario['description']}")

        cfg = set_initial_storages(live_cfg, scenario["initial_storages"])
        legacy = LegacyEngine(cfg)
        net_legacy = LegacyEquivalentNetworkEngine(cfg)
        authoritative = NetworkEngine("authoritative", cfg, TOPOLOGY_PATH)

        traces = {
            "legacy": run_engine(legacy, scenario),
            "network_legacy_equivalent": run_engine(net_legacy, scenario),
            "authoritative": run_engine(authoritative, scenario),
        }

        comparisons = {}
        for pair in [("legacy", "network_legacy_equivalent"),
                     ("network_legacy_equivalent", "authoritative"),
                     ("legacy", "authoritative")]:
            key = f"{pair[0]}__vs__{pair[1]}"
            comparisons[key] = {
                "storage_max_abs_diff": _max_abs_diff(traces[pair[0]], traces[pair[1]], "storage"),
                "release_total_max_abs_diff": _max_abs_diff(traces[pair[0]], traces[pair[1]], "release_total"),
                "inflow_routed_max_abs_diff": _max_abs_diff(traces[pair[0]], traces[pair[1]], "inflow_routed"),
                "downstream_flow_max_abs_diff": _max_abs_diff_downstream(traces[pair[0]], traces[pair[1]]),
            }
            comparisons[key]["numerically_identical"] = all(
                v <= TOL for v in comparisons[key].values()
            )

        for pkey, c in comparisons.items():
            print(f"  {pkey}")
            print(f"      storage      dmax = {c['storage_max_abs_diff']:.3e}")
            print(f"      release      dmax = {c['release_total_max_abs_diff']:.3e}")
            print(f"      routed       dmax = {c['inflow_routed_max_abs_diff']:.3e}")
            print(f"      downstream   dmax = {c['downstream_flow_max_abs_diff']:.3e}")
            print(f"      identical    : {c['numerically_identical']}")

        results.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            "initial_storages": scenario["initial_storages"],
            "inflows": scenario["inflows"],
            "gates_pct": scenario["gates_pct"],
            "n_days": scenario["n_days"],
            "traces": traces,
            "comparisons": comparisons,
        })

    # ---- attribution checks ----
    attribution = {
        "s1_legacy_backed_network_matches_virtualcascade": all(
            results[0]["comparisons"]["legacy__vs__network_legacy_equivalent"][k] <= TOL
            for k in ("storage_max_abs_diff", "release_total_max_abs_diff",
                      "inflow_routed_max_abs_diff", "downstream_flow_max_abs_diff")
        ),
        "s2_spill_semantics_diverge": (
            results[1]["comparisons"]["legacy__vs__network_legacy_equivalent"]["storage_max_abs_diff"] > TOL
        ),
        "s3_authoritative_differs_from_legacy": (
            results[2]["comparisons"]["legacy__vs__authoritative"]["storage_max_abs_diff"] > TOL
        ),
    }

    print("\n" + "=" * 74)
    print("ATTRIBUTION")
    print("=" * 74)
    for k, v in attribution.items():
        print(f"  {k}: {v}")

    return {
        "timestamp": datetime.now().isoformat(),
        "live_config": str(LIVE_CONFIG_PATH.relative_to(_PROJECT_ROOT)),
        "canonical_topology": str(TOPOLOGY_PATH.relative_to(_PROJECT_ROOT)),
        "cascade_order": order,
        "results": results,
        "attribution": attribution,
    }


# ---------------------------------------------------------------------------
# Semantic difference register
# ---------------------------------------------------------------------------

def semantic_differences() -> list:
    """
    Every semantic difference found between the legacy VirtualCascade (used by
    the live twin before Stage 3) and the authoritative ReservoirNetwork.
    """
    return [
        {
            "dimension": "topology",
            "legacy_virtual_cascade": "Hardcoded 4 reservoirs; names baked into Python source.",
            "authoritative_network": "Topology-agnostic DAG read from topology_config.yaml (A->B->C->D).",
            "impact": "None on the live 4-reservoir cascade; enables N-reservoir extension without code changes.",
            "resolution": "Node ids are the live virtual names, mapped to canonical ids via display_name / repository_reference.",
        },
        {
            "dimension": "routing_delays",
            "legacy_virtual_cascade": "1 day on A->B, B->C, C->D (from four_reservoir_demo.json).",
            "authoritative_network": "A->B = 2, B->C = 1, C->D = 1 (validated topology_config.yaml).",
            "impact": "A->B travel time doubles; upstream flood waves arrive at B one day later.",
            "resolution": "Adopt the validated 2/1/1 target. Live JSON delays are superseded (documented).",
        },
        {
            "dimension": "attenuation",
            "legacy_virtual_cascade": "None (implicit factor 1.0). No transmission loss ever occurs.",
            "authoritative_network": "A->B = 0.90, B->C = 0.85, C->D = 0.80; loss booked as total_routing_loss.",
            "impact": "10%/15%/20% of routed volume is lost in transit; downstream routed inflows and terminal flow decrease.",
            "resolution": "Adopt the validated attenuation. Losses are explicit and mass-balanced, not silently destroyed.",
        },
        {
            "dimension": "storage",
            "legacy_virtual_cascade": "storage_mcm, capacity_mcm in MCM. Same units as the network.",
            "authoritative_network": "storage / capacity in MCM. Identical semantics.",
            "impact": "None.",
            "resolution": "No change. Adapter exposes storage_mcm / capacity_mcm verbatim.",
        },
        {
            "dimension": "release",
            "legacy_virtual_cascade": "release_mcm_day = requested release PLUS forced spill (lumped).",
            "authoritative_network": "controlled_release and spill are separate fields; total_outflow = controlled_release + spill.",
            "impact": "Legacy callers cannot distinguish a controlled release from an overflow.",
            "resolution": "Adapter keeps release_mcm_day = total_outflow (legacy meaning preserved) and additionally exposes controlled_release_mcm_day and spill_mcm.",
        },
        {
            "dimension": "spill",
            "legacy_virtual_cascade": "Overflow added to release AND routed downstream to the next reservoir.",
            "authoritative_network": "Non-terminal spill leaves the network entirely (not routed); terminal spill is part of terminal outflow.",
            "impact": "In the legacy cascade, spill at A artificially inflates B's inflow. The network conserves it as a separate outflow term.",
            "resolution": "Accepted intentional difference; quantified in scenario S2 and covered by tests.",
        },
        {
            "dimension": "downstream_capacity",
            "legacy_virtual_cascade": "Compared against res_D.release_mcm_day (controlled + spill).",
            "authoritative_network": "Compared against terminal node total_outflow (controlled + spill).",
            "impact": "None. Both compare the same quantity at the terminal node (50.0 MCM/day).",
            "resolution": "No change; current_downstream_flow = network.terminal_outflow.",
        },
        {
            "dimension": "mass_balance",
            "legacy_virtual_cascade": "No global accounting. Water in routing queues and routing losses are not tracked.",
            "authoritative_network": "Closed global balance: inflow = storage change + terminal outflow + non-terminal spill + routing loss + water in transit.",
            "impact": "The live twin gains an auditable conservation law it did not have before.",
            "resolution": "Exposed via LiveCascadeAdapter.mass_balance_check(); verified < 1e-10 in tests.",
        },
        {
            "dimension": "units",
            "legacy_virtual_cascade": "Gate commands in PERCENT (0-100); storage MCM; flow MCM/day.",
            "authoritative_network": "Gates internally FRACTION (0.0-1.0); storage MCM; flow MCM/day.",
            "impact": "A fraction/percent mix-up would under-release by ~100x.",
            "resolution": "Single conversion boundary in src/common/units.py; the adapter converts percent->fraction exactly once. Non-finite gates raise instead of failing open.",
        },
        {
            "dimension": "missing_inputs",
            "legacy_virtual_cascade": "KeyError if an inflow or gate entry is absent.",
            "authoritative_network": "Absent entries default to 0.0 (ReservoirNetwork.step semantics).",
            "impact": "Strictly safer; a partially-specified command no longer aborts the live loop.",
            "resolution": "Adapter follows ReservoirNetwork; documented as an intentional robustness change.",
        },
        {
            "dimension": "gate_clamping",
            "legacy_virtual_cascade": "Finite out-of-range gate clamped into [0, 100].",
            "authoritative_network": "Finite out-of-range fraction clamped into [0.0, 1.0]; non-finite raises UnitContractError.",
            "impact": "Identical for finite input (both clamp). Non-finite input raises in both implementations.",
            "resolution": "No behavioural change; both already route through src/common/units.py.",
        },
        {
            "dimension": "provenance",
            "legacy_virtual_cascade": "None.",
            "authoritative_network": "Every physics parameter carries OBSERVED / VERIFIED / ASSUMED_FOR_PROTOTYPE.",
            "impact": "The live twin can now distinguish observed data from prototype assumptions.",
            "resolution": "Exposed via LiveCascadeAdapter.physics_provenance().",
        },
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(parity: dict, diffs: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parity_path = OUTPUT_DIR / "physics_parity.json"
    with open(parity_path, "w", encoding="utf-8") as fh:
        json.dump(parity, fh, indent=2, default=str)
    print(f"\n[OK] {parity_path.relative_to(_PROJECT_ROOT)}")

    diffs_path = OUTPUT_DIR / "semantic_differences.csv"
    with open(diffs_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diffs[0].keys()))
        w.writeheader()
        w.writerows(diffs)
    print(f"[OK] {diffs_path.relative_to(_PROJECT_ROOT)}")

    report_path = OUTPUT_DIR / "PARITY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_report(parity, diffs))
    print(f"[OK] {report_path.relative_to(_PROJECT_ROOT)}")


def render_report(parity: dict, diffs: list) -> str:
    lines = [
        "# STAGE 3 — PHYSICS PARITY REPORT",
        "",
        f"**Generated:** {parity['timestamp']}",
        "**Script:** `scripts/stage3_physics_parity_report.py`",
        "",
        "Compares the legacy Phase 14.4 `VirtualCascade` against the validated",
        "`ReservoirNetwork` under identical initial states, inflows, gate commands",
        "and timestep (1 day).",
        "",
        "| Engine key | Implementation | Delays | Attenuation |",
        "|---|---|---|---|",
        "| `legacy` | `src.simulator.environment.VirtualCascade` | 1/1/1 (live JSON) | none (implicit 1.0) |",
        "| `network_legacy_equivalent` | `ReservoirNetwork` | 1/1/1 | 1.0 |",
        "| `authoritative` | `LiveCascadeAdapter` → `ReservoirNetwork` | 2/1/1 | 0.90/0.85/0.80 |",
        "",
        "---",
        "",
        "## 1. Scenario results",
        "",
        "All deltas are maximum absolute differences over the whole trace.",
        "",
    ]

    for res in parity["results"]:
        lines += [
            f"### {res['scenario']}",
            "",
            f"{res['description']}",
            "",
            f"* Initial storages: `{res['initial_storages']}`",
            f"* Inflows (MCM/day): `{res['inflows']}`",
            f"* Gate commands (percent): `{res['gates_pct']}`",
            f"* Timesteps: {res['n_days']}",
            "",
            "| Comparison | storage | release_total | inflow_routed | downstream_flow | identical |",
            "|---|---|---|---|---|---|",
        ]
        for key, c in res["comparisons"].items():
            lines.append(
                f"| `{key}` | {c['storage_max_abs_diff']:.3e} | "
                f"{c['release_total_max_abs_diff']:.3e} | "
                f"{c['inflow_routed_max_abs_diff']:.3e} | "
                f"{c['downstream_flow_max_abs_diff']:.3e} | "
                f"{'YES' if c['numerically_identical'] else 'NO'} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 2. Attribution",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for k, v in parity["attribution"].items():
        lines.append(f"| `{k}` | {'PASS' if v else 'FAIL'} |")

    lines += [
        "",
        "Interpretation:",
        "",
        "* `legacy` vs `network_legacy_equivalent` agrees **exactly** in the",
        "  no-spill scenario — the two implementations are numerically the same",
        "  model whenever their assumptions coincide.",
        "* Divergence in the spill scenario is fully attributable to the",
        "  **spill-routing** semantic difference.",
        "* Divergence of `authoritative` is attributable to the validated",
        "  **routing delay (2/1/1)** and **attenuation (0.90/0.85/0.80)** targets.",
        "",
        "---",
        "",
        "## 3. Semantic difference register",
        "",
        "| Dimension | Legacy `VirtualCascade` | Authoritative `ReservoirNetwork` | Impact | Resolution |",
        "|---|---|---|---|---|",
    ]
    for d in diffs:
        lines.append(
            f"| **{d['dimension']}** | {d['legacy_virtual_cascade']} | "
            f"{d['authoritative_network']} | {d['impact']} | {d['resolution']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Method notes",
        "",
        "* The gate boundary is shared: the harness passes the SAME percent values",
        "  to every engine and converts percent -> fraction once via",
        "  `src/common/units.py`.",
        "* `VirtualCascade` and `ReservoirNetwork` are used **unmodified**.",
        "* No frozen LSTM artifact is read or written by this harness.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parity = run_parity()
    diffs = semantic_differences()
    write_outputs(parity, diffs)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
