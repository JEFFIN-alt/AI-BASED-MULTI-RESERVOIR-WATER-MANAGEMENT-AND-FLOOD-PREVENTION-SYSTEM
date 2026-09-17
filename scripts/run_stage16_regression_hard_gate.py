"""
Stage 16 — PHASE 15.3 REGRESSION HARD GATE: evidence.

Proves that the engineering/integration work of Stages 0-15 did NOT change the
validated Phase 15.3 research result.

What this script does:

  1. hashes the FROZEN V3 artifacts and compares them with the manifest recorded
     at validation time (``results/phase15_v3_validation/v3_integrity_check.json``),
     handling the known Windows CRLF case for the prediction CSV;
  2. hashes the six protected Phase 15.3 outputs BEFORE the run;
  3. runs the canonical reproduction
     (``scripts/stage3_phase15_3_reproduction.py``) as a subprocess and records
     its exit code and runtime;
  4. checks the three re-derived outputs are BYTE-IDENTICAL to the protected ones;
  5. asserts the HARD canonical metrics (baseline / MPC / mass-balance residual);
  6. verifies the topology, the 6^4 = 1296 candidate space, and that Reservoir D
     is genuinely optimised (not pinned);
  7. proves the Phase 15.3 scientific sources are untouched (git diff vs HEAD +
     last-commit provenance + content hashes);
  8. proves the live provenance gate was NOT relaxed to make the gate pass;
  9. re-hashes the protected outputs and the frozen artifacts AFTER everything.

READ-ONLY. This script writes only its own evidence JSON.

OUTPUT
    results/phase15_stage16_regression_hard_gate/stage16_regression_evidence.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage16_regression_hard_gate"
OUTPUT_PATH = OUTPUT_DIR / "stage16_regression_evidence.json"

PROTECTED_DIR = _PROJECT_ROOT / "results" / "phase15_v3_validation"
REPRO_DIR = (_PROJECT_ROOT / "results" / "phase15_stage3_reproduction"
             / "phase15_3_reproduction")
REPRO_SCRIPT = _PROJECT_ROOT / "scripts" / "stage3_phase15_3_reproduction.py"
MANIFEST = PROTECTED_DIR / "v3_integrity_check.json"
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"

#: Protected Phase 15.3 outputs (the validated research result).
PROTECTED_FILES = (
    "validation_metrics.csv",
    "daily_simulation_baseline.csv",
    "daily_simulation_mpc.csv",
    "provenance_audit.json",
    "v3_integrity_check.json",
    "PHASE_15_3_V3_VALIDATION_REPORT.md",
)
#: The three re-derived outputs that must be byte-identical.
FIDELITY_FILES = (
    "validation_metrics.csv",
    "daily_simulation_baseline.csv",
    "daily_simulation_mpc.csv",
)

#: The scientific computation behind Phase 15.3. Stages 10-15 must not have
#: touched ANY of these.
SCIENTIFIC_SOURCES = (
    "src/controller/mpc_controller.py",
    "src/controller/safety.py",
    "src/controller/objective.py",
    "src/controller/baseline_controller.py",
    "src/network_env/reservoir_network.py",
    "src/network_env/v3_forecast_adapter.py",
    "src/network_env/topology_config.yaml",
    "scripts/run_phase15_3_validation.py",
    "scripts/stage3_phase15_3_reproduction.py",
)

#: HARD canonical Phase 15.3 metrics (from the protected validation_metrics.csv).
CANONICAL_METRICS = {
    "overflow_events": ("7", "0"),
    "overflow_volume_mcm": ("10.75003168999999", "0.0"),
    "ds_violations": ("8", "0"),
    "peak_ds_flow": ("60.0", "30.0"),
    "total_release_mcm": ("2017.25", "2031.0"),
    "total_steps": ("74", "74"),
    "mass_balance_residual": ("-6.821210263296962e-13", "-6.821210263296962e-13"),
}
CANONICAL_RESIDUAL = -6.821210263296962e-13
PHYSICAL_DELAYS = [2, 1, 1]
PHYSICAL_ATTENUATION = [0.90, 0.85, 0.80]
EXPECTED_NODE_IDS = ["Reservoir_A", "Reservoir_B", "Reservoir_C", "Reservoir_D"]
RESERVOIR_NAMES = {"Reservoir_A": "Anayirankal", "Reservoir_B": "Ponmudi",
                   "Reservoir_C": "Idamalayar", "Reservoir_D": "Idukki"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_file_lf(path: Path) -> str:
    """Hash with CRLF normalised to LF (the documented Windows caveat)."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def hash_protected() -> dict:
    return {name: sha256_file(PROTECTED_DIR / name) for name in PROTECTED_FILES}


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(_PROJECT_ROOT),
                            capture_output=True, text=True)
    return (result.stdout or "").strip()


def git_clean(path: str) -> bool:
    """True when the working tree matches HEAD for this path."""
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path],
                            cwd=str(_PROJECT_ROOT), capture_output=True)
    return result.returncode == 0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 16 — PHASE 15.3 REGRESSION HARD GATE")
    print("=" * 78)

    evidence: dict = {"stage": 16, "timestamp": datetime.now().isoformat()}

    # ==================================================================
    # [1] FROZEN V3 ARTIFACTS vs the recorded manifest
    # ==================================================================
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen_checks = {}
    for rel, record in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel
        raw = sha256_file(path)
        lf = sha256_file_lf(path)
        expected = record["sha256"]
        frozen_checks[rel.replace("\\", "/")] = {
            "expected_sha256": expected,
            "actual_sha256_raw": raw,
            "actual_sha256_lf_normalised": lf,
            "size_bytes_expected": record["size_bytes"],
            "size_bytes_actual": path.stat().st_size,
            "match_raw": raw == expected,
            "match_lf_normalised": lf == expected,
            "verdict": ("MATCH" if raw == expected
                        else "MATCH (LF-NORMALISED)" if lf == expected
                        else "MISMATCH"),
        }
    print("\n[1] Frozen V3 artifacts vs manifest")
    for rel, info in frozen_checks.items():
        print(f"    {rel}")
        print(f"      expected {info['expected_sha256'][:32]}...")
        print(f"      actual   {info['actual_sha256_raw'][:32]}...  -> {info['verdict']}")

    # ==================================================================
    # [2] PROTECTED PHASE 15.3 OUTPUTS — before
    # ==================================================================
    protected_before = hash_protected()
    print(f"\n[2] Protected Phase 15.3 outputs: {len(protected_before)} files hashed")
    for name, digest in protected_before.items():
        print(f"    {name:<42} {digest[:24]}...")

    # ==================================================================
    # [3] CANONICAL REPRODUCTION
    # ==================================================================
    print(f"\n[3] Running the canonical reproduction: "
          f"{REPRO_SCRIPT.relative_to(_PROJECT_ROOT)}")
    t0 = time.perf_counter()
    run = subprocess.run([sys.executable, str(REPRO_SCRIPT)],
                         cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
    runtime_s = time.perf_counter() - t0
    stdout_tail = (run.stdout or "").strip().splitlines()[-14:]
    print(f"    exit code: {run.returncode} · runtime: {runtime_s:.1f} s")
    for line in stdout_tail:
        print(f"      | {line}")

    reproduction = {
        "command": "py scripts/stage3_phase15_3_reproduction.py",
        "exit_code": run.returncode,
        "runtime_s": round(runtime_s, 3),
        "stdout_tail": stdout_tail,
    }

    # ==================================================================
    # [4] OUTPUT IDENTITY (byte-for-byte)
    # ==================================================================
    fidelity = {}
    for name in FIDELITY_FILES:
        protected = PROTECTED_DIR / name
        reproduced = REPRO_DIR / name
        h_prot = sha256_file(protected) if protected.exists() else None
        h_repro = sha256_file(reproduced) if reproduced.exists() else None
        fidelity[name] = {
            "protected_sha256": h_prot,
            "reproduced_sha256": h_repro,
            "identical": h_prot is not None and h_prot == h_repro,
        }
    identical_all = all(v["identical"] for v in fidelity.values())
    print(f"\n[4] Output identity           : {identical_all} "
          f"({sum(v['identical'] for v in fidelity.values())}/{len(fidelity)} byte-identical)")

    # ==================================================================
    # [5] HARD CANONICAL METRICS
    # ==================================================================
    metrics_rows = {}
    with open(REPRO_DIR / "validation_metrics.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            metrics_rows[row["metric"]] = row

    hard_metrics = {}
    for metric, (expected_bl, expected_mpc) in CANONICAL_METRICS.items():
        row = metrics_rows[metric]
        hard_metrics[metric] = {
            "expected_baseline": expected_bl,
            "actual_baseline": row["baseline"],
            "baseline_ok": row["baseline"] == expected_bl,
            "expected_mpc": expected_mpc,
            "actual_mpc": row["mpc"],
            "mpc_ok": row["mpc"] == expected_mpc,
        }
    metrics_exact = all(v["baseline_ok"] and v["mpc_ok"] for v in hard_metrics.values())
    residual_bl = float(metrics_rows["mass_balance_residual"]["baseline"])
    residual_mpc = float(metrics_rows["mass_balance_residual"]["mpc"])
    residual_exact = (residual_bl == CANONICAL_RESIDUAL
                      and residual_mpc == CANONICAL_RESIDUAL)
    print(f"\n[5] Hard canonical metrics    : exact={metrics_exact}")
    for metric, info in hard_metrics.items():
        print(f"    {metric:<26} baseline {info['actual_baseline']:<20} "
              f"mpc {info['actual_mpc']:<20} {'OK' if info['baseline_ok'] and info['mpc_ok'] else 'FAIL'}")
    print(f"    mass-balance residual exact: {residual_exact} "
          f"({residual_bl!r} / {residual_mpc!r} vs canonical {CANONICAL_RESIDUAL!r})")

    # ==================================================================
    # [6] TOPOLOGY VERIFICATION
    # ==================================================================
    import copy

    import yaml

    from src.network_env.reservoir_network import ReservoirNetwork

    base_cfg = yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    net = ReservoirNetwork(config_dict=copy.deepcopy(base_cfg))
    connections = [{"source": c.source, "destination": c.destination,
                    "delay": c.delay, "attenuation": c.attenuation}
                   for c in net.connections]
    topology = {
        "topology_file": "src/network_env/topology_config.yaml",
        "node_ids": list(net.processing_order),
        "expected_node_ids": EXPECTED_NODE_IDS,
        "reservoir_names": RESERVOIR_NAMES,
        "connections": connections,
        "delays": [c["delay"] for c in connections],
        "delays_ok": [c["delay"] for c in connections] == PHYSICAL_DELAYS,
        "attenuation": [c["attenuation"] for c in connections],
        "attenuation_ok": all(abs(c["attenuation"] - a) < 1e-12
                              for c, a in zip(connections, PHYSICAL_ATTENUATION)),
        "chain_ok": [(c["source"], c["destination"]) for c in connections] ==
                    [("Reservoir_A", "Reservoir_B"), ("Reservoir_B", "Reservoir_C"),
                     ("Reservoir_C", "Reservoir_D")],
        "terminal_node": net._terminal_node_id,
        "downstream_capacity_mcm_day": float(net.downstream_capacity),
        "node_ids_ok": list(net.processing_order) == EXPECTED_NODE_IDS,
    }
    print(f"\n[6] Topology                  : {topology['node_ids']} -> "
          f"{topology['terminal_node']}")
    print(f"    delays {topology['delays']} ok={topology['delays_ok']} · "
          f"attenuation {topology['attenuation']} ok={topology['attenuation_ok']}")
    print(f"    downstream capacity {topology['downstream_capacity_mcm_day']} MCM/day")

    # ==================================================================
    # [7] CANDIDATE SPACE + D OPTIMISATION
    # ==================================================================
    from src.controller.mpc_controller import MPCConfig

    config = MPCConfig()
    gate_levels = list(config.gate_levels)
    candidate_space = {
        "gate_levels": gate_levels,
        "levels_per_reservoir": len(gate_levels),
        "reservoirs": len(EXPECTED_NODE_IDS),
        "computed_candidates": len(gate_levels) ** len(EXPECTED_NODE_IDS),
        "expected_candidates": 1296,
        "lookahead_steps": config.lookahead_steps,
        "max_gate_change": config.max_gate_change,
    }
    candidate_space["is_1296"] = candidate_space["computed_candidates"] == 1296

    mpc_rows = list(csv.DictReader(
        (REPRO_DIR / "daily_simulation_mpc.csv").open(encoding="utf-8")))
    gates_by_reservoir = {
        nid: sorted({float(r[f"{nid}_gate"]) for r in mpc_rows})
        for nid in EXPECTED_NODE_IDS
    }
    candidates_logged = sorted({r["candidates_evaluated"] for r in mpc_rows})
    d_gates = gates_by_reservoir["Reservoir_D"]
    d_optimisation = {
        "days": len(mpc_rows),
        "candidates_evaluated_logged": candidates_logged,
        "candidates_are_1296_every_day": candidates_logged == ["1296"],
        "forecast_used_days": sum(1 for r in mpc_rows if r["forecast_used"] == "True"),
        "mpc_status_values": sorted({r["mpc_status"] for r in mpc_rows}),
        "gates_by_reservoir": gates_by_reservoir,
        "reservoir_d_gate_values": d_gates,
        "reservoir_d_is_optimised_not_pinned": len(d_gates) > 1,
        "all_four_reservoirs_vary": all(
            len(v) > 1 for v in gates_by_reservoir.values()
        ),
        "safety_layer_corrected_days": sum(
            1 for r in mpc_rows if r["mpc_status"] == "CORRECTED"),
    }
    print(f"\n[7] Candidate space           : {len(gate_levels)}^{len(EXPECTED_NODE_IDS)} = "
          f"{candidate_space['computed_candidates']} "
          f"(logged per day: {candidates_logged})")
    print(f"    D gate values             : {d_gates} -> not pinned="
          f"{d_optimisation['reservoir_d_is_optimised_not_pinned']}")
    print(f"    MPC statuses              : {d_optimisation['mpc_status_values']} "
          f"(SafetyLayer corrected {d_optimisation['safety_layer_corrected_days']} days)")

    # ==================================================================
    # [8] SOURCE INTEGRITY
    # ==================================================================
    sources = {}
    for rel in SCIENTIFIC_SOURCES:
        path = _PROJECT_ROOT / rel
        sources[rel] = {
            "sha256": sha256_file(path),
            "clean_vs_HEAD": git_clean(rel),
            "last_commit": git("log", "-1", "--format=%h %ad %s",
                               "--date=short", "--", rel),
        }
    all_clean = all(v["clean_vs_HEAD"] for v in sources.values())
    print(f"\n[8] Scientific sources        : {len(sources)} files · "
          f"all clean vs HEAD = {all_clean}")
    for rel, info in sources.items():
        print(f"    {'CLEAN' if info['clean_vs_HEAD'] else 'DIRTY'}  {rel}"
              f"  (last: {info['last_commit'].split()[0] if info['last_commit'] else '?'})")

    # ==================================================================
    # [9] LIVE PROVENANCE GATE NOT RELAXED
    # ==================================================================
    from fastapi.testclient import TestClient

    from src.dashboard.api import state_manager
    from src.dashboard.api.app import app

    client = TestClient(app)
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    for _ in range(8):
        client.post("/api/simulation/step")
    client.post("/api/controller/mode", json={"mode": "AI"})
    state_manager.sim_state.step()
    live_state = client.get("/api/state").json()
    live_gate = {
        "mode": "AI",
        "forecast_control_eligible": live_state["control"]["forecast_control_eligible"],
        "controller_status": live_state["control"]["controller_status"],
        "final_safe_control_action_source": live_state["control"]["final_safe_control_action_source"],
        "blocked_reason": live_state["control"]["blocked_reason"],
        "live_forecasts_are_validated": live_state["forecast_provenance"]["live_forecasts_are_validated"],
        "gate_still_blocks": (
            live_state["control"]["forecast_control_eligible"] is False
            and live_state["control"]["final_safe_control_action_source"] == "HELD_CURRENT_GATES"
        ),
        "note": (
            "Phase 15.3 regression validates the HISTORICAL research pipeline. "
            "It does NOT mean the live Digital Twin has a validated real-world "
            "forecast source, and the live provenance gate was not relaxed."
        ),
    }
    print(f"\n[9] Live provenance gate      : eligible="
          f"{live_gate['forecast_control_eligible']} · source="
          f"{live_gate['final_safe_control_action_source']} · still blocks="
          f"{live_gate['gate_still_blocks']}")

    # ==================================================================
    # [10] PROTECTED OUTPUTS + FROZEN ARTIFACTS — after
    # ==================================================================
    protected_after = hash_protected()
    protected_unchanged = protected_before == protected_after

    frozen_after = {}
    for rel in manifest["pre_validation"]:
        path = _PROJECT_ROOT / rel
        frozen_after[rel.replace("\\", "/")] = {
            "raw": sha256_file(path),
            "lf": sha256_file_lf(path),
        }
    frozen_unchanged = all(
        frozen_after[rel]["raw"] == frozen_checks[rel]["actual_sha256_raw"]
        and frozen_after[rel]["lf"] == frozen_checks[rel]["actual_sha256_lf_normalised"]
        for rel in frozen_after
    )
    print(f"\n[10] Protected outputs after  : unchanged={protected_unchanged} · "
          f"frozen artifacts unchanged={frozen_unchanged}")

    # ==================================================================
    # VERDICT
    # ==================================================================
    checks = {
        "canonical_reproduction_exit_zero": run.returncode == 0,
        "frozen_artifacts_match_manifest": all(
            v["match_raw"] or v["match_lf_normalised"] for v in frozen_checks.values()
        ),
        "frozen_artifacts_unchanged_by_this_run": frozen_unchanged,
        "protected_phase15_3_outputs_unchanged": protected_unchanged,
        "reproduced_outputs_byte_identical": identical_all,
        "baseline_metrics_exact": all(
            hard_metrics[m]["baseline_ok"] for m in hard_metrics
        ),
        "mpc_metrics_exact": all(hard_metrics[m]["mpc_ok"] for m in hard_metrics),
        "mass_balance_residual_exact": residual_exact,
        "topology_delays_2_1_1": topology["delays_ok"],
        "topology_attenuation_090_085_080": topology["attenuation_ok"],
        "topology_chain_and_nodes": topology["chain_ok"] and topology["node_ids_ok"],
        "candidate_space_is_1296": candidate_space["is_1296"],
        "candidates_logged_1296_every_day": d_optimisation["candidates_are_1296_every_day"],
        "all_four_reservoirs_optimised": d_optimisation["all_four_reservoirs_vary"],
        "reservoir_d_is_optimised_not_pinned": d_optimisation["reservoir_d_is_optimised_not_pinned"],
        "scientific_sources_unmodified": all_clean,
        "live_provenance_gate_not_relaxed": live_gate["gate_still_blocks"],
    }

    evidence.update({
        "title": "Phase 15.3 regression hard gate",
        "frozen_artifact_checks": frozen_checks,
        "manifest_timestamp": manifest.get("timestamp"),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "reproduction": reproduction,
        "output_identity": fidelity,
        "hard_metrics": hard_metrics,
        "canonical_residual": CANONICAL_RESIDUAL,
        "mass_balance_residual_exact": residual_exact,
        "topology": topology,
        "candidate_space": candidate_space,
        "d_optimisation": d_optimisation,
        "scientific_sources": sources,
        "live_provenance_gate": live_gate,
        "checks": checks,
        "final_verdict": "PASS" if all(checks.values()) else "FAIL",
    })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print("\n" + "-" * 78)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"Runtime of the canonical reproduction: {runtime_s:.1f} s")
    verdict = all(checks.values())
    print("FINAL VERDICT:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
