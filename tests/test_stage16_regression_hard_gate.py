"""
Stage 16 — PHASE 15.3 REGRESSION HARD GATE: tests.

Proves that the engineering/integration work of Stages 0-15 did NOT change the
validated Phase 15.3 research result.

The canonical result must remain:

    BASELINE  overflow_events 7 · overflow_volume 10.75 MCM · ds_violations 8
              peak_ds_flow 60.0 · total_release 2017.25
    MPC       overflow_events 0 · overflow_volume 0.0   · ds_violations 0
              peak_ds_flow 30.0 · total_release 2031.0
    mass-balance residual  -6.821210263296962e-13 (baseline AND mpc)

These tests are a REGRESSION GATE: they verify, they do not develop.
"""

import ast
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.controller.mpc_controller import MPCConfig, MPCController  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402
from src.network_env.v3_forecast_adapter import V3ForecastAdapter  # noqa: E402

PROJECT_ROOT = str(_PROJECT_ROOT)
PROTECTED_DIR = _PROJECT_ROOT / "results" / "phase15_v3_validation"
REPRO_DIR = (_PROJECT_ROOT / "results" / "phase15_stage3_reproduction"
             / "phase15_3_reproduction")
REPRO_CHECK = _PROJECT_ROOT / "results" / "phase15_stage3_reproduction" / "REPRODUCTION_CHECK.json"
REPRO_SCRIPT = _PROJECT_ROOT / "scripts" / "stage3_phase15_3_reproduction.py"
VALIDATION_SCRIPT = _PROJECT_ROOT / "scripts" / "run_phase15_3_validation.py"
MANIFEST_PATH = PROTECTED_DIR / "v3_integrity_check.json"
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"
V3_CSV = _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv"

PROTECTED_FILES = (
    "validation_metrics.csv",
    "daily_simulation_baseline.csv",
    "daily_simulation_mpc.csv",
    "provenance_audit.json",
    "v3_integrity_check.json",
    "PHASE_15_3_V3_VALIDATION_REPORT.md",
)
FIDELITY_FILES = ("validation_metrics.csv", "daily_simulation_baseline.csv",
                  "daily_simulation_mpc.csv")

EXPECTED_METRICS = {
    "overflow_events": ("7", "0"),
    "overflow_volume_mcm": ("10.75003168999999", "0.0"),
    "ds_violations": ("8", "0"),
    "peak_ds_flow": ("60.0", "30.0"),
    "total_release_mcm": ("2017.25", "2031.0"),
    "total_steps": ("74", "74"),
    "mass_balance_residual": ("-6.821210263296962e-13", "-6.821210263296962e-13"),
}
CANONICAL_RESIDUAL = -6.821210263296962e-13
NODE_IDS = ["Reservoir_A", "Reservoir_B", "Reservoir_C", "Reservoir_D"]
PHYSICAL_DELAYS = [2, 1, 1]
PHYSICAL_ATTENUATION = [0.90, 0.85, 0.80]

#: Files carrying the scientific computation of Phase 15.3.
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

#: Modules that exist ONLY because of the live Digital Twin work (Stages 10-15).
#: None of them may be reachable from the historical benchmark.
LIVE_PATH_MODULES = (
    "src/dashboard/api/state_manager.py",
    "src/dashboard/api/routes.py",
    "src/dashboard/api/app.py",
    "src/dashboard/sim_bridge.py",
    "src/modeling/gnn_inference.py",
    "src/modeling/gnn_advisory.py",
    "src/controller/live_mpc_orchestrator.py",
    "src/network_env/live_cascade_adapter.py",
    "src/network_env/live_forecast_adapter.py",
    "src/network_env/mass_balance.py",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hash_protected() -> dict:
    return {name: sha256_file(PROTECTED_DIR / name) for name in PROTECTED_FILES}


def read_metrics(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["metric"]: row for row in csv.DictReader(fh)}


def read_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def git_clean(rel: str) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                            cwd=str(_PROJECT_ROOT), capture_output=True)
    return result.returncode == 0


# ===========================================================================
# 3. FROZEN ARTIFACT HASHES
# ===========================================================================

def test_frozen_v3_artifacts_match_the_recorded_manifest():
    """
    Every frozen artifact must still hash to the value recorded at validation
    time. The prediction CSV is the documented Windows case: its RAW hash
    differs because the working tree uses CRLF, so the manifest records the
    LF-normalised hash. The file is byte-identical under that normalisation and
    NO line ending is "fixed" here.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["all_match"] is True
    assert manifest["conclusion"] == "V3 artifacts UNCHANGED"

    for rel, record in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel
        assert path.exists(), rel
        expected = record["sha256"]
        expected_size = record["size_bytes"]

        raw_bytes = path.read_bytes()
        raw = sha256_bytes(raw_bytes)
        lf_bytes = raw_bytes.replace(b"\r\n", b"\n")
        normalised = sha256_bytes(lf_bytes)

        assert len(raw_bytes) == expected_size or len(lf_bytes) == expected_size, (
            f"{rel}: expected {expected_size} bytes, raw {len(raw_bytes)}, "
            f"lf {len(lf_bytes)}"
        )
        assert raw == expected or normalised == expected, (
            f"{rel}: expected {expected}, raw {raw}, lf {normalised}"
        )
        if raw != expected:
            # Only the CRLF case may take this branch, and it must be the CSV.
            assert path == V3_CSV, f"{rel} unexpectedly needed LF normalisation"
            assert normalised == expected
            # The size difference must be EXACTLY the CRLF overhead.
            assert len(raw_bytes) == len(lf_bytes) + raw_bytes.count(b"\r\n")


def test_prediction_csv_crlf_case_is_understood_not_hidden():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["pre_validation"][
        "results\\lstm_pytorch_v3_logtarget\\test_predictions_original_units.csv"
    ]["sha256"]
    raw = sha256_file(V3_CSV)
    normalised = sha256_file_lf(V3_CSV)
    assert normalised == expected, "the LF-normalised CSV no longer matches the manifest"
    assert V3_CSV.read_bytes().count(b"\r\n") > 0, "the tracked file is expected to use CRLF"
    assert raw != expected, "raw and normalised were expected to differ (documented CRLF case)"


# ===========================================================================
# 13. OUTPUT / HASH COMPARISON (protected baseline)
# ===========================================================================

def test_protected_outputs_match_the_recorded_reproduction_hashes():
    recorded = json.loads(REPRO_CHECK.read_text(encoding="utf-8"))
    for name, digest in recorded["protected_hashes_before"].items():
        assert sha256_file(PROTECTED_DIR / name) == digest, f"{name} changed"
    assert recorded["protected_hashes_before"] == recorded["protected_hashes_after"]


# ===========================================================================
# 2/8/9. HARD CANONICAL METRICS
# ===========================================================================

@pytest.mark.parametrize("metric", sorted(EXPECTED_METRICS))
def test_canonical_metrics_are_exact(metric):
    """The canonical Phase 15.3 numbers must be EXACT, not 'approximately similar'."""
    expected_baseline, expected_mpc = EXPECTED_METRICS[metric]
    row = read_metrics(PROTECTED_DIR / "validation_metrics.csv")[metric]
    assert row["baseline"] == expected_baseline, metric
    assert row["mpc"] == expected_mpc, metric


def test_mass_balance_residual_is_the_canonical_float():
    row = read_metrics(PROTECTED_DIR / "validation_metrics.csv")["mass_balance_residual"]
    assert float(row["baseline"]) == CANONICAL_RESIDUAL
    assert float(row["mpc"]) == CANONICAL_RESIDUAL
    assert abs(CANONICAL_RESIDUAL) < 1e-10  # comfortably inside the 1e-9 tolerance


def test_baseline_and_mpc_logs_have_the_canonical_shape():
    baseline = read_rows(PROTECTED_DIR / "daily_simulation_baseline.csv")
    mpc = read_rows(PROTECTED_DIR / "daily_simulation_mpc.csv")
    assert len(baseline) == len(mpc) == 74
    assert all(row["controller"] == "BASELINE" for row in baseline)
    assert all(row["controller"] == "MPC" for row in mpc)
    assert all(row["forecast_used"] == "True" for row in mpc)


# ===========================================================================
# 5. TOPOLOGY IDENTITY
# ===========================================================================

def test_topology_identity_from_the_canonical_config():
    config = yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    network = ReservoirNetwork(config_dict=copy.deepcopy(config))

    assert list(network.processing_order) == NODE_IDS
    assert network._terminal_node_id == "Reservoir_D"

    connections = list(network.connections)
    assert [(c.source, c.destination) for c in connections] == [
        ("Reservoir_A", "Reservoir_B"), ("Reservoir_B", "Reservoir_C"),
        ("Reservoir_C", "Reservoir_D"),
    ]
    assert [c.delay for c in connections] == PHYSICAL_DELAYS
    for conn, expected in zip(connections, PHYSICAL_ATTENUATION):
        assert conn.attenuation == pytest.approx(expected)
    assert float(network.downstream_capacity) == pytest.approx(50.0)


def test_physics_equations_and_timestep_semantics_unchanged():
    """One step advances exactly one timestep and conserves mass."""
    config = yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    network = ReservoirNetwork(config_dict=copy.deepcopy(config))
    before = int(network.timestep)
    network.step({nid: 1.0 for nid in NODE_IDS}, {nid: 0.3 for nid in NODE_IDS})
    assert int(network.timestep) - before == 1
    report = network.mass_balance_check()
    assert abs(report["residual_error"]) < 1e-9


# ===========================================================================
# 6/7. CANDIDATE SPACE AND D OPTIMISATION
# ===========================================================================

def test_candidate_space_is_six_to_the_four():
    config = MPCConfig()
    assert list(config.gate_levels) == [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
    assert len(config.gate_levels) ** 4 == 1296
    assert config.lookahead_steps == 3
    assert config.max_gate_change == 0.5


def test_canonical_logs_were_produced_with_1296_candidates_every_day():
    mpc = read_rows(PROTECTED_DIR / "daily_simulation_mpc.csv")
    assert {row["candidates_evaluated"] for row in mpc} == {"1296"}


def test_reservoir_d_is_optimised_not_pinned_in_the_canonical_run():
    """
    D (Idukki) must be an ordinary decision variable. In the canonical MPC log
    its gate takes more than one value — a pinned D could only ever repeat one.
    """
    mpc = read_rows(PROTECTED_DIR / "daily_simulation_mpc.csv")
    gates = {
        nid: sorted({float(row[f"{nid}_gate"]) for row in mpc})
        for nid in NODE_IDS
    }
    for nid, values in gates.items():               # all four participate
        assert len(values) > 1, f"{nid} never moved: {values}"
    assert len(gates["Reservoir_D"]) > 1, "Reservoir D is pinned"

    # The baseline log is a different controller and must also move D.
    baseline = read_rows(PROTECTED_DIR / "daily_simulation_baseline.csv")
    assert len({float(row["Reservoir_D_gate"]) for row in baseline}) > 1


def test_a_single_phase15_3_mpc_decision_covers_all_four_reservoirs():
    """
    Direct proof on the canonical inputs: one real MPC decision returns gates for
    ALL FOUR reservoirs and evaluates 1296 candidates.
    """
    config = yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    network = ReservoirNetwork(config_dict=copy.deepcopy(config))
    adapter = V3ForecastAdapter(project_root=PROJECT_ROOT)
    date = adapter.available_dates[0]
    snapshot = adapter.get_network_snapshot(date, NODE_IDS)

    inflows = {}
    for nid in NODE_IDS:
        forecast = adapter.get_forecast(nid, date)
        value = forecast.actual_1d
        inflows[nid] = 0.0 if value is None or value != value else float(value)

    decision = MPCController().decide(network, forecast_snapshot=snapshot,
                                      current_inflows=inflows)
    assert set(decision.gate_positions) == set(NODE_IDS)
    assert decision.candidates_evaluated == 1296
    assert decision.forecast_used is True


def test_safety_layer_is_in_the_canonical_research_path():
    """The canonical run recorded SafetyLayer corrections, so the layer is live."""
    mpc = read_rows(PROTECTED_DIR / "daily_simulation_mpc.csv")
    statuses = {row["mpc_status"] for row in mpc}
    assert statuses <= {"OPTIMAL", "CORRECTED"}
    assert "CORRECTED" in statuses, "the SafetyLayer never corrected a proposal"


# ===========================================================================
# 1/4. CANONICAL REPRODUCTION + OUTPUT IDENTITY  (THE HARD GATE)
# ===========================================================================

def test_canonical_reproduction_reproduces_the_frozen_outputs_byte_identically():
    """
    THE HARD GATE. Runs the canonical reproduction exactly as validated, then
    requires:
      * exit code 0;
      * the six protected Phase 15.3 outputs UNCHANGED across the run;
      * the three re-derived outputs byte-identical to the protected ones.

    The reproduction script isolates itself (its own OUTPUT_DIR) and hashes the
    protected directory itself; this test independently re-verifies both.
    """
    before = hash_protected()

    result = subprocess.run([sys.executable, str(REPRO_SCRIPT)],
                            cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, (
        "the canonical Phase 15.3 reproduction failed:\n"
        + (result.stdout or "")[-2000:] + "\n" + (result.stderr or "")[-2000:]
    )

    after = hash_protected()
    assert after == before, "the reproduction MODIFIED a protected Phase 15.3 output"

    for name in FIDELITY_FILES:
        protected = sha256_file(PROTECTED_DIR / name)
        reproduced = sha256_file(REPRO_DIR / name)
        assert protected == reproduced, f"{name} is not byte-identical"


def test_the_last_reproduction_outputs_are_still_byte_identical():
    """Cheap companion check: the checked-in reproduction copy still matches."""
    for name in FIDELITY_FILES:
        assert sha256_file(PROTECTED_DIR / name) == sha256_file(REPRO_DIR / name), name


# ===========================================================================
# 9. NO LIVE-PATH WORK CAN SILENTLY ALTER THE HISTORICAL BENCHMARK
# ===========================================================================

def test_phase15_3_pipeline_imports_only_scientific_modules():
    """The benchmark must not reach any live/Stage-10-15 module."""
    imported = imports_of(VALIDATION_SCRIPT)
    offenders = sorted(
        name for name in imported
        if any(token in name for token in (
            "state_manager", "sim_bridge", "gnn", "live_cascade", "live_forecast",
            "mass_balance", "live_mpc_orchestrator", "dashboard", "modeling.inference",
        ))
    )
    assert offenders == [], f"the Phase 15.3 pipeline imports live-path modules: {offenders}"


def test_no_live_module_is_reachable_from_the_benchmark():
    """No live-path module may be imported by the benchmark or its sources."""
    benchmark_sources = [VALIDATION_SCRIPT, REPRO_SCRIPT,
                         _PROJECT_ROOT / "src" / "network_env" / "v3_forecast_adapter.py",
                         _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py"]
    live_module_names = {Path(p).stem for p in LIVE_PATH_MODULES}
    for path in benchmark_sources:
        imported = " ".join(imports_of(path))
        for live in sorted(live_module_names):
            assert live not in imported, f"{path.name} imports the live module {live}"


def test_scientific_sources_are_unmodified_since_the_validated_baseline():
    """
    Stages 10-15 must not have altered a single line of the scientific
    computation. Verified against git: the working tree matches HEAD for every
    Phase 15.3 source, and each was last committed with the Phase 15.3 work.
    """
    for rel in SCIENTIFIC_SOURCES:
        assert git_clean(rel), f"{rel} has uncommitted modifications"

    for rel in SCIENTIFIC_SOURCES:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s", "--", rel],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True,
        )
        subject = (result.stdout or "").strip()
        assert "Phase 15.3" in subject or "added" in subject, (rel, subject)


def test_live_provenance_gate_was_not_relaxed_to_pass_this_gate():
    """
    The regression validates the HISTORICAL research pipeline. It must not have
    been achieved by relaxing the live provenance rules: the live AI path must
    still refuse to control on non-validated forecasts.
    """
    from fastapi.testclient import TestClient

    from src.dashboard.api import state_manager
    from src.dashboard.api.app import app

    client = TestClient(app)
    try:
        client.post("/api/simulation/pause")
        client.post("/api/simulation/reset")
        client.post("/api/controller/mode", json={"mode": "MANUAL"})
        for _ in range(8):
            client.post("/api/simulation/step")
        client.post("/api/controller/mode", json={"mode": "AI"})
        state_manager.sim_state.step()

        control = client.get("/api/state").json()["control"]
        assert control["forecast_control_eligible"] is False
        assert control["final_safe_control_action_source"] == "HELD_CURRENT_GATES"
        assert control["blocked_reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"

        provenance = client.get("/api/state").json()["forecast_provenance"]
        assert provenance["live_forecasts_are_validated"] is False
    finally:
        client.post("/api/simulation/pause")
        client.post("/api/simulation/reset")
        client.post("/api/controller/mode", json={"mode": "MANUAL"})
