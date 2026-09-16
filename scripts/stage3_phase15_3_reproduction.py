"""
Stage 3 — Safe Phase 15.3 baseline/MPC reproduction.

Re-runs the Phase 15.3 baseline-vs-MPC validation WITHOUT overwriting any
protected validation artifact.

HOW SAFETY IS GUARANTEED
------------------------
``scripts/run_phase15_3_validation.py`` is IMPORTED (not executed) and its
module-global ``OUTPUT_DIR`` is redirected to a fresh Stage 3 directory. The
original protected directory

    results/phase15_v3_validation/

is (a) hashed before the run, (b) hashed after the run, and the two hashes are
compared. Any difference aborts the reproduction with a non-zero exit code.

The script also verifies that the reproduced ``validation_metrics.csv`` matches
the protected one byte-for-byte, proving the Stage 3 physics change did not
perturb the validated Phase 15.3 results (Stage 3 does not touch that path).

OUTPUT:
  results/phase15_stage3_reproduction/
      phase15_3_reproduction/   <- full re-run output (safe copy)
      REPRODUCTION_CHECK.json   <- protection + fidelity evidence
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

VALIDATION_SCRIPT = _PROJECT_ROOT / "scripts" / "run_phase15_3_validation.py"
PROTECTED_DIR = _PROJECT_ROOT / "results" / "phase15_v3_validation"
STAGE3_DIR = _PROJECT_ROOT / "results" / "phase15_stage3_reproduction"
REPRO_DIR = STAGE3_DIR / "phase15_3_reproduction"
CHECK_PATH = STAGE3_DIR / "REPRODUCTION_CHECK.json"

COMPARED_FILES = [
    "validation_metrics.csv",
    "daily_simulation_baseline.csv",
    "daily_simulation_mpc.csv",
]


def hash_dir(dirpath: Path) -> dict:
    hashes = {}
    if dirpath.exists():
        for f in sorted(dirpath.rglob("*")):
            if f.is_file():
                hashes[str(f.relative_to(dirpath)).replace("\\", "/")] = hashlib.sha256(
                    f.read_bytes()
                ).hexdigest()
    return hashes


def load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "stage3_run_phase15_3_validation", VALIDATION_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # no side effects: guarded by __main__
    return module


def main() -> int:
    print("=" * 74)
    print("STAGE 3 — SAFE PHASE 15.3 REPRODUCTION")
    print("=" * 74)

    if not PROTECTED_DIR.exists():
        print(f"[FAIL] protected directory missing: {PROTECTED_DIR}")
        return 2

    protected_before = hash_dir(PROTECTED_DIR)
    print(f"\nProtected dir : {PROTECTED_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"Files         : {len(protected_before)}")
    for name, h in protected_before.items():
        print(f"  {name:<42} {h[:16]}...")

    module = load_validation_module()
    original_output_dir = module.OUTPUT_DIR
    print(f"\nOriginal OUTPUT_DIR (redirected away from): "
          f"{Path(original_output_dir).relative_to(_PROJECT_ROOT)}")

    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    module.OUTPUT_DIR = REPRO_DIR
    print(f"Redirected OUTPUT_DIR -> {REPRO_DIR.relative_to(_PROJECT_ROOT)}")

    print("\nRunning run_validation() ...\n")
    module.run_validation()

    protected_after = hash_dir(PROTECTED_DIR)
    shield_ok = protected_before == protected_after

    print("\n" + "=" * 74)
    print("PROTECTION CHECK")
    print("=" * 74)
    for name in sorted(set(protected_before) | set(protected_after)):
        before = protected_before.get(name, "<absent>")
        after = protected_after.get(name, "<absent>")
        print(f"  {name:<42} {'UNCHANGED' if before == after else '*** CHANGED ***'}")

    fidelity = {}
    for fname in COMPARED_FILES:
        p_prot = PROTECTED_DIR / fname
        p_repro = REPRO_DIR / fname
        if not p_prot.exists() or not p_repro.exists():
            fidelity[fname] = {"identical": False, "reason": "missing file"}
            continue
        h_prot = hashlib.sha256(p_prot.read_bytes()).hexdigest()
        h_repro = hashlib.sha256(p_repro.read_bytes()).hexdigest()
        fidelity[fname] = {
            "protected_sha256": h_prot,
            "reproduced_sha256": h_repro,
            "identical": h_prot == h_repro,
        }

    print("\n" + "=" * 74)
    print("REPRODUCTION FIDELITY")
    print("=" * 74)
    for fname, info in fidelity.items():
        print(f"  {fname:<42} {'IDENTICAL' if info.get('identical') else 'DIFFERS'}")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "script": "scripts/stage3_phase15_3_reproduction.py",
        "protected_dir": str(PROTECTED_DIR.relative_to(_PROJECT_ROOT)),
        "reproduction_dir": str(REPRO_DIR.relative_to(_PROJECT_ROOT)),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_artifacts_untouched": shield_ok,
        "fidelity": fidelity,
        "all_reproduced_identical": all(v.get("identical") for v in fidelity.values()),
    }

    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECK_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[OK] {CHECK_PATH.relative_to(_PROJECT_ROOT)}")

    if not shield_ok:
        print("\n*** FAIL: protected validation artifacts were modified ***")
        return 1

    print("\nVERDICT: protected artifacts untouched; reproduction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
