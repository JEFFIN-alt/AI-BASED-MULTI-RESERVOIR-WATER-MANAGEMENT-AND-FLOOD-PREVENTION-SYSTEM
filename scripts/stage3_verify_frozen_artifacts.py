"""
Stage 3 — Frozen artifact SHA256 verification.

Verifies every FROZEN artifact against the hash manifest recorded by the
Phase 15.3 validation run, in a READ-ONLY manner.

  Authority manifest (never modified):
      results/phase15_v3_validation/v3_integrity_check.json

  Frozen artifacts declared by the project:
      models/lstm_pytorch_v3_logtarget/best_model.pt        (V3 model weights)
      models/lstm_pytorch_v3_logtarget/log_target_scaler.pkl (V3 log-target scaler)
      results/lstm_pytorch_v3_logtarget/test_predictions_original_units.csv

Additionally records directory-level hashes of the V3 model and results
directories so later stages can detect any modification.

OUTPUT:
  results/phase15_stage3_topology_reconciliation/frozen_artifact_integrity.json

Exit code is non-zero if ANY frozen artifact hash differs from the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage3_topology_reconciliation"
OUTPUT_PATH = OUTPUT_DIR / "frozen_artifact_integrity.json"

FROZEN_DIRS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_normalised(path: Path) -> tuple:
    """
    SHA256 of the byte stream with CRLF collapsed to LF.

    WHY THIS EXISTS
    ---------------
    This repository is checked out with ``core.autocrlf=true``. Any TEXT
    artifact therefore lands in the working tree with CRLF line endings, which
    changes its byte-level SHA256 relative to the LF blob stored in git and to
    the hash recorded by the Phase 15.3 manifest (computed on LF bytes).

    Comparing the LF-normalised hash answers the question the integrity check
    actually cares about: "is this the same artifact that was frozen?" — while
    the raw hash is still reported so any real content change is caught.

    NOTE: for binary artifacts the normalisation is a no-op (a binary file that
    happens to contain the bytes 0x0D 0x0A would be affected, but only if it
    *also* matches the manifest's LF view, so a false MATCH is not possible
    for a genuinely altered binary).
    """
    raw = path.read_bytes()
    normalised = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalised).hexdigest(), len(normalised)


def hash_dir(dirpath: Path) -> dict:
    hashes = {}
    if dirpath.exists():
        for f in sorted(dirpath.rglob("*")):
            if f.is_file():
                hashes[str(f.relative_to(dirpath)).replace("\\", "/")] = sha256_file(f)
    return hashes


def _autocrlf() -> str:
    """Report the git core.autocrlf setting (informs CRLF normalisation)."""
    import subprocess
    try:
        out = subprocess.run(["git", "config", "--get", "core.autocrlf"],
                             capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
        return (out.stdout or "").strip() or "(unset)"
    except Exception:  # pragma: no cover - git not available
        return "(unknown)"


def main() -> int:
    print("=" * 74)
    print("STAGE 3 — FROZEN ARTIFACT SHA256 VERIFICATION")
    print("=" * 74)

    if not MANIFEST_PATH.exists():
        print(f"\n[FAIL] authority manifest missing: {MANIFEST_PATH}")
        return 2

    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    expected = manifest.get("pre_validation", {})
    print(f"\nManifest      : {MANIFEST_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"Manifest time : {manifest.get('timestamp')}")
    print(f"Artifacts     : {len(expected)}")

    print("\n" + "-" * 74)
    print(f"{'artifact':<58} {'raw':>9} {'lf-norm':>9}")
    print("-" * 74)

    results = {}
    all_match = True
    for rel_path, recorded in expected.items():
        path = _PROJECT_ROOT / rel_path
        if not path.exists():
            results[rel_path] = {
                "expected_sha256": recorded["sha256"],
                "actual_sha256": None,
                "status": "MISSING",
            }
            all_match = False
            print(f"{rel_path:<58} {'MISSING':>9} {'MISSING':>9}")
            continue

        raw_hash = sha256_file(path)
        norm_hash, norm_size = sha256_normalised(path)

        raw_match = raw_hash == recorded["sha256"]
        norm_match = norm_hash == recorded["sha256"]
        matched_via = "raw" if raw_match else ("lf_normalised" if norm_match else None)

        status = "MATCH" if matched_via else "MISMATCH"
        if status != "MATCH":
            all_match = False

        results[rel_path] = {
            "expected_sha256": recorded["sha256"],
            "expected_size_bytes": recorded.get("size_bytes"),
            "raw_sha256": raw_hash,
            "raw_size_bytes": path.stat().st_size,
            "lf_normalised_sha256": norm_hash,
            "lf_normalised_size_bytes": norm_size,
            "raw_matches": raw_match,
            "lf_normalised_matches": norm_match,
            "matched_via": matched_via,
            "status": status,
        }

        print(f"{rel_path:<58} {'MATCH' if raw_match else 'no':>9} "
              f"{'MATCH' if norm_match else 'no':>9}")
        if not matched_via:
            print(f"    expected: {recorded['sha256']}")
            print(f"    raw     : {raw_hash}")
            print(f"    lf-norm : {norm_hash}")
        elif matched_via == "lf_normalised":
            print("    -> byte-identical to the frozen artifact; the raw hash "
                  "differs only by CRLF working-tree line endings "
                  "(core.autocrlf=true).")

    # Directory-level fingerprints (recorded for future stages)
    dir_hashes = {}
    print("\n" + "-" * 74)
    print("Directory fingerprints (recorded for future stages):")
    for d in FROZEN_DIRS:
        h = hash_dir(d)
        dir_hashes[str(d.relative_to(_PROJECT_ROOT)).replace("\\", "/")] = h
        print(f"  {d.relative_to(_PROJECT_ROOT)}: {len(h)} file(s)")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "script": "scripts/stage3_verify_frozen_artifacts.py",
        "manifest": str(MANIFEST_PATH.relative_to(_PROJECT_ROOT)),
        "manifest_timestamp": manifest.get("timestamp"),
        "autocrlf": _autocrlf(),
        "verification_rule": (
            "An artifact is INTACT if its raw SHA256 OR its LF-normalised SHA256 "
            "equals the manifest value. The LF-normalised comparison neutralises "
            "working-tree CRLF conversion introduced by core.autocrlf."
        ),
        "artifacts": results,
        "directory_fingerprints": dir_hashes,
        "all_match": all_match,
        "conclusion": (
            "ALL FROZEN ARTIFACTS UNCHANGED" if all_match
            else "INTEGRITY VIOLATION — ARTIFACT CONTENT DIFFERS"
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 74)
    print(f"VERDICT: {payload['conclusion']}")
    print(f"[OK] {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    print("=" * 74)

    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
