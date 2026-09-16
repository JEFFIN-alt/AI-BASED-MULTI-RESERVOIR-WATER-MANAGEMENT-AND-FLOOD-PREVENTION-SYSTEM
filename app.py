"""
AQUA FLOW — 3D Multi-Reservoir Digital Twin · One-Command Launcher
==================================================================

Usage (from the project root):

    python app.py            # start on port 8000 and open the browser
    python app.py --port     # start on port 8000 and open the browser
    python app.py --port 8080
    python app.py --no-browser
    python app.py --install  # only install/verify dependencies, then exit

What it does:
  1. Verifies required packages (fastapi, uvicorn, torch, pandas, ...)
     and pip-installs any that are missing (into the active environment).
  2. Starts the FastAPI backend (src/dashboard/api/app.py) which serves:
       - The 3D Digital Twin UI  ->  http://127.0.0.1:8000/
       - The REST API            ->  http://127.0.0.1:8000/api/...
       - The live WebSocket feed ->  ws://127.0.0.1:8000/ws/state
  3. Opens your default browser at the UI.

No Streamlit involved — this is the Three.js 3D twin only.
"""

import argparse
import importlib
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Packages required by the 3D twin backend (import name, pip name)
#
# NOTE: the import-name -> pip-name mapping matters. Two of these are imported
# at module scope by the backend import chain and will abort startup if absent:
#   * yaml            -> src/network_env/reservoir_network.py (top-level import)
#   * torch_geometric -> src/modeling/gnn_inference.py (top-level import), which
#                        is imported by src/dashboard/api/state_manager.py
# Order is preserved so that heavy downloads (torch, torch_geometric) come last.
REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("networkx", "networkx"),
    ("joblib", "joblib"),
    ("yaml", "PyYAML"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
]

BANNER = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║   A Q U A   F L O W                                    ║
    ║   Multi-Reservoir Digital Twin · 3D Simulation          ║
    ╚════════════════════════════════════════════════════════╝
"""


def ensure_packages() -> None:
    """Install any missing required packages via pip."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )
        print("[setup] Done.")
    else:
        print("[setup] All required packages already installed.")


def open_browser_later(url: str, delay_seconds: float = 2.5) -> None:
    """Open the browser after the server has had a moment to boot."""

    def _open():
        time.sleep(delay_seconds)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Aqua Flow 3D Multi-Reservoir Digital Twin."
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the browser"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Only install/verify dependencies, then exit"
    )
    args = parser.parse_args()

    print(BANNER)

    # 1. Dependencies
    ensure_packages()
    if args.install:
        print("[setup] Dependency check complete. Exiting (--install).")
        return

    # 2. Make project-root imports (src.*) work regardless of cwd
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    url = f"http://{args.host}:{args.port}/"

    # 3. Import the FastAPI app (after deps are guaranteed present)
    try:
        from src.dashboard.api.app import app  # noqa: F401 (import validates chain)
    except Exception as exc:  # pragma: no cover
        print(f"[error] Failed to import the backend: {exc}")
        print("        Try running:  python -m pip install -r requirements.txt")
        sys.exit(1)

    # 4. Serve with uvicorn
    print()
    print("  3D Digital Twin UI : " + url)
    print("  REST API          : " + url + "api/state")
    print("  WebSocket         : ws://" + args.host + f":{args.port}/ws/state")
    print()
    print("  Press CTRL+C to stop the server.")
    print()

    if not args.no_browser:
        open_browser_later(url)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[aqua-flow] Server stopped. Bye!")
