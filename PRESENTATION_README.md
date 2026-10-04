# Aqua Flow presentation copy

This folder contains the complete repository and the updated interface. The
original `test-case-3` folder has not been edited.

## Launch the live application

Open a terminal in this folder and use your existing Python environment:

```powershell
python app.py
```

Alternatively, double-click `START_PRESENTATION.cmd`. The live application
opens at http://127.0.0.1:8000/. Keep its terminal open while presenting.
The shortcut uses your installed Python 3.12 directly when available, and saves
terminal output to `presentation-startup.log`. If the browser says the connection
was refused, inspect this terminal/log for a startup error. Wait for the actual
`Uvicorn running on http://127.0.0.1:8000` message before opening the page.
If that port is occupied, use `python app.py --port 8001`.

The existing launcher checks dependencies and installs any missing packages.
Start it before the presentation so dependency setup can finish if needed.

The tab at port **8060** is a **recorded layout preview**. Its controls are
disabled. Use the live application's port 8000 for the actual simulation.

## AUTO reserve fix

Stop the old Python server with Ctrl+C, restart this copy, and refresh the
browser with Ctrl+F5. AUTO now protects a 30% prototype operating reserve using
actual daily inflow and due routed water. Eight reserve regressions passed.
In the isolated 12-day comparison, AUTO ended at A=84.66%, B=30%, C=74.80%,
D=47.01%, with no empty reservoirs or spill. Historical replay remains a what-if
forecast source. Manual mode does not enforce the AUTO reserve.

Use [the updated comparison guide](docs/MANUAL_AUTO_COMPARISON.md) for the two
fresh runs and the examiner explanation. Rehearse these results in the live app.

## Rehearsal sequence

1. Click **Initialize scenario**. It resets storage to 50%, closes gates and
   selects deterministic inflows. The button then reads **Restart scenario**.
2. Click **+1 day · Step**. A storage should become **6.41 MCM**.
3. Enter **50** in A's **Requested %** table cell and press Enter. Step again:
   A release is **2.5 MCM/day**, and A storage becomes **4.91 MCM**.
4. Step twice more. A storage becomes **1.91 MCM**; B receives **2.25 MCM/day**
   after the routing delay and reaches **14.88 MCM**.
5. Select **Historical replay**, then **AUTO · MPC**, then Step. Explain that
   stored frozen LSTM predictions drive MPC and safety checks in a historical
   what-if experiment. Inspect the actual proposal, safety output and applied
   action in **Models & control**.
6. Return to Manual, request D=100%, and Step. In this sequence the final
   boundary limits the applied D gate to **25%**. Show Requested versus Applied.
7. Demonstrate Play and Pause. Use the camera selector to focus on Idukki.

GCN-LSTM is spatial-temporal **advisory** in this repository. It does not feed
MPC or directly control gates. It needs seven distinct simulation days of
history; unavailable or synthetic-input outputs remain labelled. The displayed
training correlation graph has 16 nodes and 41 edges and differs from the
four-reservoir simulation topology. Open **Assumptions** for these distinctions.

## Verification and limits

The refresh passed Node transport regressions, 14 risk-engine unit tests,
Python/JavaScript syntax checks, and isolated checks of the actual daily network,
historical MPC, fractional safety boundary, error recovery and GNN input layout.
Frozen artifact hashes are unchanged. Results are in `results/presentation_refresh`.

The full FastAPI/PyTorch application could not be started inside this chat:
the installed Python executable is blocked by the execution sandbox, and the
allowed runtime lacks those dependencies. Rehearse the live sequence above
using your usual `python app.py` before presenting. A recorded browser preview
does not substitute for this live check.

Detailed causal examples and scientific assumptions are in
`docs/CLASSROOM_DEMONSTRATION.md`.
