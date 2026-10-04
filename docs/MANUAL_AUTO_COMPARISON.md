# Fixed gates versus AUTO MPC — updated rehearsal

For the follow-up question about large inflows, use the separate
[increased-inflow and overload demonstration](HIGH_INFLOW_DEMO.md).

Compare two fresh 12-day runs. Both start at 50% storage, with the same capacities,
routing and local inflows: A=1, B=0.5, C=8, D=0 MCM/day. Do not switch to AUTO
halfway through the manual run and treat that as the comparison.

## Load the controller update first

1. In the terminal running the application, press **Ctrl+C** to stop the old server.
2. Start the updated presentation copy:

```powershell
& "C:\Users\JEFFIN\Documents\Codex\2026-10-04\inspect-this-whole-repository-and-tell\AquaFlow-Presentation\START_PRESENTATION.cmd"
```

3. Wait for `Uvicorn running on http://127.0.0.1:8000`, then refresh the browser
   at http://127.0.0.1:8000/ using **Ctrl+F5**. Keep the terminal open.

Refreshing the page alone cannot update a running Python controller. A depleted
old scenario also will not refill automatically: start a fresh scenario below.

## Run 1 — fixed manual gates

1. Click **Restart scenario** (or **Initialize scenario**). Verify day 0,
   50% storage and closed gates.
2. Select **Historical replay**, leaving **Manual** active.
3. Enter Requested % values **A=0, B=50, C=0, D=25**. Press Enter after each.
4. Click **+1 day · Step** exactly **12 times**, keeping requests unchanged.
5. Show the final storage table, A's spill and the trend chart. Save a screenshot.

Say:

> This baseline leaves fixed gate requests unchanged. A accumulates water and
> spills, while B and D run out of stored water. It illustrates the weakness of
> this fixed policy; it does not represent every possible human decision.

Manual uses gate bounds, movement limits and downstream checks. The new AUTO
storage-reserve policy is inactive in Manual, as the interface explicitly states.

## Run 2 — coordinated AUTO with reserve protection

1. Click **Restart scenario** again. Verify the same day-0 conditions.
2. Select **Historical replay**, then **AUTO · MPC**.
3. Step exactly **12 times**, keeping AUTO active throughout.
4. After the first step, confirm the sidebar says **AUTO reserve 30% · checked
   against actual daily inflow and due arrivals**.
5. At day 12, show storage and applied gates. Open **Models & control** to show
   forecasts, the most recent MPC decision and the storage protection explanation.
6. Save the second screenshot.

Say:

> MPC replans joint gate actions each day, considering forecast inflows and
> routing delays. A hard reserve check prevents today's release from spending
> water that has only been forecast for the future. In this scenario the updated
> controller avoids spill and keeps every reservoir at or above its reserve.

## Expected results from the actual core code

These results use the real network, live MPC orchestrator, gate safety,
downstream guard and frozen historical LSTM outputs. The full application still
needs a fresh live rehearsal after restarting the server.

| Measurement after 12 days | Fixed manual A0/B50/C0/D25 | Updated AUTO |
|---|---:|---:|
| A storage | 100% | 84.66% |
| B storage | 0% | 30.00% |
| C storage | 81.50% | 74.80% |
| D storage | 0% | 47.01% |
| Total spill over the run | 6.59 MCM | 0 MCM |
| Peak actual downstream flow | 50 MCM/day | 30 MCM/day |
| Days exceeding downstream capacity | 0 | 0 |
| Empty reservoir-days | 18 | 0 |

An empty reservoir-day counts one empty reservoir on one day. The UI's Spill
column shows that day's spill; total spill is summed from the daily evidence.
Both runs stay within downstream capacity: do not claim the manual run flooded.

## Explain the reserve honestly

The **30% threshold is an assumed prototype operating reserve**, not a verified
Kerala dam rule. It is configurable in the live MPC configuration. At each step:

`maximum release = current storage + actual local inflow + due routed inflow − protected reserve`

Release is also bounded by physical capacity and gate movement constraints.
Water arriving on a later day cannot fund today's release. If storage already
starts below 30%, the controller prevents further depletion; it cannot create
water to restore 30% instantly. If movement and reserve limits conflict, the
simulation refuses the step and exposes the error.

Historical B forecasts remain larger than its preset 0.5 MCM/day local inflow.
Replay is a **what-if experiment**, not a forecast of these fixed inputs.
The protection handles this mismatch at the daily action boundary; it does not
make the historical forecasts accurate for this scenario.

## Is the improvement only the added reserve?

Reserve protection explains why B no longer empties. An extra core experiment
gave the fixed gate policy the identical 30% reserve check. It kept B and D at
30%, but A still reached 100% and spilled **6.59 MCM**. AUTO avoided that spill
and retained **47.01%** in D. This additional experiment is saved with the
evidence; it is not a separate Manual-mode button in the current UI.

An all-closed baseline also avoids depletion but spills 6.59 MCM at A. Present
spill, depletion and downstream flow together. This is an improvement on tested
baselines, not proof that MPC always outperforms a skilled operator.

GCN-LSTM remains advisory. This evaluates fixed-gate policies against
LSTM-forecast-driven MPC with reserve protection, not GCN-LSTM control.

## Evidence and checks

Daily results and conservation checks are in
`results/presentation_refresh/policy_comparison.json`.
The parent workspace contains `compare_demo_policies.py` and
`run_reserve_checks.py`. Eight reserve regressions passed, covering forecast
mismatch over 12 days, due arrivals, below-reserve starting states, movement
conflicts, downstream replacements and the final application boundary.
The dependency-light runner omits unused YAML loading and does not start the
full FastAPI/PyTorch application. Frozen model artifacts are unchanged.
