# Aqua Flow — presentation rehearsal guide

Allow about 6–8 minutes. Follow the steps in order for the numerical examples.

To lead with the benefit of automatic replanning, use
[the fixed-manual-versus-AUTO comparison](MANUAL_AUTO_COMPARISON.md) instead of
the opening walkthrough below. It specifies two fresh 12-day runs and reports
both improvements and limitations; the two sequences should not be mixed.

## Before presenting

After the reserve update, stop any old server with **Ctrl+C** and restart it
with the command below. Then use **Ctrl+F5** in the browser. A page refresh
alone does not load new Python controller code.

Run this in PowerShell. The `&` executes the quoted path; without it PowerShell
only prints the path.

```powershell
& "C:\Users\JEFFIN\Documents\Codex\2026-10-04\inspect-this-whole-repository-and-tell\AquaFlow-Presentation\START_PRESENTATION.cmd"
```

Keep the terminal open. Wait for `Uvicorn running on http://127.0.0.1:8000`,
then open http://127.0.0.1:8000/. The header should show **Backend live**.
Port 8060 is a recorded layout preview with disabled controls.

If startup fails, read `presentation-startup.log` in the project folder. A
browser connection error does not identify the Python startup failure.

Use **Whole cascade**, a readable light/dark theme, and fullscreen if useful.
Stay on the three presentation tabs: **Simulation**, **Models & control**, and
**Assumptions**. The **More** dashboard is optional.

## 1. Introduce the system — 30 seconds

Say:

> Our prototype uses GCN-LSTM for spatial-temporal reservoir analysis, LSTM for
> inflow forecasting, and MPC for coordinated release planning. In this version,
> GCN-LSTM is advisory. The control path uses LSTM forecasts, MPC and safety checks.

Point out A Anayirankal, B Ponmudi, C Idamalayar and D Idukki. Explain that the
names come from the dataset and the displayed cascade is an assumed prototype
topology. Idukki's larger arch-dam graphic is schematic.

## 2. Show storage and manual release — 90 seconds

Click **Initialize scenario** (or **Restart scenario** if already active).
This selects Manual, 50% starting storage, closed gates and fixed local inflows
A=1, B=0.5, C=8, D=0 MCM/day. One **+1 day · Step** advances one physical day.

| Action | Expected observation | What to explain |
|---|---|---|
| Step once: day 1 | A storage changes from 5.41 to **6.41 MCM**; release is zero. | Local inflow increases storage when the gate is closed. |
| Enter **50** in A's **Requested %** cell and press Enter | Request is accepted; the physical gate changes on the next Step. | Requested and Applied are separate values. |
| Step once: day 2 | A applied gate is **50%**, release **2.5 MCM/day**, storage **4.91 MCM**. B routed inflow is still zero. | A receives 1 MCM and releases 2.5 MCM, so its storage falls by 1.5 MCM. |

Storage is in **MCM** (million cubic metres). Flow is in **MCM/day**. The storage
history chart records completed simulation days, rather than an invented curve.

## 3. Show delayed coordination — 60 seconds

Keep A requested at 50% and step twice more.

| Day | A storage | B routed inflow | B storage |
|---|---:|---:|---:|
| 3 | 3.41 MCM | 0 MCM/day | 12.13 MCM |
| 4 | 1.91 MCM | 2.25 MCM/day | 14.88 MCM |

Say:

> The release made at A on day 2 reaches B on day 4. Our A-to-B link has a
> two-day delay and a 0.90 transmission factor, so 2.5 becomes 2.25 MCM/day.
> B also receives its own 0.5 MCM/day inflow. This is why coordinated planning
> must consider downstream reservoirs and water already in transit.

The other assumed links are B→C: one day ×0.85, and C→D: one day ×0.80.

## 4. Demonstrate LSTM → MPC → safety — 90 seconds

At day 4, select **Historical replay**, then **AUTO · MPC**, then Step once.
Open **Models & control**.

Point to the +1/+3/+7-day LSTM forecast table and the actual action stages:
**MPC**, **Gate safety**, and **Final**. In the rehearsed baseline sequence,
the expected final vector is **A=0%, B=50%, C=0%, D=0%**. Show what the backend
actually reports; extra steps or changed inputs can produce another decision.
The table is labelled **Most recent MPC decision**. After returning to Manual,
it retains the earlier AUTO decision. Use **Simulation → Applied %** for current
physical gates; later manual safety corrections can differ from that record.

Say:

> These are stored predictions from the frozen LSTM model's historical test
> data. MPC uses them to evaluate coordinated gate candidates over eight daily
> steps, accounting for storage and routing. Safety checks the proposed action
> before the simulation applies it. AUTO also checks a 30% prototype storage
> reserve using actual daily inflow and water arriving today. This is a historical what-if experiment,
> rather than a forecast of this preset's current inflows.

Do not describe **Simulation inputs** as a validated real-time forecast. With
missing measured features or insufficient provenance, AUTO may report blocked
or unavailable and hold a proposal that still passes final safety checks.

## 5. Demonstrate the safety boundary — 60 seconds

Return to **Simulation**. Select **Manual**, enter **100** in D's Requested %
cell, press Enter, and Step once: day 6 in this sequence.

The rehearsed result is **Requested D=100%**, **Applied D=25%**, and a controlled
release of **50 MCM/day**. Expand the downstream-check explanation if needed.

Say:

> An operator request does not directly become the physical gate position.
> The final boundary checks gate movement and predicted downstream capacity.
> In this state, the request is reduced to 25%. The protection claim applies
> to the scenario that was checked; it is not a guarantee against flooding.

If the backend reports an unprotected scenario, present that result honestly.
Do not equate a changed gate with successful downstream protection.

## 6. Explain GCN-LSTM and the model's limits — 60 seconds

Open **Models & control** and point to the GCN-LSTM training graph.

Say:

> This graph represents statistical relationships among 16 dataset reservoirs,
> with 41 undirected training correlations. GCN-LSTM combines that spatial
> structure with seven-day temporal histories. Its output is advisory in this
> implementation; it does not supply the MPC control forecast or move gates.
> The training correlation graph and the four-node water-routing cascade
> represent different relationships.

The live advisory needs seven distinct simulation days of history. You can
step once more to day 7 and inspect its status, but availability also depends
on the loaded model and usable features. Display the actual status. The static
training graph is not proof that live inference ran.

Open **Assumptions** and briefly state:

- Daily storage follows `S_next = S + local inflow + routed inflow − release − spill`.
- Gates use a linear release mapping; water height is a storage proxy.
- Only controlled release routes between reservoirs. Upstream spill exits
  through separate assumed lateral outlets; terminal D release plus spill
  defines downstream river flow.
- Evaporation and detailed floodplain hydraulics are omitted. The cascade
  connectivity and rendered dam geometry are schematic.

## 7. Finish — 30 seconds

Return to **Simulation**, show **Play**, then **Pause**. Explain that playback
advances the same daily physics; the speed setting changes viewing cadence.

Close with:

> The demonstrated contribution is a traceable reservoir-control workflow:
> forecast inputs, coordinated MPC proposals, safety corrections and applied
> actions can be inspected separately. The next development step is to validate
> richer hydraulics and any proposed integration of the spatial advisory into control.

## Likely questions

| Question | Answer |
|---|---|
| Does GCN-LSTM control the gates? | No. It is advisory; LSTM-based forecasts feed MPC, followed by safety checks. |
| Are these current real-world forecasts? | Historical replay uses stored model predictions on historical data. Simulation-derived forecasts are labelled separately. |
| Is the Kerala cascade geographically verified? | No. The current four-node topology is an assumed prototype arrangement. |
| Why did B empty in the earlier AUTO run? | Storage protection was only a soft penalty and historical forecasts overstated scenario inflow. The updated AUTO policy enforces a daily 30% prototype reserve using actual available water. |
| Why does Requested differ from Applied? | The backend applies the final safety-checked action on the next simulation step. |
| Does the system guarantee flood prevention? | No. Downstream protection is conditional on forecast assumptions and the finite candidate search. |
| Has this refreshed application been fully rehearsed live? | Targeted core and UI checks passed. Complete the live sequence in the installed Python environment before presenting. |

## Recovery during rehearsal

- **Connection refused:** inspect the open terminal/startup log and wait for the
  server-running message. Reloading the browser cannot start the server.
- **Controls disabled with Layout preview:** switch from port 8060 to the live app.
- **Unexpected numerical values:** click Restart scenario and follow the exact
  sequence above without extra steps, storm changes or gate requests.
- **Step error:** read the visible error, preserve its text, and restart the
  scenario. Do not silently treat a failed step as completed.
