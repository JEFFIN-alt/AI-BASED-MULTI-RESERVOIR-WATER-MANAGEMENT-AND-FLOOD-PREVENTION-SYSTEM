# Increased inflow and overload — examiner demonstration

Use this after the 12-day preservation comparison. This sequence uses existing
controls and needs no application update. It tests the controller's response
to larger actual inflow; historical replay forecasts remain unchanged and are
not forecasts of the storm being introduced.

## First: elevated inflow response

1. Pause the simulation. Expand **Playback & extra controls** in the sidebar.
2. Click **Exit preset / reset**. This starts a new 50% storage run and leaves
   the fixed A=1/B=0.5/C=8/D=0 preset. Do not click Initialize scenario afterward.
3. Set **Storm input** to its minimum, **0**. Outside the preset this uses
   standard local inflows **A=3, B=4, C=90, D=0 MCM/day**. Inflows update on Step.
4. Select **Historical replay**, then **AUTO · MPC**.
5. Step three times, showing Local in, Applied %, Release, Spill and storage.

C's local inflow is now 90 MCM/day, versus 8 in the preservation preset.
In the core test, C's gate changes from **15% → 30% → 50%**, producing controlled
releases **22.5 → 45 → 75 MCM/day**. After three days:

| Measure | Fixed manual A0/B50/C0/D25 | AUTO |
|---|---:|---:|
| Total spill | 107.945 MCM | 0 MCM |
| A storage | 100% | 80.04% |
| B storage | 35.89% | 30% |
| C storage | 100% | 89.60% |
| D storage | 12.63% | 41.03% |
| Peak actual terminal flow | 50 MCM/day | 30 MCM/day |

For the manual comparison, independently repeat steps 1–3, select Manual,
enter A=0/B=50/C=0/D=25 and step three times. Reset between runs.

Say:

> With increased actual inflow, the controller raises releases as storage
> rises, accounts for delayed transfers and preserves the minimum reserve.
> It avoids the spill seen with this fixed gate policy over the tested interval.

Do not present three days as an unlimited guarantee. In the same AUTO test,
continuing these inflows to day 6 produces **24.208 MCM** total spill at A and C.
The historical forecast mismatch and the finite controller/search assumptions
remain limitations. The three-day sequence needs a fresh live rehearsal.

## Second: maximum storm overload

1. Pause; use **Exit preset / reset** again for a fresh 50% run.
2. Keep Historical replay and AUTO selected. Set **Storm input** to **100**.
3. Step once and inspect all four Local in values and the Spill column.

Maximum storm sets actual local inflows to **A=12, B=16, C=360, D=0 MCM/day**.
On the first day the core test gives C=100% storage, a 50% gate, a 75 MCM/day
controlled release and **110.855 MCM spill**. C starts with 174.145 MCM of free
storage; the first-day gate movement limit permits at most 75 MCM release.
Its incoming 360 MCM therefore cannot fit into that available storage and release.
Even its unrestricted maximum release is only 150 MCM/day, so sustained
360 MCM/day inflow cannot be accommodated indefinitely.

Say:

> MPC coordinates releases within physical and operating constraints. It cannot
> guarantee zero spill when inflow exceeds storage and outlet capacity. This
> overload test exposes those limits rather than hiding overflow.

The tested first-day terminal flow is **50 MCM/day**, with D's gate at 25%.
This is a terminal-flow observation, not proof of overall flood prevention:
**upstream spill exits through separate assumed lateral outlets in this model**.
The impact of those outlets on real downstream communities is not simulated.

Keep this distinction clear even if the panel reports that terminal downstream
flow is protected. Do not describe the synthetic storm as an LSTM-predicted
event, or claim that no spill means no flooding in the real world.

## Evidence

The dependency-light core tests use the production network, MPC, reserve,
downstream guard and frozen historical forecast adapter. Gates obey bounds
and movement limits; reserve and conservation checks run on every step.
Daily evidence is in `results/presentation_refresh/high_inflow_probe.json`.
The reproducible runner is `probe_high_inflow.py` in the parent workspace.
These checks do not start the full FastAPI/PyTorch application.
