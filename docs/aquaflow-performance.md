# Rendering-only performance pass

No backend, MPC, SafetyLayer, ReservoirNetwork, forecast, AUTO-control,
hydraulic, or Gmail code is changed by this pass. The benchmark reads the live
state and never sends simulation commands. State can therefore evolve between
captures; these are real operating frames, not injected/frozen telemetry.

## GPU diagnosis

Windows reports NVIDIA GeForce RTX 5050 Laptop GPU and Intel UHD Graphics.
Headed Chromium was tested with `--enable-gpu --ignore-gpu-blocklist
--use-angle=d3d11`. WebGL reports **Intel UHD / ANGLE D3D11**, not NVIDIA.
The WebGL context reports the requested mode as `high-performance` and WebGL 2.0.
The application cannot force Windows/browser adapter selection. No Windows
graphics preference, driver, or registry setting was changed.

Some headed samples were throttled to approximately 1 FPS when the window was
occluded, even with Chromium background-throttling switches disabled. These are
excluded from the final performance comparison. Headed GPU-selection evidence
is retained. The final same-adapter comparison uses hardware-backed headless
Chromium, a 1600 × 1000 viewport, and five one-second FPS samples per view.

## Isolated investigation

The initial unthrottled headed baseline measured 25.69 FPS Overview / 23.58 FPS
Focus. Diagnostic removals were temporary response-route overrides, never
shipped product changes:

| Isolated diagnostic | Overview FPS | Focus FPS |
|---|---:|---:|
| Vegetation hidden | 37.34 | 33.80 |
| Terrain hidden | 40.52 | 42.49 |
| Ground detail shader bypassed | 32.42 | 31.82 |
| Reservoir water hidden | 28.83 | 25.85 |
| Mist hidden | 28.66 | 25.54 |
| Shadow sampling disabled | 29.82 | 32.05 |

Terrain and forest rendering dominate these measurements. Removing a system is
not a pure GPU-time measurement: visibility, overdraw, and scheduling also change.
The measured improvements support prioritizing those systems over rewriting
water. The last two experimental entries (`half_pixels`, `labels`) in that early
run overlapped a source edit and are not used as isolated evidence. Later runs
freeze both HTML and module source for reproducibility.

There is one primary sun and one cached 2048² shadow map. Shadow passes do not
run every animation frame. Four 64² cubemap reflection targets are captured only
at startup. There is no post-processing stack or volumetric atmosphere.
The starting canvas uses DPR 1, not DPR 2; geometry cost and shader work remain
significant even before reducing pixel count.

## Changes

- Precomputed 528² single-channel noise lattice replaces repeated trigonometric
  ground-noise hashing with filtered texture lookups. It preserves terrain
  geometry, multiscale detail, slope shading, and roughness variation. The new
  texture adds approximately 272 KiB of source texel data without mipmaps.
- All 12,500 trees retain their placement and instance colour. Nearby views use
  spatial batches with per-patch bounds and distance-based geometry LOD. Wide
  views use combined batches to avoid excessive draw-call overhead. Both paths
  reuse existing crown geometry; no mesh/geometry/material is created per frame.
- Labels project at most 15 Hz and skip projection/DOM work entirely while camera,
  water heights, and label state are unchanged. Text values still follow backend
  updates. Channel inlet buffers upload only when receiving water height changes.
  Rain updates skip an empty draw range.
- A compact Visual Quality selector defaults to Balanced and persists locally.
  Resolution adapts at three-second intervals inside each preset's bounds.
  CSS/DOM labels remain at native resolution. Quality never affects simulation.

| Preset | Pixel ratio bounds, capped by display DPR | Near-detail distance | Shadow map |
|---|---|---:|---:|
| Ultra | fixed, up to 1.25 | 1100 scene units | 2048² |
| High | 0.9–1.0 | 850 | 2048² |
| Balanced | 0.8–1.0 | 680 | 2048² |
| Performance | 0.65–0.85 | 500 | 1024² |

The distance values are presentation units, not surveyed dimensions. Water
reflections, Fresnel, directional flow, outlet/spill effects, reservoir geometry,
dam detail, atmosphere, and lighting are retained at every quality level.

## Reproduction and evidence

```powershell
py scripts/profile_twin_rendering.py --stage final-stable --headless --matrix
py scripts/verify_realism.py --stage performance-flow --hardware --flows
py scripts/compare_realism_frames.py --root results/realism/performance-flow
```

`results/performance/` retains intermediate measurements and source snapshots.
`before-stable/report.json` and `final-forest-culling/report.json` are the final
comparable runs. Reports include FPS, frame time, draw calls, triangles, geometry/texture
counts, shadow dimensions/update policy, drawing-buffer/CSS sizes, DPR, actual
vendor/renderer, requested mode, WebGL version, and console errors. Each measured
camera/quality has an actual screenshot. `performance-flow` uses positive live
backend releases; it fails instead of inventing values if releases are zero.

## Final verification (2 October 2026)

The final forest-culling matrix was captured after the last frontend edit in
`results/performance/final-forest-culling/`. It rendered all 28 combinations
(four presets across Overview, Reservoir 1–4, Downstream, and Focus) with zero
console/page errors. At the 1600 × 1000 browser viewport, the Overview capture
reported 73 draw calls, 1,222,728 triangles, a 2048² cached sun shadow, and
WebGL 2.0. The runtime renderer was Intel UHD through ANGLE D3D11; the context
reported `high-performance`, but Windows did not select the installed RTX 5050.

Same-machine headless Chromium baseline and final Balanced medians:

| Capture | Baseline | Final Balanced |
|---|---:|---:|
| Overview | 30.0 FPS | 35.0 FPS |
| Focus | 26.1 FPS | 26.6 FPS |

FPS varies with camera, view content, and live backend updates. Across the seven
views, observed ranges were 28.5–42.3 FPS (Ultra), 31.5–42.3 (High), 19.0–45.0
(Balanced), and 30.0–40.7 (Performance). The non-monotonic samples are retained
in the report; preset labels describe resolution/LOD/shadow tradeoffs, not an
FPS guarantee. The Balanced overview used pixel ratio 0.8; Performance may adapt
between 0.65 and 0.85. The inspected Overview and close-reservoir images retained
the terrain, forest, water, dams, labels, and network. Lower raster resolution
and distant forest LOD are the visible quality tradeoffs, especially in
Performance.

`results/realism/performance-flow-final/report.json` records actual positive
controlled releases for all four reservoirs (34.72, 77.55, 520.83, and
347.22 m³/s) and downstream flow (416.67 m³/s); spill is independently positive
at R3 and R4. The read-only Playwright run issued no simulation or gate commands.
Four frames at t0/t1/t2/t4 were captured for each network leg. Screenshot pixel
checks along each projected channel found more than 10 pixels changing by over
8 intensity levels at every later timestamp on every link. The contact sheet
and measurements are `results/realism/performance-flow-final/flow-contact-sheet.png`
and `pixel-comparison.json`.

Final reproduction commands:

```powershell
py scripts/profile_twin_rendering.py --stage final-forest-culling --headless --matrix
py scripts/verify_realism.py --stage performance-flow-final --hardware --flows
py scripts/compare_realism_frames.py --root results/realism/performance-flow-final
```

The final UI/API/WebSocket/integration regression run passed 14 tests. No
backend, scientific, or control files were modified.
