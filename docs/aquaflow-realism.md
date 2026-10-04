# AquaFlow presentation geometry and verification

The renderer remains a Three.js/WebGL digital twin. The supplied aerial photo
guided the forested relief, irregular water bodies, concrete dam faces, daylight,
and atmospheric treatment. It is not used as a texture, backdrop, or replacement
for the interactive scene.

## Geometry and data boundary

The scene explicitly says **STYLIZED TOPOLOGY / NOT GEOGRAPHIC TERRAIN**.
Coordinates, elevations, basin extents, dam sections, and visual animation speeds
are artistic scene units, not surveyed dimensions or calculated hydraulic values.
Relative basin prominence follows the reservoir ordering in
`configs/simulation/four_reservoir_demo.json`; it is not an area conversion from
storage capacity. Basin geometry is fixed. Received water level moves the water
surface; zero water level hides it without deleting the basin.

The authoritative cascade remains A → B → C → D → downstream. Only WebSocket
state supplies operational values. No control, forecast, physical model,
notification, or backend files were changed.

| Visual | Backend field |
|---|---|
| Controlled gate jet and A/B/C connecting current | `controlled_release` |
| Separate crest overflow and spill continuations | `spill_mcm` |
| Terminal river | `downstream.flow_m3_s` |
| Water elevation | `water_level` |
| Physical gate position | `gate` |
| Label storage, inflow, risk, safe limit | Corresponding received fields |

Flow strength uses a monotonic bounded display curve, with distinct normalization
for flow rates and spill volume. It does not calculate or publish physical flow.
Zero and missing values hide the corresponding effects. Water is seated against
the actual terrain triangle surface, and receiving ends remain above the bed.
Foam moves monotonically downstream between wraps; sparse unequal markers avoid
one-second stroboscopic reversal. Their motion is not a velocity measurement.

## Rendering

- Fine terrain plus a separate outer mountain mesh; world-space ground detail,
  slope/elevation colors, carved basins and river gorges.
- Shared instanced broadleaf/tall canopy, trunks, shrubs, and rocks. Two prebuilt
  canopy detail levels switch with camera distance; no geometry is built per frame.
- Shaped gravity/arch sections, crest roads, railings, gate housings, independent
  overflow structures, stilling aprons/baffles, and access roads.
- Four small static cubemap captures of the 3D surroundings. Reflections are
  approximate, not real-time planar mirrors; no reflection cameras run per frame.
- Fresnel, depth tint, irregular surface normals, shoreline blending, separate
  outlet/spill sheets, reusable foam instances, and bounded spray buffers.
- Aerial camera presets, orbit/pan/zoom, selected shoreline and label highlight,
  existing data drawers, and focus mode. Inspection daylight is independent of
  the light/dark UI theme.

## Site View configuration

Site View does not invent maps or Street View. Without an actual provider URL it
shows `SITE IMAGERY NOT CONFIGURED` for MAP, SATELLITE, and STREET VIEW.
A deployment can define `window.AQUAFLOW_SITE_IMAGERY` before the module loads:
each `reservoir_1` through `reservoir_4` entry may contain `map`, `satellite`, and
`street` HTTPS embed URLs from a legitimate provider. Provider credentials,
imagery availability, embedding permissions, and attribution remain the provider's
requirements. No provider is configured by this change.

## Reproduce the evidence

With the app running at `http://127.0.0.1:8000/`:

```powershell
py scripts/verify_realism.py --stage final --hardware --exercise
py scripts/compare_realism_frames.py
```

The browser script captures the six presets and Focus Simulation, t0/t1/t2/t4
sequences for all four links, actual renderer diagnostics, interaction checks,
Site View, and real zero-release/spill scenarios. It uses public API commands,
never injected state, and leaves the demo paused with an explicitly applied
nonzero gate scenario. RESET preserves gate requests, so the script explicitly
applies its test requests before STEP. Do not run this scenario against an
operator's live session without authorization.

`--hardware` requests Chromium's D3D11 backend on Windows. The browser's reported
adapter, not the launch flag, is the evidence of GPU use. Omitting the flag also
supports a software-renderer comparison. `--original` loads the unchanged HEAD
HTML through a browser response route for a same-machine baseline; state is still
received from the real backend.

Results are under `results/realism/`. JSON retains exact state, capture start
times, measured FPS samples, draw calls, triangles, GPU/WebGL, and errors. The
contact sheet contains crops of real captured frames. Pixel differences establish
visible changes along the channel; inspect the sequence to assess direction and
continuity. Procedural terrain and approximate vegetation do not constitute
photogrammetry or a surveyed reconstruction of any reservoir.

## Recorded verification — 2026-10-02

At 1600 × 1000 in headless Chromium using the browser-reported Intel UHD
Graphics / ANGLE D3D11 adapter and WebGL 2.0:

| Measurement | Result |
|---|---|
| Original scene, same adapter, focus median | 60 FPS |
| Final overview sample | 28.53 FPS |
| Final focus median, five one-second samples | 25.16 FPS |
| Final focus draw calls | 72 |
| Final focus triangles | 1,603,104 |
| Browser console/page errors | 0 |
| Existing targeted tests | 14 passed, 35 existing dependency/test warnings |
| `git diff --check` and module syntax check | Passed |

All six presets and Focus Simulation were captured and visually inspected.
Orbit, pan, zoom, reservoir drawer selection, and Site View transitions passed.
Frame sequences show moving highlights along all four channel paths at t0/t1/t2/t4;
the minimum changed channel-pixel count was 344 (difference > 8/255).
The recorded positive controlled releases were 23.148, 40.509, 868.056, and
231.481 m³/s, supplied by the actual backend. Zero controlled release hid the
jets, and a subsequent real spill scenario displayed separate overflow for
A/B/C while their controlled jets stayed hidden. R4 reported no spill in that
scenario. The demo was left paused with nonzero applied gate requests.

Remaining visual limitations: this is a substantially denser procedural scene,
not drone-photo-level realism. Canopies and ground still look procedural in
close inspection; connecting reaches retain some ribbon-like appearance and
flow cues are clearer in focused views than Overview. Reflections are static
environment approximations. The increased detail has a measured performance
cost versus the original. These are not represented as passing photorealistic
acceptance criteria.
