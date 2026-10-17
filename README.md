# Freefall

Freefall is a game engine and world editor written in C# on Direct3D 12, built with the help of Opus, Fable, Grok and Gemini. The editor can be fully remote-controlled by AI agents.

It is being built for an open-world sandbox RPG, so the focus so far is on the renderer, large outdoor scenes and the tools to author them. The first gameplay is being written as project scripts on top of the engine; networking is designed but not built — see [Status](#status).

## Highlights

- **Fully remote-controllable by AI agents** — an MCP server (about 75 tools) lets an agent such as Claude Code operate the editor end to end: launch it, open a project, import assets, build and save scenes, enter play mode, and check its own work through screenshots, the console and the profiler
- **GPU-driven deferred renderer** — bindless (SM 6.6), compute culling with Hi-Z occlusion, GPU LOD selection, `ExecuteIndirect`; point lights and their shadows are culled, ranked and picked on the GPU
- **Outdoor world** — CDLOD terrain authored entirely with non-destructive stamps, procedural grass, flowers and ferns swaying in the scene's wind, FFT ocean, lakes and rivers, procedural sky with moons, stars, time of day and weather
- **Editor** — docking UI, inspectors, gizmos, prefabs, PCG node graph, animation state-machine editor, CPU/GPU profiler, C# scripts and shaders that hot-reload

## AI automation (MCP)

The editor can be driven entirely by an AI agent, without anyone touching the UI. It hosts an MCP endpoint at `http://localhost:21721/mcp` whose tools cover the whole workflow: launching and shutting down the editor, opening projects, loading and saving scenes, importing and refreshing assets, creating and editing entities, components, prefabs and materials, pressing inspector buttons, terrain queries, PCG graphs, navmesh baking, animation clips (preview, events), play mode, camera, selection and settings. The agent reads the console, takes screenshots of the viewport or the full editor, reads CPU and GPU timings from the profiler, and fingerprints a scene to check that procedural output is reproducible.

`Tools/FreefallMcp` is a stdio bridge that MCP clients launch. It proxies to the editor, keeps the connection alive while the editor is closed, and can start the editor itself. Building it installs it to `%LOCALAPPDATA%\Freefall\McpBridge`, which is where the checked-in `.mcp.json` points.

## Rendering architecture

Renderers register their instances with the GPU once and only touch them again when something changes. Each frame a compute pipeline culls every instance, picks its LOD, groups the survivors by mesh part and writes the indirect draw commands. Each batch keeps its instances' culling data in video memory and refreshes only those whose transform or record changed. The CPU does no per-frame sorting or per-object draw submission.

Point lights follow the same rule: a persistent GPU registry that the light's setters write to. The GPU culls it into a visible list per view and ranks the lights for the shadow atlas, so there is no per-light work on the main thread and no light cap.

```
 CPU                                     GPU
 ───                                     ───
 MeshRenderer / SkinnedMeshRenderer      Culling compute (cull_instances.hlsl)
   register once ──▶ InstanceBatch  ──▶    cached instance data, frustum + Hi-Z,
   (persistent instance records,           LOD select, histogram per mesh part
    transform slots, material IDs)         ─▶ prefix sum ─▶ scatter ─▶ commands
                                                      │
 PointLight ──▶ PointLightBuffer    ──▶  Light cull + shadow selection
   (setters write the registry)                       │
                                                      ▼
 Components with per-frame work          ExecuteIndirect (camera, 4 sun cascades,
 (terrain, ocean, sky)                     point-light shadow atlas)
   enqueue via CommandBuffer
```

### Frame

| Stage | What happens |
|-------|--------------|
| **G-Buffer** | GPU cull → opaque geometry, terrain (depth prepass first, splat shading after the meshes), ground cover and sky into the G-buffer (reverse-Z) |
| **Shadows** | Sun: four cascades in a single pass, casters culled against Hi-Z and a mask of where visible surfaces receive shadow, SDSM depth analysis. Point lights: cube faces in their own atlas, every face of every picked light drawn in one pass |
| **Screen-space** | Hi-Z pyramid, contact shadows (the sun and the top four shadowed point lights), GTAO (half resolution by default), terrain screen-space displacement |
| **Lighting** | Directional light — the sun, or the brightest moon at night (compute) — and tiled point lights from each view's GPU-culled light list (compute) |
| **Composition** | The lit scene, with hemisphere ambient (sky from above, ground bounce from below), is composed into an HDR target |
| **Forward** | Ocean, water bodies, transparents, particles, trails and vfx meshes |
| **Post** | Bloom pyramid that spreads light rather than adding it → ACES tonemap → optional SMAA → backbuffer |

### Key systems

| System | Description |
|--------|-------------|
| **InstanceBatch** | The GPU-driven batcher. Holds persistent instance records and generic per-instance SoA buffers (descriptors, bounding spheres, bones, terrain patches), and runs the culling pipeline and `ExecuteIndirect`. One batch per effect. |
| **GPUCuller / SceneCuller** | Compute culling shared by the camera, the sun cascades and the point-light shadow atlas: visibility, LOD, histogram, prefix sum, scatter, command generation. Each stage runs for all batches before one barrier. |
| **MeshRegistry** | Global GPU buffer of mesh-part metadata and LOD chains. Shaders look up vertex and index data by ID; there is no input assembler. |
| **TransformBuffer** | Pooled persistent transform slots with dirty-flag uploads. |
| **PointLightBuffer / LocalShadows** | Persistent GPU registry of point lights, culled per view on the GPU. Shadowed lights are ranked by brightness and on-screen size, with hysteresis and fades, and given tiles in the point-light shadow atlas. |
| **Effect / Material** | The `.fx` format: techniques and passes are inferred from the shader, `@RenderState` annotations configure blend, depth and raster state, and push constants are discovered by reflection. Effect settings declared with `//=` defaults become material properties (ranges show as sliders); a `struct Instance` holds per-object values set on the renderer. A project's own `.fx` files are assets. Compiled with DXC. |
| **CommandBuffer** | Thread-safe draw collector for components that submit per frame. Thread-local buckets, block-copy merge. |
| **DeferredRenderer / RenderView** | Orchestrates the frame. Several views can render at once (viewport, previews, thumbnails). |

## Features

### Rendering
- Deferred shading with PBR (GGX), foliage wrap lighting and translucency
- Cascaded shadow maps: four cascades in one pass, stabilized, Vogel-disc PCF, adaptive (SDSM) splits, casters culled by where shadows are received
- Point lights on the GPU with no light cap; point-light shadows in an atlas, picked on the GPU each frame; screen-space shadows for the top four so grass and flowers shadow under lanterns
- Lamps that light up when it gets dark: they follow how much daylight reaches the ground, so a thunderstorm lights them too, and lamp glass goes dark with its light
- Screen-space contact shadows and GTAO
- HDR pipeline with bloom that keeps the image's total brightness, and ACES tonemapping; SMAA
- GPU skinning
- GPU particles: every emitter in one shared GPU pool, so a view costs one cull, one simulate and one indirect draw per render mode however many emitters there are; shapes, collision, flipbooks, soft particles, emitters that move (a carried torch's flame stays on it while its smoke trails behind)
- Trails: ribbons behind moving objects and the sweep of a blade (`TrailRenderer`)
- VFX: a universal effect for glowing meshes (beams, lasers, spells) with scrolling noise, dissolve and rim glow, and built-in tube and cone meshes
- Mesh, amplification, hull/domain and compute shader support
- Shader hot reload: engine and project `.fx` / `.hlsl` files recompile on save
- Async texture upload
- Experimental: radiance-cascades GI, screen-space displacement mapping

### Terrain and world
- **Terrain** — GPU quadtree CDLOD with seam stitching, Hi-Z culling, a depth prepass and up to 32 splat layers
- **Splatting** — hex-tiling on every layer against visible repetition, two-scale sampling, side projection on steep ground, cavity shading from the heightmap, a large-scale variation in colour across ground and plants; the ground under vegetation takes on its colour, so a meadow does not end on ground of another colour
- **Stamp-only authoring** — height, layer weights and decoration coverage are baked on the GPU from stamp components in the scene (`HeightStamp`, `SplatStamp`, `DecoStamp`, `CoverageStamp`). Stamps reference `TerrainLayer`, `TerrainDecorator` and `Tag` assets. There are no brushes.
- **Ground cover** — grass cards, procedural grass blades, flowers, spiky plants and ferns through amplification and mesh shaders. Each variant switches between blades, crossed cards and billboards in distance bands.
- **Wind** — one scene wind, taken from the weather preset: irregular gusts that push and spring back move grass, flowers and leaves, and bend whole trees from the base
- **Splines** — Catmull-Rom splines with variable width drive roads, walls and pavements (`RuntimeMesh`), terrain stamps and PCG
- **PCG** — node graph for scattering meshes and prefabs: surface and spline samplers, terrain and mesh projection, slope/height/density filters, obstacle and stamp exclusion, self-pruning, set operations. Output depends on the scene alone: a load, a reload and a re-run give the same result.
- **Water** — FFT ocean (multi-band spectrum, foam, shoreline waves); lakes and rivers at any elevation via `WaterBody`, sharing the ocean's shading and receiving sun shadows
- **Sky and weather** — Rayleigh/Mie atmosphere, day/night cycle, moons with phases (`Moon` assets) that light the night, stars and the Milky Way turning with the sky, clouds with cloud shadows, aerial perspective, weather presets (rain, snow, hail, dust) that cross-fade
- **Navigation** — navmesh baked per tile on background threads, with incremental rebakes that rebuild only the tiles whose inputs changed; pathfinding and crowd agents via DotRecast

### Runtime
- Entity/component model with class components, a parallel update path (`IParallel`) and ordered systems (`EntitySystem` in system groups, placed with `UpdateInGroup`, `UpdateBefore`, `UpdateAfter`); project scripts can define their own systems
- YAML scenes and prefabs
- C# scripting compiled with Roslyn, hot-reloaded with live component migration
- PhysX 5: rigid bodies, box/sphere/capsule/mesh colliders, collider groups for queries, terrain heightfield, capsule character controller
- Animation: state machine with conditions and velocity-matched inertial transitions (or cross-fades), 2D blend trees, animation sets that swap a graph's clips (one graph, a set per weapon type), root motion that moves the character, mirrored playback, footfalls found from the clip itself, clip events saved with the model's import settings, clips that belong to a place (sitting down on a seat), upper-body aim and head look, bone masks, humanoid retargeting
- 3D positional audio (XAudio2; WAV, and OGG decoded at import)
- Keyboard and raw mouse input
- Game UI: a second Squid desktop laid over the viewport in play mode, styled by `UISkin` and `Typeface` assets

### Editor
- Landing page with recent projects; a project is a folder with a `.ffproject` file
- Docking layout: scene hierarchy, inspector, asset browser with thumbnails, console, stats, profiler, viewport
- GPU picking, translate/rotate/scale gizmos with snapping, drag-to-place with ghost preview
- Prefabs, material and mesh inspectors, spline editing in the viewport
- Animation clip inspector (scrub a clip on a character and mark events) and animation set inspector
- `[Button]` on a component method puts a button in its inspector
- Graph editor for PCG graphs and an animation state-machine editor
- CPU and GPU profiler: a call tree with a 300-frame history, allocations and GPU timestamps per render pass
- Play-in-editor, with the game UI over the viewport
- UI built with [Squid](https://github.com/Roderik11/Squid)

### Asset pipeline
- GUID-based asset database with `.meta` files, sub-assets, rename detection and hot reload
- Models: FBX, DAE, OBJ and X via Assimp, with LODs, skeletons, animation clips and cooked collision meshes
- Textures: BCn compression via texconv, PSD; linear data such as normal maps is never colour-converted
- Effects: a project's own `.fx` files, with includes
- Fonts: `Typeface` assets
- Unity asset-pack and scene importer
- Watabou town, city and dwelling importer (walls, gates, houses, multi-floor interiors)

## Project structure

```
Freefall/
├── Freefall.Engine/
│   ├── Engine.cs            # Frame lifecycle, project and scene handling
│   ├── Base/                # Entity, Component, systems, update/draw interfaces, physics and navmesh worlds
│   ├── Components/          # Renderers, lights, terrain and stamps, spline, PCG, water, particles, trails, audio
│   ├── Graphics/            # Device, deferred renderer, batching and culling, lights and shadows, effects, post-processing
│   ├── Animation/           # Skeletons, clips, state machine, blend trees, animation sets, root motion, retargeting
│   ├── Assets/              # Asset database, importers, loaders, packers, terrain baker
│   ├── Graph/               # Node graph core, PCG nodes and scheduler
│   ├── Navigation/          # Tiled navmesh builder
│   ├── UI/                  # Game UI desktop and UI skins
│   ├── Serialization/       # YAML scene and asset serialization
│   └── Resources/Shaders/   # .fx effects and .hlsl compute shaders
├── Freefall.Editor/
│   ├── UI/                  # Docking, inspectors, property controls, graph and animation editors, profiler
│   ├── Mcp/                 # MCP tools
│   ├── Commands/            # Editor commands behind the automation API
│   ├── Tools/               # Unity and Watabou importers, thumbnail rendering
│   └── Scripts/             # Editor camera, gizmos, selection
├── Tools/FreefallMcp/       # stdio MCP bridge
└── Freefall.slnx
```

## Tech stack

- .NET 10 / C#
- Direct3D 12 via [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows), HLSL SM 6.6 compiled with DXC
- PhysX 5 via [PhysX.Net](https://github.com/stilldesign/PhysX.Net)
- [DotRecast](https://github.com/ikpil/DotRecast) for navigation
- Assimp for model import, texconv for texture compression
- Roslyn for scripts
- XAudio2 for audio, [NVorbis](https://github.com/NVorbis/NVorbis) for OGG
- [Squid](https://github.com/Roderik11/Squid) for the editor and game UI
- [MCP C# SDK](https://github.com/modelcontextprotocol/csharp-sdk) for the automation server

## Building

Requirements: Windows, the .NET 10 SDK, and a D3D12 GPU with mesh shader support.

The editor references Squid by relative path, two directories above the repository root, so clone the two like this:

```
<root>/Squid/                 https://github.com/Roderik11/Squid
<root>/<any folder>/Freefall/ this repository
```

Then build and run the editor:

```
dotnet build Freefall.Editor/Freefall.Editor.csproj
dotnet run --project Freefall.Editor
```

`Freefall.slnx` still lists `Freefall.Game`, the standalone runtime the engine started with. Play-in-editor has replaced it and it is not part of this repository, so build the editor project rather than the solution. It will be removed once the editor can export a standalone executable with packed resources.

To use the MCP tools from Claude Code or another MCP client:

```
dotnet build Tools/FreefallMcp/FreefallMcp.csproj
```

## Status

Active development. The renderer, terrain, world authoring and editor are the mature parts. Known gaps:

- **World scale** — one terrain tile (about 4 km); no multi-tile terrain, world streaming or large-world precision yet
- **Rendering** — no spot lights, no reflections beyond the sky, no TAA
- **Editor** — no undo/redo; exporting a standalone executable with packed resources is in progress
- **Runtime** — no fixed-update loop; physics lacks triggers and joints, and collider groups filter queries only, not collisions between bodies; animation has aim and head look but no foot or hand IK
- **Gameplay** — the first systems are project scripts in the game project, outside this repository: bows and melee, stats, status effects and damage types, NPC daily routines, held items and torches, bindstones and respawn, a HUD. Inventory and saves are not started.
- **Networking** — designed (server-decided hits and projectiles, client movement checked by the server), not built yet
