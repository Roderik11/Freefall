# Freefall

Freefall is a game engine and world editor written in C# on Direct3D 12, built with the help of Opus, Fable, Grok and Gemini.

It is being built for an open-world sandbox RPG, so the focus so far is on the renderer, large outdoor scenes and the tools to author them. Gameplay systems and networking are still to come — see [Status](#status).

## Highlights

- **GPU-driven deferred renderer** — bindless (SM 6.6), compute culling with Hi-Z occlusion, GPU LOD selection, `ExecuteIndirect`
- **Outdoor world** — CDLOD terrain authored entirely with non-destructive stamps, mesh-shader ground cover, FFT ocean, lakes and rivers, procedural sky with time of day and weather
- **Editor** — docking UI, inspectors, gizmos, prefabs, PCG node graph, animation state-machine editor, C# scripts and shaders that hot-reload
- **AI-drivable** — the editor hosts an MCP server (about 60 tools), so an agent such as Claude Code can build and inspect scenes in the running editor

## Rendering architecture

Renderers register their instances with the GPU once and only touch them again when something changes. Each frame a compute pipeline culls every instance, picks its LOD, groups the survivors by mesh part and writes the indirect draw commands. The CPU does no per-frame sorting or per-object draw submission.

```
 CPU                                     GPU
 ───                                     ───
 MeshRenderer / SkinnedMeshRenderer      Culling compute (cull_instances.hlsl)
   register once ──▶ InstanceBatch  ──▶    frustum + Hi-Z visibility, LOD select
   (persistent instance records,           histogram per mesh part ─▶ prefix sum
    transform slots, material IDs)         scatter ─▶ indirect command generation
                                                      │
 Components with per-frame work                       ▼
 (terrain, ocean, sky, particles)        ExecuteIndirect (camera + 4 shadow cascades)
   enqueue via CommandBuffer
```

### Frame

| Stage | What happens |
|-------|--------------|
| **G-Buffer** | GPU cull → opaque geometry, terrain, ground cover and sky into the G-buffer (reverse-Z) |
| **Shadows** | Four cascades rendered in a single pass, with their own Hi-Z caster culling and SDSM depth analysis |
| **Screen-space** | Hi-Z pyramid, contact shadows, GTAO, terrain screen-space displacement |
| **Lighting** | Directional light (compute) and tiled point lights (compute, Hi-Z pre-culled) |
| **Composition** | The lit scene is composed into an HDR target |
| **Forward** | Ocean, water bodies, transparents and particles |
| **Post** | Bloom pyramid → ACES tonemap → optional SMAA → backbuffer |

### Key systems

| System | Description |
|--------|-------------|
| **InstanceBatch** | The GPU-driven batcher. Holds persistent instance records and generic per-instance SoA buffers (descriptors, bounding spheres, bones, terrain patches), and runs the culling pipeline and `ExecuteIndirect`. One batch per effect. |
| **GPUCuller / SceneCuller** | Compute culling shared by the camera and the shadow cascades: visibility, LOD, histogram, prefix sum, scatter, command generation. |
| **MeshRegistry** | Global GPU buffer of mesh-part metadata and LOD chains. Shaders look up vertex and index data by ID; there is no input assembler. |
| **TransformBuffer** | Pooled persistent transform slots with dirty-flag uploads. |
| **Effect / Material** | The `.fx` format: techniques and passes are inferred from the shader, `@RenderState` annotations configure blend, depth and raster state, and push constants are discovered by reflection. Compiled with DXC. |
| **CommandBuffer** | Thread-safe draw collector for components that submit per frame. Thread-local buckets, block-copy merge. |
| **DeferredRenderer / RenderView** | Orchestrates the frame. Several views can render at once (viewport, previews, thumbnails). |

## Features

### Rendering
- Deferred shading with PBR (GGX), foliage wrap lighting and translucency
- Cascaded shadow maps: four cascades in one pass, stabilized, Vogel-disc PCF, adaptive (SDSM) splits
- Screen-space contact shadows and GTAO
- Tiled point lights
- HDR pipeline with bloom and ACES tonemapping; SMAA
- GPU skinning
- GPU particles (compute-simulated; shapes, collision, flipbooks, soft particles)
- Mesh, amplification, hull/domain and compute shader support
- Shader hot reload: engine `.fx` / `.hlsl` files recompile on save
- Async texture upload
- Experimental: radiance-cascades GI, screen-space displacement mapping

### Terrain and world
- **Terrain** — GPU quadtree CDLOD with seam stitching, Hi-Z culling and up to 32 splat layers
- **Stamp-only authoring** — height, layer weights and decoration coverage are baked on the GPU from stamp components in the scene (`HeightStamp`, `SplatStamp`, `DecoStamp`, `CoverageStamp`). Stamps reference `TerrainLayer` and `TerrainDecorator` assets. There are no brushes.
- **Ground cover** — grass, flowers and rocks through amplification and mesh shaders, with wind
- **Splines** — Catmull-Rom splines with variable width drive roads, walls and pavements (`RuntimeMesh`), terrain stamps and PCG
- **PCG** — node graph for scattering meshes and prefabs: surface and spline samplers, terrain and mesh projection, slope/height/density filters, obstacle and stamp exclusion, self-pruning, set operations
- **Water** — FFT ocean (multi-band spectrum, foam, shoreline waves); lakes and rivers at any elevation via `WaterBody`, sharing the ocean's shading and receiving sun shadows
- **Sky and weather** — Rayleigh/Mie atmosphere, day/night cycle, clouds with cloud shadows, aerial perspective, weather presets (rain, snow, hail, dust) that cross-fade
- **Navigation** — navmesh baking, pathfinding and crowd agents via DotRecast

### Runtime
- Entity/component model with class components and a parallel update path (`IParallel`)
- YAML scenes and prefabs
- C# scripting compiled with Roslyn, hot-reloaded with live component migration
- PhysX 5: rigid bodies, box/sphere/capsule/mesh colliders, terrain heightfield, capsule character controller
- Animation: state machine with conditions and cross-fades, 2D blend trees, events, bone masks, humanoid retargeting
- 3D positional audio (XAudio2; WAV and OGG)
- Keyboard and raw mouse input

### Editor
- Landing page with recent projects; a project is a folder with a `.ffproject` file
- Docking layout: scene hierarchy, inspector, asset browser with thumbnails, console, stats, viewport
- GPU picking, translate/rotate/scale gizmos with snapping, drag-to-place with ghost preview
- Prefabs, material and mesh inspectors, spline editing in the viewport
- Graph editor for PCG graphs and an animation state-machine editor
- Play-in-editor
- UI built with [Squid](https://github.com/Roderik11/Squid)

### Asset pipeline
- GUID-based asset database with `.meta` files, sub-assets, rename detection and hot reload
- Models: FBX, DAE, OBJ and X via Assimp, with LODs, skeletons, animation clips and cooked collision meshes
- Textures: BCn compression via texconv, PSD
- Unity asset-pack and scene importer
- Watabou town, city and dwelling importer (walls, gates, houses, multi-floor interiors)

### Automation (MCP)
The editor hosts an MCP endpoint at `http://localhost:21721/mcp`. Its tools cover scenes, entities and components, assets, prefabs, terrain queries, PCG graphs, camera, settings, console and screenshots.

`Tools/FreefallMcp` is a stdio bridge that MCP clients launch. It proxies to the editor, keeps the connection alive while the editor is closed, and can start the editor itself. Building it installs it to `%LOCALAPPDATA%\Freefall\McpBridge`, which is where the checked-in `.mcp.json` points.

## Project structure

```
Freefall/
├── Freefall.Engine/
│   ├── Engine.cs            # Frame lifecycle, project and scene handling
│   ├── Base/                # Entity, Component, update/draw interfaces, physics and navmesh worlds
│   ├── Components/          # Renderers, lights, terrain and stamps, spline, PCG, water, particles, audio
│   ├── Graphics/            # Device, deferred renderer, batching and culling, effects, post-processing
│   ├── Animation/           # Skeletons, clips, state machine, blend trees, retargeting
│   ├── Assets/              # Asset database, importers, loaders, packers, terrain baker
│   ├── Graph/               # Node graph core, PCG nodes and scheduler
│   ├── Navigation/          # Navmesh builder
│   ├── Serialization/       # YAML scene and asset serialization
│   └── Resources/Shaders/   # .fx effects and .hlsl compute shaders
├── Freefall.Editor/
│   ├── UI/                  # Docking, inspectors, property controls, graph and animation editors
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
- XAudio2 for audio
- [Squid](https://github.com/Roderik11/Squid) for the editor UI
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

`Freefall.slnx` also lists a `Freefall.Game` project (the standalone game runtime). It is not part of this repository yet, so build the editor project rather than the solution.

To use the MCP tools from Claude Code or another MCP client:

```
dotnet build Tools/FreefallMcp/FreefallMcp.csproj
```

## Status

Active development. The renderer, terrain, world authoring and editor are the mature parts. Known gaps:

- **World scale** — one terrain tile (about 4 km); no multi-tile terrain, world streaming or large-world precision yet
- **Rendering** — no spot lights or local-light shadows, no reflections beyond the sky, no TAA
- **Editor** — no undo/redo, no build/packaging step
- **Runtime** — no fixed-update loop; physics lacks triggers, collision layers and joints; animation lacks root motion and IK
- **Gameplay** — combat, stats, inventory, AI, UI/HUD and saves are not started
- **Networking** — planned last
