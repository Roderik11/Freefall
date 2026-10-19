# Freefall

Freefall is a game engine and world editor written in C# on Direct3D 12, built with the help of Opus, Fable, Grok and Gemini. The editor can be fully remote-controlled by AI agents.

It is being built for an open-world sandbox RPG, so the focus so far is on the renderer, large outdoor scenes and the tools to author them. The first gameplay is being written as project scripts on top of the engine; networking is designed but not built — see [Status](#status).

## Highlights

- **Fully remote-controllable by AI agents** — an MCP server (about 75 tools) lets an agent such as Claude Code operate the editor end to end: launch it, open a project, import assets, build and save scenes, enter play mode, and check its own work through screenshots, the console and the profiler
- **GPU-driven deferred renderer** — bindless (SM 6.6), compute culling with Hi-Z occlusion, GPU LOD selection, `ExecuteIndirect`; point lights and their shadows are culled, ranked and picked on the GPU
- **Outdoor world** — CDLOD terrain authored entirely with non-destructive stamps, procedural grass, flowers and ferns swaying in the scene's wind, an FFT ocean whose own waves break on the shore, lakes and rivers, time of day and weather
- **One physical atmosphere** — sky, sun and moon light, ambient, haze and fog all come from one model (Hillaire's LUTs) in physical units with auto exposure; god rays through a near fog volume, and clouds, terrain and buildings shading the light in the air and on the ground
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
 Components with per-frame work          ExecuteIndirect (camera, up to 6 sun
 (terrain, ocean, sky)                     cascades, sky captures, point-light
   enqueue via CommandBuffer               shadow atlas)
```

### Frame

| Stage | What happens |
|-------|--------------|
| **Atmosphere** | Transmittance and multiple-scattering LUTs when the air changes; per view the sky-view LUT, the aerial perspective volume, the ambient and the sky probe (a cube with SH ambient, made every frame), plus the clouds' cover toward the sun and moon |
| **G-Buffer** | GPU cull → opaque geometry, terrain (depth prepass first, splat shading after the meshes), ground cover and sky into the G-buffer (reverse-Z) |
| **Shadows** | Sun: up to six cascades in a single pass, casters culled against Hi-Z and a mask of where visible surfaces receive shadow, logarithmic splits from an SDSM depth analysis. Point lights: cube faces in their own atlas, every face of every picked light drawn in one pass. Past the cascades the terrain's shadow comes from a shadow height map per light |
| **Sky and air** | Sky captures (16 depth maps of the meshes seen from the sky, one direction a frame), the lamps' daylight test, and the near fog volume: froxels over the first 128 m lit with the cascades, the clouds and the terrain's shadow |
| **Screen-space** | Hi-Z pyramid, contact shadows (the sun and the top four shadowed point lights), GTAO (half resolution by default), terrain screen-space displacement |
| **Lighting** | Sky visibility per pixel (terrain horizons, cave portals, the sky captures), then the directional light — the sun, or the brightest moon at night (compute) — and tiled point lights from each view's GPU-culled light list (compute) |
| **Composition** | The lit scene, with the sky probe's ambient scaled by sky visibility and AO, is composed into an HDR target and fogged: the near fog volume, then the aerial perspective past it |
| **Forward** | Ocean (with the breaking lips as a second draw), water bodies, transparents, particles, trails and vfx meshes, all through the same fog |
| **Post** | Auto exposure (the frame's histogram and a light meter at the camera, read back three frames late) → bloom pyramid that spreads light rather than adding it → ACES tonemap → optional SMAA → backbuffer |

### Key systems

| System | Description |
|--------|-------------|
| **InstanceBatch** | The GPU-driven batcher. Holds persistent instance records and generic per-instance SoA buffers (descriptors, bounding spheres, bones, terrain patches), and runs the culling pipeline and `ExecuteIndirect`. One batch per effect. |
| **GPUCuller / SceneCuller** | Compute culling shared by the camera, the sun cascades and the point-light shadow atlas: visibility, LOD, histogram, prefix sum, scatter, command generation. Each stage runs for all batches before one barrier. |
| **MeshRegistry** | Global GPU buffer of mesh-part metadata and LOD chains. Shaders look up vertex and index data by ID; there is no input assembler. |
| **TransformBuffer** | Pooled persistent transform slots with dirty-flag uploads. |
| **PointLightBuffer / LocalShadows** | Persistent GPU registry of point lights, culled per view on the GPU. Shadowed lights are ranked by brightness and on-screen size, with hysteresis and fades, and given tiles in the point-light shadow atlas. |
| **Effect / Material** | The `.fx` format: techniques and passes are inferred from the shader, `@RenderState` annotations configure blend, depth and raster state, and push constants are discovered by reflection. Effect settings declared with `//=` defaults become material properties (ranges show as sliders); a `struct Instance` holds per-object values set on the renderer. A project's own `.fx` files are assets. Compiled with DXC. |
| **Atmosphere** | Hillaire's physical atmosphere (Rayleigh, Mie, ozone, plus a ground fog layer per weather preset): the LUTs, the sky probe and the ambient. The one source of sky colour, sun and moon light through the air, ambient and haze. A CPU twin of its transmittance lights the directional light. |
| **VolumetricFog / AerialPerspectiveVolume** | The air between the camera and a surface. The near volume covers the first 128 m in 16-pixel froxels, jittered and blended over frames; the far one covers the screen in ~20-pixel cells to 32 km, marched once per column. Both are shaded by the terrain's shadow map and the clouds. One fog function joins them for every pass. |
| **SkyVisibility / SkyCapture** | How much sky a point sees: terrain horizons and cave portals from the heightmap, and 16 depth maps of the meshes from the sky for walls, roofs and tree crowns. It scales ambient, reflections and haze, so caves, rooms and forest floors are dark. |
| **AutoExposure** | Meters the light on a sphere and the brightness round the camera, so the exposure follows where you stand rather than what fills the screen; the screen's histogram only nudges it. |
| **CommandBuffer** | Thread-safe draw collector for components that submit per frame. Thread-local buckets, block-copy merge. |
| **DeferredRenderer / RenderView** | Orchestrates the frame. Several views can render at once (viewport, previews, thumbnails). |

## Features

### Rendering
- Deferred shading with PBR (GGX), foliage wrap lighting and translucency
- Cascaded shadow maps: up to six cascades in one pass, stabilized, Vogel-disc PCF, logarithmic splits from an SDSM depth analysis, casters culled by where shadows are received; past the cascades the terrain's own shadow (a mountain's shadow over the plain at sunset) from a shadow height map per light
- Point lights on the GPU with no light cap; point-light shadows in an atlas, picked on the GPU each frame; screen-space shadows for the top four so grass and flowers shadow under lanterns
- Lamps that light up when it gets dark: they follow how much daylight reaches the ground, so a thunderstorm lights them too, and lamp glass goes dark with its light
- Screen-space contact shadows and GTAO
- Physical light units: the sun and the moons in lux, the sky in cd/m², point lights in lumens, flames and emissives in nits, all pre-exposed in the HDR targets; light sources seen directly (flames, lit windows, the moon's disc) adapt only partly with the eye
- Auto exposure metered at the camera: the light round where you stand decides it, so a shaded valley stays shaded when you face the sunlit slope; dim light keeps part of its darkness, so dusk falls and nights stay dark
- HDR pipeline with bloom that keeps the image's total brightness, and ACES tonemapping; SMAA
- Sky visibility and ambient: ambient from a sky probe made every frame, scaled by how much sky each point sees (terrain horizons, cave portals and sky captures of the meshes), so caves, rooms and forest floors are dark without hand-placed volumes
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
- **Ocean** — FFT ocean with a multi-band spectrum that follows the weather's wind: a calm sea in clear weather, storm seas that stand taller with whitecaps from the band that holds the spectral peak. Near the coast the sea's own waves are carried to the beach through a warp baked from the seabed: they refract toward the shore, shorten and grow, lee shores get only the swell that wraps round, and sets out at sea arrive as sets. They break by depth with curling lips, spray and mist, run up the sand as bores and drain back. The sea shows only where it can flow in from open water (a sea mask), and is lit and reflects the sky through the sky probe.
- **Lakes and rivers** — at any elevation via `WaterBody`, sharing the ocean's shading and receiving sun shadows
- **Sky and weather** — one physical atmosphere (Hillaire 2020: Rayleigh, Mie, ozone, plus a ground fog layer per preset) for the sky, the sun's and moons' light through the air, the ambient and the haze. Day/night with real twilight. Moons with phases and relief (`Moon` assets) light the night. Stars and the Milky Way turn with the sky. A cloud layer is lit by the sun and moon. Weather presets (rain, snow, hail, dust) cross-fade.
- **Fog, haze and god rays** — a near fog volume (froxels over the first 128 m) lit with the sun's cascades, the clouds and the terrain's shadow: shafts through forest canopies, dark air in shaded squares. Past it the aerial perspective, sized to the screen, carries the terrain's and the clouds' shadow too, so the haze in front of a mountain the sun stands behind stays dark, and an overcast sky does not glow where the sun is. Every pass (opaque, sky, glass, water, particles, trails) goes through the same fog.
- **Clouds and light** — cloud shadows on the ground and in the air; under a deck the air gets diffused light instead of a bright beam toward the sun, and the sky's air below the deck is shaded with it
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

The standalone runtime the engine started with (`Freefall.Game`) was removed on 2026-10-17: play-in-editor replaced it. A standalone build will come from the editor exporting an executable with packed resources.

To use the MCP tools from Claude Code or another MCP client:

```
dotnet build Tools/FreefallMcp/FreefallMcp.csproj
```

## Status

Active development. The renderer, terrain, world authoring and editor are the mature parts. Known gaps:

- **World scale** — one terrain tile (about 4 km); no multi-tile terrain, world streaming or large-world precision yet
- **Rendering** — no spot lights, no reflections beyond the sky, no TAA; point lights do not light the fog yet (no torch halos), and light through openings and bounce indoors are a stand-in until GI
- **Editor** — no undo/redo; exporting a standalone executable with packed resources is in progress
- **Runtime** — no fixed-update loop; physics lacks triggers and joints, and collider groups filter queries only, not collisions between bodies; animation has aim and head look but no foot or hand IK
- **Gameplay** — the first systems are project scripts in the game project, outside this repository: bows and melee, stats, status effects and damage types, NPC daily routines, held items and torches, bindstones and respawn, a HUD. Inventory and saves are not started.
- **Networking** — designed (server-decided hits and projectiles, client movement checked by the server), not built yet
