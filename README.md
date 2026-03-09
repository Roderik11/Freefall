# Freefall

Freefall is a game engine written in C# with Direct3D 12, built entirely with [Google Antigravity](https://antigravity.google/blog).

## Architecture

Freefall is a fully GPU-driven deferred renderer. The CPU submits unsorted draw calls in parallel; a multi-pass compute pipeline handles visibility culling, histogram-based grouping, and indirect command generation — so the GPU draws only what is visible, with zero CPU sorting.

```
 ┌─────────────────────────── CPU ───────────────────────────┐     ┌──────────────── GPU ─────────────────┐
 │                                                           │     │                                      │
 │  Components (IParallel)                                   │     │  Compute (cull_instances.hlsl)        │
 │    StaticMesh ─┐                                          │     │    CSVisibility (frustum cull)        │
 │    SkinnedMesh ├──▶ ThreadLocal DrawBuckets               │     │    CSHistogram (count per mesh-part)  │
 │    Terrain     ┘     (zero contention)                    │     │    CSPrefixSum (offsets)              │
 │                         │                                 │     │    CSGlobalScatter (compact SoA)      │
 │                   Block-Copy Merge ─▶ GPU Upload (O(1))   │     │    CSMain (build draw commands)       │
 │                                                           │     │         │                             │
 └───────────────────────────────────────────────────────────┘     │    ExecuteIndirect                    │
                                                                   │    (draws only visible instances)     │
                                                                   └──────────────────────────────────────┘
```

### Render Loop

| Step | Pass | What happens |
|------|------|--------------|
| 1 | **Script Draw** | Components enqueue draws into the `CommandBuffer` |
| 2 | **Opaque** | Merge buckets → upload → GPU cull → `ExecuteIndirect` → G-Buffer |
| 3 | **Shadow** | Re-cull opaque batches per cascade → 4× `ExecuteIndirect` → shadow maps |
| 4 | **Sky** | Skybox rendered as inside-out cube via the standard pipeline |
| 5 | **Light** | Fullscreen quad reconstructs world pos, samples cascades, accumulates lighting |
| 6 | **Compose** | `Albedo × LightBuffer` → Composite |
| 7 | **Forward** | CompositeSnapshot copy → ocean and transparent forward objects render to Composite with depth testing |
| 8 | **Blit** | Composite → backbuffer copy |

### Key Systems

| System | Description |
|--------|-------------|
| **InstanceBatch** | Core GPU-driven batcher. All per-instance data (descriptors, bounding spheres, subbatch IDs, terrain patches, bones, lights) flows through a unified `PerInstanceBuffer` system with auto-resize. Manages histogram culling pipeline and `ExecuteIndirect`. One batch per Effect. |
| **CommandBuffer** | Thread-safe draw call collector. `Enqueue` dispatches to all applicable passes based on the Effect's declared passes. Thread-local `DrawBucket`s enable lock-free parallel submission. |
| **MeshRegistry** | Global GPU buffer of mesh metadata. Persistent `MeshPartID`s eliminate per-frame CPU grouping — the GPU looks up vertex/index info directly. |
| **TransformBuffer** | Pooled persistent GPU transform slots with dirty-flag uploads. Entities hold a stable `TransformSlot` for their lifetime. |
| **MaterialBlock** | Per-instance parameter overrides. Data is staged into contiguous byte arrays at enqueue time and uploaded as generic per-instance SoA buffers via push constants. |
| **Material / Effect** | Data-driven PSO management. `@RenderState` annotations in shaders auto-configure blend, depth, raster state. `MasterEffects` pattern for global parameter broadcast. Effect push constants discovered via shader reflection on named `cbuffer PushConstants`. |
| **GPUCuller** | 6-pass compute pipeline: Clear → Visibility → Histogram → PrefixSum → Scatter → CommandGen. Shared between camera and shadow passes. |
| **DeferredRenderer** | Orchestrates the G-Buffer, shadow atlas, light accumulation, and composition passes. |

## Features

### Rendering
- **GPU-Driven Rendering** — Indirect draw calls with compute-based visibility culling and instance scatter
- **Reverse-Z Depth Buffer** — Near→1, far→0 projection for improved floating-point precision
- **Deferred Shading** — GBuffer-based pipeline with directional and point light support
- **Cascaded Shadow Maps** — 4-cascade PSSM with texel snapping, bounding-sphere stabilization, Vogel disc filtering, and adaptive SDSM splits
- **Hi-Z Occlusion Culling** — Hierarchical depth buffer for GPU-side occlusion tests with visibility feedback to prevent false-positive culling loops
- **Unified Per-Instance Buffers** — All per-instance data flows through a single generic SoA channel system with auto-resize and push-constant binding
- **Persistent Transform Buffer** — Pooled GPU transform slots with dirty-flag uploads
- **Skeletal Animation** — GPU skinning via per-instance bone buffers
- **Terrain** — Fully GPU-driven restricted quadtree with compute-based node evaluation, screen-space error LOD, edge stitching, and indirect rendering
- **Ocean** — FFT-based ocean with 4 spectrum bands, GPU tessellation, world-space shore displacement attenuation via terrain heightmap, PS shore effects (terrain show-through with refraction, animated foam, shallow water color), and distance-based normal band fadeout
- **LOD System** — Automatic LOD level selection for static meshes based on screen-space size
- **Point Lights** — Deferred point lights via per-instance `StructuredBuffer`, rendered as sphere volumes with additive blending
- **Bindless SM 6.6** — All resources accessed via `ResourceDescriptorHeap` and reflection-driven named push constants; no Input Assembler
- **Debug Visualization** — F5 cycles through: cascade colors, shadow factor, and linear depth

### Physics
- **PhysX Integration** — NVIDIA PhysX via [MagicPhysX](https://github.com/Cysharp/MagicPhysX) for rigid body simulation
- **Character Controller** — PhysX capsule controller with ground detection, slope handling, and gravity
- **Precooked Collision Meshes** — Triangle mesh collision cooked at import time and stored as hidden subassets
- **Terrain Collision** — Precooked terrain physics meshes for static world collision
- **Foot IK** — Ground-height raycasting with differential IK corrections for natural foot placement on terrain

### Asset Pipeline
- **Asset Database** — GUID-based asset tracking with `.meta` files, import caching, and compound asset support (subassets with stable GUIDs)
- **Importers** — StaticMesh (YAML + PhysX collision), Mesh (FBX/DAE/OBJ via Assimp), Material, Texture, AnimationClip, Skeleton, Terrain
- **Loaders** — Cache-based asset loading with GUID reference resolution between assets
- **Async Resource Streaming** — Two-phase loading (CPU parse → main-thread GPU upload) with time-budgeted work queue
- **Scene Serialization** — YAML-based scene files with streaming entity spawn

### Audio
- **3D Positional Audio** — XAudio2-based spatial audio with distance attenuation

### Editor
- **WinForms Editor** — Scene hierarchy explorer, basic property inspector, 3D viewport
- **Asset Browser** — Navigate project assets with GUID tracking
- **Scene Loading** — Load and inspect scenes with entity hierarchy

### Animation
- **Clip Playback** — FBX animation clip import with bone-space keyframe sampling
- **Blended Transitions** — Smooth crossfade between animation clips

## Project Structure

```
Freefall/
├── Freefall.Engine/
│   ├── Engine.cs               # Frame lifecycle, entity management, main-thread marshalling
│   ├── Base/                   # Entity-component framework (Entity, ComponentCache, ScriptExecution)
│   ├── Components/             # ECS components (Transform, Camera, Lights, Renderers, Terrain, RigidBody)
│   ├── Graphics/
│   │   ├── GraphicsDevice.cs   # D3D12 device, swap chain, root signature, descriptor heaps
│   │   ├── DeferredRenderer.cs # Render loop orchestration (G-Buffer → Shadows → Light → Compose)
│   │   ├── CommandBuffer.cs    # Thread-safe draw call collection and pass dispatch
│   │   ├── InstanceBatch.cs    # GPU-driven batching, culling, and ExecuteIndirect
│   │   ├── GPUCuller.cs        # 6-pass compute culling pipeline
│   │   ├── MeshRegistry.cs     # Global GPU mesh metadata buffer
│   │   ├── TransformBuffer.cs  # Persistent pooled GPU transform slots
│   │   ├── Material.cs         # Bindless material system with MaterialBlock overrides
│   │   ├── Effect.cs           # Shader compilation, technique/pass management, push constant reflection
│   │   ├── OceanFFT.cs         # GPU compute FFT for ocean displacement, slope, and foam
│   │   └── ...                 # Mesh, Texture, ConstantBuffer, StreamingManager, etc.
│   ├── Animation/              # Skeletal animation, clip playback, bone matrix management
│   ├── Assets/                 # Asset database, importers, loaders, packers, serialization
│   └── Resources/Shaders/      # HLSL shaders (gbuffer, terrain, skybox, cull_instances, etc.)
├── Freefall.Editor/            # WinForms editor (scene explorer, inspector, viewport)
├── Freefall.Game/              # Game runtime (CharacterController, ThirdPersonCamera, etc.)
└── ROADMAP.md                  # Project roadmap and vision
```

## Tech Stack

- .NET 10 / C#
- Direct3D 12 via [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows)
- HLSL compute and graphics shaders (SM 6.6)
- NVIDIA PhysX via [MagicPhysX](https://github.com/Cysharp/MagicPhysX)
- Assimp for mesh importing
- XAudio2 for spatial audio

## Building

```
dotnet build
dotnet run --project Freefall.Game
```

Requires Windows with a D3D12-capable GPU.

## Status

Active development. See [ROADMAP.md](ROADMAP.md) for the project vision and planned features.
