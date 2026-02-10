# Freefall

Freefall is an experimental game engine written in C# with Direct3D 12, built entirely with [Google Antigravity](https://antigravity.google/blog).

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
| 6 | **Compose** | `Albedo × LightBuffer` → backbuffer blit |

### Key Systems

| System | Description |
|--------|-------------|
| **InstanceBatch** | Core GPU-driven batcher. All per-instance data (descriptors, bounding spheres, subbatch IDs, terrain patches, bones, lights) flows through a unified `PerInstanceBuffer` system with auto-resize. Manages histogram culling pipeline and `ExecuteIndirect`. One batch per Effect. |
| **CommandBuffer** | Thread-safe draw call collector. `Enqueue` dispatches to all applicable passes based on the Effect's declared passes. Thread-local `DrawBucket`s enable lock-free parallel submission. |
| **MeshRegistry** | Global GPU buffer of mesh metadata. Persistent `MeshPartID`s eliminate per-frame CPU grouping — the GPU looks up vertex/index info directly. |
| **TransformBuffer** | Pooled persistent GPU transform slots with dirty-flag uploads. Entities hold a stable `TransformSlot` for their lifetime. |
| **MaterialBlock** | Per-instance parameter overrides. Data is staged into contiguous byte arrays at enqueue time and uploaded as generic per-instance SoA buffers via push constants. |
| **Material / Effect** | Data-driven PSO management. `@RenderState` annotations in shaders auto-configure blend, depth, raster state. `MasterEffects` pattern for global parameter broadcast. |
| **GPUCuller** | 6-pass compute pipeline: Clear → Visibility → Histogram → PrefixSum → Scatter → CommandGen. Shared between camera and shadow passes. |
| **DeferredRenderer** | Orchestrates the G-Buffer, shadow atlas, light accumulation, and composition passes. |

## Features

- **GPU-Driven Rendering** — Indirect draw calls with compute-based visibility culling and instance scatter
- **Deferred Shading** — GBuffer-based pipeline with directional and point light support
- **Cascaded Shadow Maps** — 4-cascade PSSM with texel snapping, bounding-sphere stabilization, and Vogel disc filtering
- **Hi-Z Occlusion Culling** — Hierarchical depth buffer for GPU-side occlusion tests
- **Unified Per-Instance Buffers** — All per-instance data (transforms, materials, bounding spheres, terrain patches, bones, lights) flows through a single generic SoA channel system with auto-resize and push-constant binding
- **Persistent Transform Buffer** — Pooled GPU transform slots with dirty-flag uploads
- **Skeletal Animation** — GPU skinning via per-instance bone buffers (registered as generic SoA channels)
- **Terrain** — Quadtree LOD with splatmap-based multi-texture blending, integrated into the standard InstanceBatch pipeline
- **GPU Terrain** — Fully GPU-driven restricted quadtree with compute-based node evaluation, CDLOD morphing, and indirect rendering
- **Point Lights** — Deferred point lights via per-instance `StructuredBuffer`, rendered as sphere volumes with additive blending
- **Bindless SM 6.6** — All resources accessed via `ResourceDescriptorHeap` and push constants; no Input Assembler
- **Async Resource Streaming** — Two-phase loading (CPU parse → main-thread GPU upload) with time-budgeted work queue
- **Shader System** — Custom FX parser with automatic render pass and pipeline state management
- **Debug Visualization** — 10-mode diagnostic overlay: cascade colors, shadow factor, depth, normals, occlusion x-ray (F5 to cycle)
- **Performance Profiling** — Per-phase CPU timing instrumentation (Prepare, Draw, Upload, Shadow, Opaque, Light, Compose)

## Project Structure

```
Freefall/
├── Program.cs              # Entry point, Win32 message loop
├── Engine.cs               # Frame lifecycle, entity management, main-thread marshalling
├── Base/                   # Entity-component framework (Entity, ComponentCache, ScriptExecution)
├── Components/             # ECS components (Transform, Camera, Lights, Renderers, Terrain, GPUTerrain)
├── Graphics/
│   ├── GraphicsDevice.cs   # D3D12 device, swap chain, root signature, descriptor heaps
│   ├── DeferredRenderer.cs # Render loop orchestration (G-Buffer → Shadows → Light → Compose)
│   ├── CommandBuffer.cs    # Thread-safe draw call collection and pass dispatch
│   ├── InstanceBatch.cs    # GPU-driven batching, culling, and ExecuteIndirect
│   ├── GPUCuller.cs        # 6-pass compute culling pipeline
│   ├── MeshRegistry.cs     # Global GPU mesh metadata buffer
│   ├── TransformBuffer.cs  # Persistent pooled GPU transform slots
│   ├── Material.cs         # Bindless material system with MaterialBlock overrides
│   ├── Effect.cs           # Shader compilation, technique/pass management
│   └── ...                 # Mesh, Texture, ConstantBuffer, StreamingManager, etc.
├── Animation/              # Skeletal animation, clip playback, bone matrix management
├── Resources/Shaders/      # HLSL shaders (gbuffer, gputerrain, terrain, skybox, cull_instances, terrain_quadtree, etc.)
├── Scripts/                # Gameplay scripts (CharacterController, ThirdPersonCamera, etc.)
└── Assets/                 # Runtime assets (scenes, textures, models)
```

## Tech Stack

- .NET 10 / C#
- Direct3D 12 via [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows)
- HLSL compute and graphics shaders (SM 6.6)
- Assimp for mesh importing

## Building

```
dotnet build
dotnet run
```

Requires Windows with a D3D12-capable GPU.

## Status

Work in progress. This is a learning project and research platform, not a production engine.
