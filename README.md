# Freefall

Freefall is an experimental game engine written in C# with Direct3D 12, built entirely with [Google Antigravity](https://blog.google/technology/google-deepmind/antigravity-ai-code-editor/).

## Features

- **GPU-Driven Rendering** — Indirect draw calls with compute-based visibility culling and instance scatter
- **Deferred Shading** — GBuffer-based pipeline with directional and point light support
- **Cascaded Shadow Maps** — 4-cascade PSSM with configurable lambda blending and texel snapping
- **Hi-Z Occlusion Culling** — Hierarchical depth buffer for GPU-side occlusion tests
- **Instance Batching** — Automatic draw call merging with per-material sub-batches
- **Persistent Transform Buffer** — Pooled GPU transform slots with dirty-flag uploads
- **Skeletal Animation** — Bone matrix manager with GPU skinning support
- **Terrain** — Quadtree LOD with splatmap-based multi-texture blending
- **Async Resource Streaming** — Two-phase loading (CPU parse → main-thread GPU upload) with time-budgeted work queue
- **Shader System** — Custom FX parser with automatic render pass and pipeline state management
- **Debug Visualization** — Cascade colors, shadow factor, depth, occlusion x-ray (F5 to cycle)

## Tech Stack

- .NET 10 / C#
- Direct3D 12 via [Vortice.Windows](https://github.com/amerkoleci/Vortice.Windows)
- HLSL compute and graphics shaders
- Assimp for mesh importing

## Building

```
dotnet build
dotnet run
```

Requires Windows with a D3D12-capable GPU.

## Status

Work in progress. This is a learning project and research platform, not a production engine.
