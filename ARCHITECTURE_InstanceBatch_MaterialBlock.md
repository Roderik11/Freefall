# Unified InstanceBatch: Per-Instance MaterialBlock Pipeline

Components express per-instance data via `MaterialBlock`; the system auto-creates GPU buffers. Unifies terrain, skinned, and static meshes — no special-casing.

## Core Concept

**Scatter moves one `uint` — the original instance index.** All per-instance data stays in place; the VS uses the compacted index to read from any number of per-instance buffers.

```
Enqueue → MaterialBlock params staged in DrawBucket (all params are per-instance by default)
Upload  → each registered param gets its own GPU StructuredBuffer (SoA)
Cull    → CSVisibility checks BoundingSpheres
Scatter → CSGlobalScatter compacts instance indices only
VS      → compactedIndices[SV_InstanceID] → idx into all per-instance buffers
```

```hlsl
uint idx = compactedIndices[offset + SV_InstanceID];
uint transformSlot = transformSlots[idx];
uint materialId    = materialIds[idx];
// Bones are just another per-instance buffer, indexed by transformSlot:
float4x4 bone = BoneBuffer[transformSlot * NumBones + boneIdx];
```

> [!IMPORTANT]
> Per-instance params are **SoA** — each registered param gets its own GPU buffer. The system creates buffers only for params with a registered push constant slot via `RegisterPerInstanceSlot()`.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Simplified Scatter (single-index compaction) | ✅ Complete |
| Phase 2 | Generic Per-Instance Buffers | ✅ Complete |
| Phase 3 | Terrain Integration | Planned |

---

## Phase 1: Simplified Scatter ✅

Replaced dual-output scatter with single-index scatter. `CSGlobalScatter` writes one `uint instanceIdx`, VS double-indirects through input buffers.

**Files modified**: `cull_instances.hlsl`, `gbuffer.fx`, `gbuffer_skinned.fx`, `InstanceBatch.cs`

---

## Phase 2: Generic Per-Instance Buffers ✅

Removed all bone-specific code (`IsSkinned`, `meshBoneBuffers`, `UploadBoneBuffers`, `EnsureBoneBuffer`, `ForwardRenderer`). Replaced with a generic system:

### Registration (startup)
```csharp
// In CommandBuffer.InitializeCuller():
InstanceBatch.RegisterPerInstanceSlot("Bones", pushConstantSlot: 30, elementStride: 64);
// Future: InstanceBatch.RegisterPerInstanceSlot("TerrainPatch", 28, sizeof(TerrainPatchData));
```

### Component API (unchanged)
```csharp
// Skinned — bones are just a MaterialBlock param
_materialBlock.SetParameterArray("Bones", boneMatrices);
CommandBuffer.Enqueue(mesh, i, material, _materialBlock, transformSlot);

// Static — no per-instance params needed
CommandBuffer.Enqueue(mesh, i, material, block, transformSlot);
```

### System Internals
1. **MaterialBlock** uses `Dictionary<int, ParameterValue>` with hashcode keys (Apex pattern)
2. **DrawBucket.Add()** stages all MaterialBlock params into `PerInstanceParams` dictionary
3. **InstanceBatch.MergeFromBucket()** marks matching per-instance buffers dirty
4. **UploadPerInstanceBuffers()** uploads sparse data indexed by transformSlot × elementsPerInstance
5. **Cull/CullShadow** pushes each registered slot's SRV index as a push constant

**Files modified**: `Material.cs` (MaterialBlock), `CommandBuffer.cs` (DrawBucket), `InstanceBatch.cs`  
**Files deleted**: `ForwardRenderer.cs`

---

## Phase 3: Terrain Integration (Planned)

With Phases 1–2, terrain can use the standard pipeline:

- Remove `Bucket`, `BucketSnapshot`, custom draw path from `Terrain.cs`
- Pre-allocate pooled `TransformSlot`s for visible patches
- `Draw()`: quadtree → assign slots → `CommandBuffer.Enqueue` per patch with MaterialBlock
- Terrain gets frustum culling, Hi-Z occlusion, and shadows for free
- Add `Shadow` pass to `terrain.fx`

---

## Verification

- **Phase 1**: Build + visual parity. Shadows and culling unchanged. ✅
- **Phase 2**: Generic per-instance path works for skinned meshes. Static meshes unaffected. Shadows work. ✅
- **Phase 3**: Terrain renders through pipeline. Terrain casts shadows. Hi-Z culls objects behind ridges.
