# Unified InstanceBatch: InstanceDescriptor + Per-Instance MaterialBlock

Refactor `InstanceBatch` from hardcoded parallel arrays to a single compacted-index pattern. Components express per-instance data via `MaterialBlock`; the system auto-creates GPU buffers. Unifies terrain, skinned, and static meshes — terrain gets occlusion culling and shadows with zero special-casing.

## Core Concept

**Scatter moves one `uint` — the original instance index.** All per-instance data stays in place; the VS uses the compacted index to read from any number of per-instance buffers.

```
Enqueue → per-instance params written to per-param staging arrays at instanceIdx
Upload  → block-copy each staging array to its GPU buffer
Cull    → CSVisibility checks BoundingSpheres (unchanged)
Scatter → CSGlobalScatter compacts instance indices only (1 write per visible instance)
VS      → compactedIndices[SV_InstanceID] → idx into all per-instance buffers
```

```hlsl
uint idx = compactedIndices[offset + SV_InstanceID];
// Each per-instance param is its own buffer, all indexed by the same idx
uint transformSlot = transformSlots[idx];
uint materialId    = materialIds[idx];
// Effect-specific — only terrain reads this:
TerrainPatch patch = terrainPatches[idx];
```

> [!IMPORTANT]
> Per-instance params are **SoA** — each declared param gets its own GPU buffer. But grouping is the **Effect's choice**: `TerrainPatch` stays as one struct if the shader reads all fields together. The system doesn't force splitting or packing.

---

## Phase 1: Simplified Scatter

Replace the dual-output scatter with single-index scatter. Keep existing per-instance arrays as-is initially.

#### [MODIFY] [cull_instances.hlsl](file:///d:/Projects/2026/Freefall/Resources/Shaders/cull_instances.hlsl)
- `CSGlobalScatter`: output one `uint` (instanceIdx) instead of scatterring TransformSlot + MaterialId
- `CSVisibility`/`CSVisibilityShadow`: unchanged (still reads bounding spheres + transforms for culling)
- `CSMain`: update `IndirectDrawCommand` to reference per-instance buffers directly

#### [MODIFY] [gbuffer.fx](file:///d:/Projects/2026/Freefall/Resources/Shaders/gbuffer.fx)
- VS reads `compactedIndices[offset + SV_InstanceID]` → use that index into TransformSlots and MaterialIds buffers

#### [MODIFY] [InstanceBatch.cs](file:///d:/Projects/2026/Freefall/Graphics/InstanceBatch.cs)
- Remove `scatteredMaterialBuffers[]` and related SRV/UAV pairs (scatter no longer writes them)
- Shadow path: same simplification (remove `shadowScatteredMaterial*` arrays)
- Keep TransformSlots, MaterialIds, BoundingSpheres, SubbatchIds as separate arrays (Phase 2 generalizes them)

---

## Phase 2: Per-Instance MaterialBlock Buffers

Make `MaterialBlock` the sole interface for per-instance data. System auto-creates GPU buffers.

### Component API
```csharp
// Terrain — just sets params. No GPU buffers, no indices.
var block = new MaterialBlock();
block.SetParameter("TerrainPatch", patchData);  // struct, stays as one buffer
CommandBuffer.Enqueue(patchMesh, 0, terrainMaterial, block, pooledTransformSlot);

// Skinned — bone index is just another per-instance param
block.SetParameter("BoneBufferIdx", boneIdx);
CommandBuffer.Enqueue(mesh, i, material, block, transformSlot);
```

### System Internals
1. **Effect** declares per-instance params via shader metadata (e.g., `// [PerInstance] TerrainPatch`)
2. **At `Enqueue`**: `DrawBucket` writes each per-instance param value into its own staging array at `instanceIdx` — one write per param, same pattern as current TransformSlot write
3. **At upload**: each staging array is block-copied to its own GPU `StructuredBuffer`
4. **Per-batch push constant**: each buffer's SRV index is written into `IndirectDrawCommand`

> [!NOTE]
> TransformSlot and MaterialId are just two of N per-instance param buffers — not architecturally special. They happen to exist on every Effect, but the pipeline treats them identically to custom params.

#### [NEW] [PerInstanceLayout.cs](file:///d:/Projects/2026/Freefall/Graphics/PerInstanceLayout.cs)
- Describes per-instance params for an Effect (name, type, size)
- Parsed from shader metadata at Effect load time

#### [MODIFY] [CommandBuffer.cs](file:///d:/Projects/2026/Freefall/Graphics/CommandBuffer.cs)
- `DrawBucket`: N staging arrays (one per per-instance param), sized dynamically
- `Enqueue`: pack MaterialBlock values into staging arrays

#### [MODIFY] [InstanceBatch.cs](file:///d:/Projects/2026/Freefall/Graphics/InstanceBatch.cs)
- `UploadInstanceData`: upload N per-instance buffers (driven by `PerInstanceLayout`)
- Remove hardcoded TransformSlot/MaterialId buffer management — folded into generic system

---

## Phase 3: Terrain Integration

With Phases 1–2, terrain uses standard pipeline.

#### [MODIFY] [Terrain.cs](file:///d:/Projects/2026/Freefall/Components/Terrain.cs)
- Remove `Bucket`, `BucketSnapshot`, custom draw path
- Pre-allocate pooled `TransformSlot`s (sized to max visible patches)
- `Draw()`: quadtree selects → assign pooled slots → `CommandBuffer.Enqueue` per patch with MaterialBlock
- Terrain gets frustum culling, Hi-Z occlusion, *and* shadows for free

#### [MODIFY] [terrain.fx](file:///d:/Projects/2026/Freefall/Resources/Shaders/terrain.fx)
- VS reads `TerrainPatch` from per-instance buffer via compacted index (replaces push constants)
- Add `Shadow` pass (depth-only VS, no PS)

---

## Phase 4: Cleanup

- Remove `IsSkinned` — bones are a per-instance param like any other
- Remove redundant buffer arrays (`scatteredMaterial*`, `shadowScatteredMaterial*`)
- Consolidate shadow buffer management

---

## Verification

- **Phase 1**: Build + visual parity with current rendering. Shadows and culling unchanged.
- **Phase 2**: New per-instance path works for existing meshes. MaterialBlock API unchanged for components.
- **Phase 3**: Terrain renders through pipeline. Objects behind terrain ridges are Hi-Z culled. Terrain casts shadows.
- **Each phase compiles and runs independently** — no big-bang refactor.
