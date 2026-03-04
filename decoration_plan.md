## Decoration Plan

### Problem

Sampling 30 density maps per cell in the MS is expensive and makes prototype selection O(30).

### Solution: CONTROL Prepass

#### 1. Compute Shader: RAW → CONTROL

- **RAW** = existing `Texture2DArray` of 30 density maps (1024×1024 grayscale)
- **CONTROL** = new `Texture2DArray`, **2 slices**, format `RGBA16_UINT`
- Each channel packs: `(decoratorIndex << 8) | weight`
- 2 slices × 4 channels = **8 decorator slots per texel**

Compute dispatch: `(1024/8, 1024/8, 1)` — one thread per texel.

Per texel:
1. Sample all 30 RAW slices
2. Find top 8 by value (insertion sort into 8-element array)
3. Pack `(index << 8) | value` into CONTROL channels
4. Unused slots: write `0xFFFF` (index=255, sentinel)

Run **once** on load / when decorations change (not per frame).

#### 2. AS: Sample CONTROL → Dispatch per decorator

Per tile:
1. Sample CONTROL at tile center (2 texture reads → 8 candidates)
2. For each active decorator (index != 255):
   - Count cells × weight → instance count
   - `DispatchMesh(groups, 1, 1, payload)` with decorator index in payload

This gives **true multi-instance per cell** — grass AND flowers in the same area.

#### 3. MS: Single-decorator instances

Each MS group knows its decorator from the payload. No per-cell selection needed.
Just place instances with jitter, sample heightmap, apply LOD.

### Format Summary

```
CONTROL[slice=0].rgba = { pack(idx0,w0), pack(idx1,w1), pack(idx2,w2), pack(idx3,w3) }
CONTROL[slice=1].rgba = { pack(idx4,w4), pack(idx5,w5), pack(idx6,w6), pack(idx7,w7) }

pack(idx, weight) = (idx << 8) | weight     // 8-bit index, 8-bit weight
unpack_idx(v)     = v >> 8
unpack_weight(v)  = v & 0xFF
```

### Key Design Notes

1. **Instance jitter per decorator** — use `hash(seed + decoratorIndex)` for position jitter.
   Without this, multiple decorators in the same cell stack on the exact same XZ position.

2. **AS group count** — per-decorator dispatch means up to 8× more MS groups per tile.
   In practice most tiles have 1-3 active decorators. Monitor worst case.

3. **vertsPerInstance per decorator** — each MS group gets exact vertex count for its prototype.
   No more wasting threads on MAX vertex count across all types. This is a win.

### Implementation Steps

1. [x] Create CONTROL texture (RGBA16_UINT, 2 slices, 1024×1024) in `TerrainRenderer`
2. [x] Write compute shader `decoration_prepass.hlsl`: RAW → CONTROL
3. [x] Dispatch compute on load / invalidation
4. [x] Rework AS: sample CONTROL, dispatch per-decorator MS groups
5. [x] Rework MS: remove 30-slot loop, use payload decorator index directly
6. [x] Remove old `DecoMapsArray` binding from MS (replaced by CONTROL)
