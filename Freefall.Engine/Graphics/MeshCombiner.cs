using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    public enum NoLodStrategy { IncludeInAll, LOD0Only }

    public class MergeOptions
    {
        public NoLodStrategy NoLodHandling { get; set; } = NoLodStrategy.IncludeInAll;
    }

    public class MergeResult
    {
        public List<MeshData> MeshDataPerLOD = new();
        public List<List<Material>> MaterialsPerLOD = new();
        public int SourceCount;
    }

    public static class MeshCombiner
    {
        private class MergeAccumulator
        {
            public Material Material;
            public List<Vector3> Positions = new();
            public List<Vector3> Normals = new();
            public List<Vector2> UVs = new();
            public List<uint> Indices = new();
        }

        /// <summary>
        /// Merge all MeshRenderer children of 'root' into combined geometry.
        /// One MeshPart per unique Material reference.
        /// </summary>
        public static MergeResult Combine(Entity root, MergeOptions options = null)
        {
            options ??= new MergeOptions();

            // 1. Collect children
            var children = CollectRenderers(root);
            if (children.Count == 0) return null;

            Matrix4x4.Invert(root.Transform.WorldMatrix, out var rootInverse);

            // 2. Determine max LOD count across all child meshes
            int maxLodCount = 1;
            foreach (var (entity, renderer) in children)
            {
                if (renderer.Mesh == null) continue;
                int lods = Math.Max(1, renderer.Mesh.LODs.Count);
                maxLodCount = Math.Max(maxLodCount, lods);
            }

            Debug.Log($"[MeshCombiner] {children.Count} renderers, {maxLodCount} LOD levels");

            // 3. Cache MeshData reads
            var meshDataCache = new Dictionary<string, MeshData>();

            // 4. Merge each LOD level
            var result = new MergeResult { SourceCount = children.Count };

            for (int lod = 0; lod < maxLodCount; lod++)
            {
                // Accumulators keyed by Material reference
                var accumulators = new Dictionary<Material, MergeAccumulator>();

                foreach (var (entity, renderer) in children)
                {
                    var mesh = renderer.Mesh;
                    if (mesh == null) continue;

                    // Determine which MeshPart indices to use for this LOD level
                    int[] partIndices = GetLodPartIndices(mesh, lod, options);
                    if (partIndices == null || partIndices.Length == 0) continue;

                    // Read MeshData (cached per mesh GUID)
                    var meshData = GetMeshData(mesh, meshDataCache);
                    if (meshData == null) continue;

                    var relativeTransform = entity.Transform.WorldMatrix * rootInverse;
                    Matrix4x4.Invert(relativeTransform, out var inv);
                    var normalMatrix = Matrix4x4.Transpose(inv);

                    foreach (var partIdx in partIndices)
                    {
                        if (partIdx >= mesh.MeshParts.Count) continue;
                        var part = mesh.MeshParts[partIdx];

                        // Resolve material using MeshRenderer's logic
                        var mat = ResolveMaterial(renderer, part.MaterialSlot);
                        if (mat == null) continue;

                        // Get or create accumulator for THIS material
                        if (!accumulators.TryGetValue(mat, out var acc))
                        {
                            acc = new MergeAccumulator { Material = mat };
                            accumulators[mat] = acc;
                        }

                        AppendPartGeometry(acc, meshData, part, relativeTransform, normalMatrix);
                    }
                }

                var (meshDataResult, materials) = BuildMergedMeshData(accumulators, null);
                result.MeshDataPerLOD.Add(meshDataResult);
                result.MaterialsPerLOD.Add(materials);
            }

            var lod0 = result.MeshDataPerLOD[0];
            Debug.Log($"[MeshCombiner] LOD0: {lod0?.Positions?.Length ?? 0} verts, " +
                      $"{(lod0?.Indices?.Length ?? 0) / 3} tris, {lod0?.Parts?.Count ?? 0} parts, " +
                      $"{result.MaterialsPerLOD[0].Count} materials");

            return result;
        }

        /// <summary>
        /// Create an in-memory Mesh + material list from a MergeResult.
        /// </summary>
        public static (Mesh mesh, List<MaterialOverride> materials) CreateMergedMesh(
            GraphicsDevice device, MergeResult result)
        {
            if (result == null || result.MeshDataPerLOD.Count == 0)
                return (null, null);

            var lod0Data = result.MeshDataPerLOD[0];
            if (lod0Data.Positions == null || lod0Data.Positions.Length == 0)
                return (null, null);

            // Build LOD0 mesh
            var mesh = new Mesh(device, lod0Data);

            // Build material overrides from LOD0 materials
            var lod0Mats = result.MaterialsPerLOD[0];
            var materials = new List<MaterialOverride>();
            for (int i = 0; i < lod0Data.Parts.Count; i++)
            {
                materials.Add(new MaterialOverride
                {
                    MaterialSlot = lod0Data.Parts[i].MaterialSlot,
                    Material = i < lod0Mats.Count ? lod0Mats[i] : InternalAssets.DefaultMaterial,
                });
            }

            // TODO: LOD merging — for now, merged meshes have LOD0 only.
            // Per-mesh LODs would require building additional MeshLOD entries
            // from result.MeshDataPerLOD[1..N], which means appending their
            // vertex/index data to the same Mesh buffer. This is non-trivial
            // and rarely needed (merge is typically for static scenery).

            mesh.RegisterMeshParts();
            return (mesh, materials);
        }

        // ─── Helpers ───────────────────────────────────────────────

        private static List<(Entity entity, MeshRenderer renderer)> CollectRenderers(Entity root)
        {
            var result = new List<(Entity, MeshRenderer)>();
            void Walk(Entity e, bool isRoot)
            {
                if (!isRoot)
                {
                    var r = e.GetComponent<MeshRenderer>();
                    if (r != null) result.Add((e, r));
                }
                for (int i = 0; i < e.Transform.GetChildCount(); i++)
                {
                    var child = e.Transform.GetChild(i);
                    if (child?.Entity != null)
                        Walk(child.Entity, false);
                }
            }
            Walk(root, true);
            return result;
        }

        /// <summary>
        /// Get the MeshPart indices for a given LOD level from a Mesh.
        /// Handles fallback when a mesh has fewer LOD levels than requested.
        /// </summary>
        private static int[] GetLodPartIndices(Mesh mesh, int lodLevel, MergeOptions options)
        {
            if (mesh.LODs.Count == 0)
            {
                // No LODs — return all parts for LOD0, or based on strategy
                if (lodLevel == 0 || options.NoLodHandling == NoLodStrategy.IncludeInAll)
                    return Enumerable.Range(0, mesh.MeshParts.Count).ToArray();
                return null;
            }

            if (lodLevel < mesh.LODs.Count)
                return mesh.LODs[lodLevel].MeshPartIndices;

            // Source doesn't have this LOD level
            if (options.NoLodHandling == NoLodStrategy.IncludeInAll)
                return mesh.LODs[^1].MeshPartIndices; // Use lowest available LOD

            return null;
        }

        /// <summary>
        /// Resolve material for a MeshPart slot using MeshRenderer's override chain.
        /// </summary>
        private static Material ResolveMaterial(MeshRenderer renderer, int materialSlot)
        {
            // Check sparse overrides first
            if (renderer.Materials != null)
            {
                for (int i = 0; i < renderer.Materials.Count; i++)
                {
                    if (renderer.Materials[i].MaterialSlot == materialSlot)
                        return renderer.Materials[i].Material;
                }
            }

            // Fall back to default Material
            return renderer.Material;
        }

        private static MeshData GetMeshData(Mesh mesh, Dictionary<string, MeshData> cache)
        {
            var key = mesh.Guid ?? mesh.GetHashCode().ToString();
            if (cache.TryGetValue(key, out var cached))
                return cached;

            MeshData data = null;

            // Try cache file first
            if (!string.IsNullOrEmpty(mesh.Guid))
            {
                try
                {
                    var path = AssetDatabase.ResolveCachePathByGuid(mesh.Guid, typeof(MeshData));
                    if (path != null && File.Exists(path))
                    {
                        var packer = new MeshPacker();
                        using var stream = File.OpenRead(path);
                        data = packer.Read(stream);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("MeshCombiner", $"Cache read failed for '{mesh.Name}': {ex.Message}");
                }
            }

            // Fallback: retained CPU data
            if (data == null && mesh.Positions != null && mesh.CpuIndices != null)
            {
                var normals = new Vector3[mesh.Positions.Length];
                Array.Fill(normals, Vector3.UnitY);
                data = new MeshData
                {
                    Positions = mesh.Positions,
                    Normals = normals,
                    UVs = new Vector2[mesh.Positions.Length],
                    Indices = mesh.CpuIndices,
                    Parts = mesh.MeshParts.ToList(),
                };
            }

            cache[key] = data;
            return data;
        }

        private static void AppendPartGeometry(
            MergeAccumulator acc, MeshData meshData, MeshPart part,
            Matrix4x4 transform, Matrix4x4 normalMatrix)
        {
            int indexStart = part.BaseIndex;
            int indexEnd = Math.Min(indexStart + part.NumIndices, meshData.Indices.Length);
            int baseVertex = part.BaseVertex;

            // Find actual vertex range (indices + BaseVertex)
            int minVert = int.MaxValue, maxVert = int.MinValue;
            for (int i = indexStart; i < indexEnd; i++)
            {
                int v = (int)meshData.Indices[i] + baseVertex;
                minVert = Math.Min(minVert, v);
                maxVert = Math.Max(maxVert, v);
            }
            if (minVert > maxVert) return;

            int vertexOffset = acc.Positions.Count;

            // Copy and transform vertices
            for (int v = minVert; v <= maxVert; v++)
            {
                acc.Positions.Add(v < meshData.Positions.Length
                    ? Vector3.Transform(meshData.Positions[v], transform)
                    : Vector3.Zero);

                acc.Normals.Add(v < meshData.Normals.Length
                    ? Vector3.Normalize(Vector3.TransformNormal(meshData.Normals[v], normalMatrix))
                    : Vector3.UnitY);

                acc.UVs.Add(v < meshData.UVs.Length
                    ? meshData.UVs[v]
                    : Vector2.Zero);
            }

            // Copy indices, rebased
            for (int i = indexStart; i < indexEnd; i++)
            {
                int actual = (int)meshData.Indices[i] + baseVertex;
                acc.Indices.Add((uint)(vertexOffset + actual - minVert));
            }
        }

        /// <summary>
        /// Build final MeshData. Materials list order = Parts list order. Direct 1:1.
        /// </summary>
        /// <param name="materialOrder">If non-null, forces parts to be emitted in this order.
        /// Materials not present get empty parts skipped. If null, order is determined by accumulators.</param>
        private static (MeshData, List<Material>) BuildMergedMeshData(
            Dictionary<Material, MergeAccumulator> accumulators,
            List<Material> materialOrder)
        {
            var allPositions = new List<Vector3>();
            var allNormals = new List<Vector3>();
            var allUVs = new List<Vector2>();
            var allIndices = new List<uint>();
            var parts = new List<MeshPart>();
            var materials = new List<Material>();

            var meshMin = new Vector3(float.MaxValue);
            var meshMax = new Vector3(float.MinValue);

            // Determine iteration order: use explicit materialOrder if provided,
            // otherwise build from accumulators (LOD0 path)
            var orderedMaterials = materialOrder ?? accumulators.Keys.ToList();

            foreach (var mat in orderedMaterials)
            {
                if (!accumulators.TryGetValue(mat, out var acc) || acc.Positions.Count == 0)
                    continue;

                int baseIndex = allIndices.Count;
                int baseVertex = allPositions.Count;

                // Bounding box
                var pMin = new Vector3(float.MaxValue);
                var pMax = new Vector3(float.MinValue);
                foreach (var p in acc.Positions)
                {
                    pMin = Vector3.Min(pMin, p);
                    pMax = Vector3.Max(pMax, p);
                }
                meshMin = Vector3.Min(meshMin, pMin);
                meshMax = Vector3.Max(meshMax, pMax);

                var center = (pMin + pMax) * 0.5f;
                var radius = (pMax - center).Length();

                // Offset indices by global vertex base
                foreach (var idx in acc.Indices)
                    allIndices.Add(idx + (uint)baseVertex);

                allPositions.AddRange(acc.Positions);
                allNormals.AddRange(acc.Normals);
                allUVs.AddRange(acc.UVs);

                parts.Add(new MeshPart
                {
                    Name = mat.Name ?? $"Part_{parts.Count}",
                    Enabled = true,
                    BaseVertex = 0,
                    BaseIndex = baseIndex,
                    NumIndices = acc.Indices.Count,
                    MaterialSlot = parts.Count,
                    BoundingBox = new BoundingBox(pMin, pMax),
                    BoundingSphere = new Vector4(center, radius),
                });

                materials.Add(mat);
            }

            if (allPositions.Count == 0)
                return (new MeshData(), materials);

            var meshData = new MeshData
            {
                Positions = allPositions.ToArray(),
                Normals = allNormals.ToArray(),
                UVs = allUVs.ToArray(),
                Indices = allIndices.ToArray(),
                Parts = parts,
                BoundingBox = new BoundingBox(meshMin, meshMax),
            };

            return (meshData, materials);
        }
    }
}
