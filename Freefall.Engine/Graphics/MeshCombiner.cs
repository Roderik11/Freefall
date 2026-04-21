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
        /// Merge all StaticMeshRenderer children of 'root' into combined geometry.
        /// One MeshPart per unique Material reference.
        /// </summary>
        public static MergeResult Combine(Entity root, MergeOptions options = null)
        {
            options ??= new MergeOptions();

            // 1. Collect children
            var children = CollectRenderers(root);
            if (children.Count == 0) return null;

            Matrix4x4.Invert(root.Transform.WorldMatrix, out var rootInverse);

            // 2. Determine max LOD count
            int maxLodCount = 0;
            foreach (var (entity, renderer) in children)
            {
                if (renderer.StaticMesh == null) continue;
                int lods = 1 + (renderer.StaticMesh.LODs?.Count ?? 0);
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
                    var staticMesh = renderer.StaticMesh;
                    if (staticMesh?.Mesh == null) continue;

                    var lodMesh = GetLODMesh(staticMesh, lod, options);
                    if (lodMesh == null) continue;

                    var elements = lodMesh.MeshParts;
                    if (elements == null || elements.Count == 0) continue;

                    var relativeTransform = entity.Transform.WorldMatrix * rootInverse;
                    Matrix4x4.Invert(relativeTransform, out var inv);
                    var normalMatrix = Matrix4x4.Transpose(inv);

                    foreach (var element in elements)
                    {
                        if (element.Material == null) continue;

                        // Resolve which Mesh object this element uses
                        var mesh = element.Mesh ?? lodMesh.Mesh ?? staticMesh.Mesh;
                        if (mesh == null) continue;

                        // Read MeshData (cached per mesh GUID) — only for vertex/index buffers
                        var meshData = GetMeshData(mesh, meshDataCache);
                        if (meshData == null) continue;

                        int partIndex = element.MeshPartIndex;
                        if (partIndex < 0 || partIndex >= mesh.MeshParts.Count)
                            continue;

                        // Use LIVE MeshPart for offsets — cache parts can be in different order
                        var part = mesh.MeshParts[partIndex];

                        // Get or create accumulator for THIS material
                        if (!accumulators.TryGetValue(element.Material, out var acc))
                        {
                            acc = new MergeAccumulator { Material = element.Material };
                            accumulators[element.Material] = acc;
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
        /// Create an in-memory StaticMesh from a MergeResult.
        /// </summary>
        public static StaticMesh CreateStaticMesh(GraphicsDevice device, MergeResult result)
        {
            if (result == null || result.MeshDataPerLOD.Count == 0)
                return null;

            var lod0Data = result.MeshDataPerLOD[0];
            if (lod0Data.Positions == null || lod0Data.Positions.Length == 0)
                return null;

            var staticMesh = new StaticMesh { Name = "Merged" };
            staticMesh.Mesh = new Mesh(device, lod0Data);

            // MeshElement[i] → MeshPart[i] + Material[i]. Direct 1:1.
            var lod0Mats = result.MaterialsPerLOD[0];
            for (int i = 0; i < lod0Data.Parts.Count; i++)
            {
                staticMesh.MeshParts.Add(new MeshElement
                {
                    Mesh = staticMesh.Mesh,
                    Material = i < lod0Mats.Count ? lod0Mats[i] : InternalAssets.DefaultMaterial,
                    MeshPartIndex = i,
                });
            }

            // LODs — each gets its own Mesh + its own material list
            for (int lod = 1; lod < result.MeshDataPerLOD.Count; lod++)
            {
                var lodData = result.MeshDataPerLOD[lod];
                if (lodData.Positions == null || lodData.Positions.Length == 0) continue;

                var lodMesh = new Mesh(device, lodData);
                var lodEntry = new StaticMeshLOD { Mesh = lodMesh };
                var lodMats = result.MaterialsPerLOD[lod];

                for (int i = 0; i < lodData.Parts.Count; i++)
                {
                    lodEntry.MeshParts.Add(new MeshElement
                    {
                        Mesh = lodMesh,
                        Material = i < lodMats.Count ? lodMats[i] : InternalAssets.DefaultMaterial,
                        MeshPartIndex = i,
                    });
                }

                staticMesh.LODs.Add(lodEntry);
            }

            staticMesh.LODGroup = LODGroups.LargeProps;
            staticMesh.Mesh.RegisterMeshParts();
            staticMesh.MarkReady();
            return staticMesh;
        }

        // ─── Helpers ───────────────────────────────────────────────

        private static List<(Entity entity, StaticMeshRenderer renderer)> CollectRenderers(Entity root)
        {
            var result = new List<(Entity, StaticMeshRenderer)>();
            void Walk(Entity e, bool isRoot)
            {
                if (!isRoot)
                {
                    var r = e.GetComponent<StaticMeshRenderer>();
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

        private static IStaticMesh GetLODMesh(StaticMesh sm, int lodLevel, MergeOptions options)
        {
            int available = 1 + (sm.LODs?.Count ?? 0);

            if (lodLevel == 0) return sm;
            if (lodLevel < available) return sm.LODs[lodLevel - 1];

            // Source doesn't have this LOD
            if (options.NoLodHandling == NoLodStrategy.IncludeInAll)
                return sm.LODs?.Count > 0 ? sm.LODs[^1] : sm;

            return null;
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
