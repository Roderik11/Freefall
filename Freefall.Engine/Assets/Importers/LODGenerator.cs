using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Graphics;
using static Freefall.Graphics.MeshOptimizer;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Makes a LOD chain for a mesh that has no LODs of its own (no _LOD1, _LOD2… objects in the file).
    /// Every drawn part is simplified with meshoptimizer, each level from the one before, to a share of
    /// its LOD 0 triangles. The simplified levels reuse the mesh's vertices and only add indices.
    ///
    /// Kept as they are: UV seams and hard edges (vertices that share a position but not their UV or
    /// normal are never merged) and open borders (also the edges between parts of different materials, so
    /// the parts still meet). The normals weigh a little into the error.
    ///
    /// Each level may change the shape by at most the importer's LODPixelError, measured where the level takes
    /// over (meshoptimizer's error estimate, which runs well above what shows: on the pack rocks levels of
    /// 3-4 px looked the same as LOD 0, 8-10 px held up at their switch size, 24 px broke the silhouette).
    /// </summary>
    internal static unsafe class LODGenerator
    {
        // How much a change of the vertex normals counts against a change of shape (meshoptimizer's attribute
        // weight; 1 makes a normal change of 1 count like a move across the whole mesh). Low on purpose: on the
        // faceted pack rocks a weight of 1 spent the whole error limit on normals and the chains stalled at
        // 70-80 % of the triangles. The detail of such meshes is in their normal maps; the shape is what counts,
        // and hard edges and UV seams are kept anyway.
        private const float NormalWeight = 0.1f;

        // A level must cut at least this share of the triangles of the level before, or the chain ends:
        // the simplifier is stuck on seams, borders or the error limit and the level would cost a draw for nothing.
        private const float MinReduction = 0.15f;

        // Screen height the pixel error is measured on
        private const float ReferenceScreenHeight = 1080f;

        private static bool _libraryMissing;

        private sealed class Source
        {
            public int Part;            // LOD 0 part index
            public int Triangles0;
            public int VertexStart, VertexCount;
            public uint[] Indices;      // current level, relative to VertexStart
            public int BaseIndex = -1;  // where Indices sit in the index list, once appended
            public float Error;         // accumulated absolute error
        }

        /// <summary>
        /// Generate LODs for one mesh. Appends the simplified parts to <paramref name="parts"/> and their indices
        /// to <paramref name="indices"/>. Returns the LODs (LOD 0 = the drawn parts as they are), or an empty
        /// list when no level was worth making.
        /// </summary>
        /// <param name="ratios">Share of LOD 0's triangles per level, decreasing.</param>
        /// <param name="minTriangles">The chain ends before a level that would have fewer triangles than this.</param>
        /// <param name="pixelError">
        /// Largest shape change of a level, in pixels of a 1080p screen at the size where the level takes over. A level
        /// that cannot reach its ratio within it keeps more triangles. 0 = no limit.
        /// </param>
        public static List<MeshLOD> Generate(List<MeshPart> parts, List<Vector3> positions, List<Vector3> normals, List<uint> indices,
            IReadOnlyList<float> ratios, int minTriangles, float pixelError, string meshName)
        {
            var none = new List<MeshLOD>();
            if (_libraryMissing || ratios == null || ratios.Count == 0 || positions.Count == 0 || normals.Count != positions.Count)
                return none;

            // LOD 0: every drawn part (collision parts are not drawn and stay outside the chain)
            var sources = new List<Source>();
            int triangles0 = 0;
            for (int p = 0; p < parts.Count; p++)
            {
                var part = parts[p];
                if (!part.Enabled || part.NumIndices < 3) continue;

                int min = int.MaxValue, max = int.MinValue;
                for (int i = part.BaseIndex; i < part.BaseIndex + part.NumIndices; i++)
                {
                    int v = (int)indices[i];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                var local = new uint[part.NumIndices - part.NumIndices % 3];
                for (int i = 0; i < local.Length; i++)
                    local[i] = (uint)((int)indices[part.BaseIndex + i] - min);

                sources.Add(new Source
                {
                    Part = p,
                    Triangles0 = local.Length / 3,
                    VertexStart = min,
                    VertexCount = max - min + 1,
                    Indices = local,
                    BaseIndex = part.BaseIndex,
                });
                triangles0 += local.Length / 3;
            }
            if (sources.Count == 0) return none;

            var lods = new List<MeshLOD> { new MeshLOD { MeshPartIndices = sources.ConvertAll(s => s.Part).ToArray() } };
            var counts = new List<int> { triangles0 };

            var posSpan = CollectionsMarshal.AsSpan(positions);
            var normSpan = CollectionsMarshal.AsSpan(normals);
            float* weights = stackalloc float[3] { NormalWeight, NormalWeight, NormalWeight };

            try
            {
                fixed (Vector3* pos = posSpan)
                fixed (Vector3* norm = normSpan)
                {
                    // The mesh's bounding sphere, as the renderer measures screen size (box half-diagonal)
                    Vector3 boxMin = new(float.MaxValue), boxMax = new(float.MinValue);
                    foreach (var p in posSpan) { boxMin = Vector3.Min(boxMin, p); boxMax = Vector3.Max(boxMax, p); }
                    float diameter = Vector3.Distance(boxMin, boxMax);

                    float previousRatio = 1f;
                    int previousTriangles = triangles0;
                    var levelPixels = new List<float>();

                    foreach (float ratio in ratios)
                    {
                        if (!(ratio > 0f && ratio < previousRatio)) continue;
                        if (triangles0 * ratio < minTriangles) break;

                        // Errors are absolute (metres), so every part of the mesh gets the same limit: the level's
                        // deviation from LOD 0 may cover pixelError pixels of a 1080p screen where the level takes
                        // over. Mesh.ComputeLODScreenSizes puts that at LODReferenceScreenSize * sqrt(ratio), where
                        // one pixel spans diameter / (screen size * 1080) metres at any field of view.
                        float pixel = diameter / (Mesh.LODReferenceScreenSize * MathF.Sqrt(ratio) * ReferenceScreenHeight);
                        float errorLimit = pixelError > 0f ? pixelError * pixel : float.MaxValue;

                        // Simplify every part from its previous level
                        var levelIndices = new uint[sources.Count][];
                        var levelErrors = new float[sources.Count];
                        int levelTriangles = 0;
                        for (int s = 0; s < sources.Count; s++)
                        {
                            var src = sources[s];
                            levelIndices[s] = src.Indices;
                            levelErrors[s] = src.Error;

                            int target = Math.Max(1, (int)MathF.Round(src.Triangles0 * ratio)) * 3;
                            float budget = errorLimit - src.Error;
                            if (src.Indices.Length > target && budget > 0f)
                            {
                                var result = new uint[src.Indices.Length];
                                float error = 0f;
                                nuint count;
                                fixed (uint* dst = result)
                                fixed (uint* idx = src.Indices)
                                {
                                    count = meshopt_simplifyWithAttributes(dst, idx, (nuint)src.Indices.Length,
                                        (float*)(pos + src.VertexStart), (nuint)src.VertexCount, (nuint)sizeof(Vector3),
                                        (float*)(norm + src.VertexStart), (nuint)sizeof(Vector3), weights, 3,
                                        null, (nuint)target, budget, SimplifyLockBorder | SimplifyErrorAbsolute, &error);
                                    if (count >= 3 && (int)count < src.Indices.Length)
                                        meshopt_optimizeVertexCache(dst, dst, count, (nuint)src.VertexCount);
                                }
                                if (count >= 3 && (int)count < src.Indices.Length)
                                {
                                    Array.Resize(ref result, (int)count);
                                    levelIndices[s] = result;
                                    levelErrors[s] = src.Error + error;
                                }
                            }
                            levelTriangles += levelIndices[s].Length / 3;
                        }

                        if (levelTriangles > previousTriangles * (1f - MinReduction)) break;

                        // Append the level: one part per LOD 0 part, same material slot (the GPU LOD chain links
                        // parts by slot); a part that could not shrink reuses its previous index range.
                        int level = lods.Count;
                        var levelParts = new int[sources.Count];
                        for (int s = 0; s < sources.Count; s++)
                        {
                            var src = sources[s];
                            if (!ReferenceEquals(levelIndices[s], src.Indices))
                            {
                                src.Indices = levelIndices[s];
                                src.Error = levelErrors[s];
                                src.BaseIndex = indices.Count;
                                foreach (uint i in src.Indices)
                                    indices.Add(i + (uint)src.VertexStart);
                            }

                            var basePart = parts[src.Part];
                            levelParts[s] = parts.Count;
                            parts.Add(new MeshPart
                            {
                                Name = LevelName(basePart.Name, level),
                                Enabled = true,
                                BaseVertex = basePart.BaseVertex,
                                BaseIndex = src.BaseIndex,
                                NumIndices = src.Indices.Length,
                                MaterialSlot = basePart.MaterialSlot,
                                BoundingBox = basePart.BoundingBox,
                                BoundingSphere = basePart.BoundingSphere,
                            });
                        }

                        lods.Add(new MeshLOD { MeshPartIndices = levelParts });
                        counts.Add(levelTriangles);
                        float worst = 0f;
                        foreach (var src in sources) worst = MathF.Max(worst, src.Error);
                        levelPixels.Add(worst / pixel);
                        previousRatio = ratio;
                        previousTriangles = levelTriangles;
                    }

                    if (lods.Count > 1)
                        Debug.Log("ModelImporter", $"Generated LODs for '{meshName}': {string.Join(" → ", counts)} tris " +
                                                   $"(error {string.Join(" / ", levelPixels.ConvertAll(p => p.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)))} px)");
                }
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                // Thrown by the first native call, before anything was appended
                _libraryMissing = true;
                Debug.LogWarning("ModelImporter", $"meshoptimizer.dll missing or too old, meshes are imported without generated LODs: {ex.Message}");
                return none;
            }

            return lods.Count > 1 ? lods : none;
        }

        /// <summary>"Rock_A" → "Rock_A_LOD2", "Rock_A [Stone]" → "Rock_A_LOD2 [Stone]".</summary>
        private static string LevelName(string partName, int level)
        {
            int bracket = partName.IndexOf(" [", StringComparison.Ordinal);
            string name = bracket >= 0 ? partName.Substring(0, bracket) : partName;
            string suffix = bracket >= 0 ? partName.Substring(bracket) : string.Empty;
            return $"{ModelImporter.StripLODSuffix(name)}_LOD{level}{suffix}";
        }
    }
}
