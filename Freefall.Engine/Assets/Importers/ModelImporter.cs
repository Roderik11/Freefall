using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.ComponentModel;
using Assimp;
using Freefall.Animation;
using Freefall.Graphics;
using Matrix4x4 = System.Numerics.Matrix4x4;
using Bone = Freefall.Animation.Bone;
using VectorKey = Freefall.Animation.VectorKey;
using QuaternionKey = Freefall.Animation.QuaternionKey;

namespace Freefall.Assets.Importers
{
    [Serializable]
    public class MeshPartConfig
    {
        [ReadOnly(true)]
        public string Name = string.Empty;
        public bool Render = true;
        public bool Collision = false;
        public int LODMin = -1;
        public int LODMax = -1;

        public override string ToString() => Name;
    }

    /// <summary>
    /// Unified model importer for FBX, DAE, OBJ files.
    /// Produces all artifacts from a single source: mesh data, skeleton, and animations.
    /// Replaces MeshImporter and AnimationClipImporter for standard model files.
    /// </summary>
    [AssetImporter(".fbx", ".dae", ".obj")]
    public class ModelImporter : IImporter
    {
        bool ConvertUnits = true;
        public bool Optimize = false;
        public bool LeftHanded = true;
        public bool FlipUVs = false;
        public bool FlipWinding = true;
        public bool CalculateTangents = true;
        public bool PreTransform = false;

        public bool ImportMesh = true;
        public bool ImportSkeleton = true;
        public bool ImportAnimations = true;


        [ValueRange(0, 180)]
        public float SmoothingAngle = 66f;
        public float Scale = 1;

        public List<MeshPartConfig> Parts = new();

        private List<Bone> _skeleton = new();
        private List<string> _boneNames = new();

        // Assimp tracking dictionaries for skeleton generation
        private Dictionary<string, Node> _nodes = new();
        private Dictionary<string, bool> _validNodes = new();
        private Dictionary<string, Assimp.Bone> _bones = new();

        /// <summary>
        /// Full import: produce all artifacts (mesh + animations) from a source file.
        /// </summary>
        public ImportResult Import(string filepath)
        {
            var result = new ImportResult { Compound = true };
            var scene = LoadScene(filepath, out float scale);
            var name = System.IO.Path.GetFileNameWithoutExtension(filepath);

            // Generate skeleton first (needed by both mesh and animation extraction)
            if (ImportSkeleton || ImportMesh)
                GenerateSkeleton(scene);

            // ── Mesh ──
            if (ImportMesh && scene.HasMeshes)
            {
                var meshData = ExtractMeshData(scene, scale);
                result.Artifacts.Add(new ImportArtifact
                {
                    Name = name,
                    Type = nameof(MeshData),
                    Data = meshData
                });
            }

            // ── Skeleton ──
            if (ImportSkeleton && _skeleton.Count > 0)
            {
                var skeleton = new Skeleton
                {
                    Name = name,
                    Bones = _skeleton.ToArray(),
                    BoneNames = _boneNames.ToArray()
                };
                result.Artifacts.Add(new ImportArtifact
                {
                    Name = name,
                    Type = nameof(Skeleton),
                    Data = skeleton
                });
            }

            // ── Animations ──
            if (ImportAnimations)
            {
                var animImporter = new AnimationClipImporter();
                try
                {
                    var clip = animImporter.Load(filepath);
                    if (clip.Channels.Count > 0)
                    {
                        var animName = clip.Name;
                        if (string.IsNullOrEmpty(animName) || _boneNames.Contains(animName))
                            animName = name;
                        clip.Name = animName;

                        result.Artifacts.Add(new ImportArtifact
                        {
                            Name = animName,
                            Type = nameof(AnimationClip),
                            Data = clip
                        });
                    }
                }
                catch (Exception ex)
                {
                    Debug.Log($"[ModelImporter] No animation data in '{name}': {ex.Message}");
                }
            }

            Debug.Log($"[ModelImporter] '{name}': {result.Artifacts.Count} artifacts " +
                      $"({(ImportMesh && scene.HasMeshes ? 1 : 0)} mesh, {scene.AnimationCount} animations, " +
                      $"{_skeleton.Count} bones)");

            return result;
        }

        /// <summary>
        /// Parse mesh data only (CPU arrays, no GPU). Thread-safe.
        /// Used by StaticMeshImporter and other callers that only need the mesh.
        /// </summary>
        public MeshData ParseMesh(string filepath)
        {
            var scene = LoadScene(filepath, out float scale);
            GenerateSkeleton(scene);
            return ExtractMeshData(scene, scale);
        }

        /// <summary>
        /// Parse animations only from a model file.
        /// </summary>
        public List<AnimationClip> ParseAnimations(string filepath)
        {
            var scene = LoadScene(filepath, out float scale);
            GenerateSkeleton(scene);

            var clips = new List<AnimationClip>();
            if (scene.HasAnimations)
            {
                foreach (var anim in scene.Animations)
                    clips.Add(ExtractAnimation(anim, scale));
            }
            return clips;
        }

        // ══════════════════════════════════════════════════════════════
        // Scene Loading
        // ══════════════════════════════════════════════════════════════

        private Scene LoadScene(string filepath, out float scale)
        {
            _skeleton.Clear();
            _boneNames.Clear();
            _nodes.Clear();
            _validNodes.Clear();
            _bones.Clear();

            // Unit conversion: DAE defaults to 0.01, FBX to 1.0
            // If Scale is explicitly set (non-1), use it directly. Otherwise use convention.
            if (Scale != 1f)
                scale = Scale;
            else if (ConvertUnits && filepath.EndsWith(".dae", StringComparison.OrdinalIgnoreCase))
                scale = 0.01f;
            else
                scale = Scale;

            var importer = new AssimpContext();

            importer.SetConfig(new Assimp.Configs.NormalSmoothingAngleConfig(SmoothingAngle));
            importer.SetConfig(new Assimp.Configs.GlobalScaleConfig(scale));
            importer.SetConfig(new Assimp.Configs.KeepSceneHierarchyConfig(true));

            // Build post-process steps from config fields
            var steps = PostProcessSteps.Triangulate
                      | PostProcessSteps.GenerateNormals
                      | PostProcessSteps.GenerateUVCoords
                      | PostProcessSteps.SortByPrimitiveType
                      | PostProcessSteps.ImproveCacheLocality
                      | PostProcessSteps.JoinIdenticalVertices
                      | PostProcessSteps.ValidateDataStructure
                      | PostProcessSteps.GlobalScale;

            if (LeftHanded)         steps |= PostProcessSteps.MakeLeftHanded;
            if (FlipWinding)        steps |= PostProcessSteps.FlipWindingOrder;
            if (FlipUVs)            steps |= PostProcessSteps.FlipUVs;
            if (CalculateTangents)  steps |= PostProcessSteps.CalculateTangentSpace;
            if (Optimize)           steps |= PostProcessSteps.OptimizeMeshes | PostProcessSteps.OptimizeGraph;
            if (PreTransform)       steps |= PostProcessSteps.PreTransformVertices;

            var scene = importer.ImportFile(filepath, steps);

            if (scene == null)
                throw new Exception($"Failed to load model from {filepath}");

            return scene;
        }

        // ══════════════════════════════════════════════════════════════
        // Mesh Extraction (ported from MeshImporter.ParseRaw)
        // ══════════════════════════════════════════════════════════════

        private MeshData ExtractMeshData(Scene scene, float scale)
        {
            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();
            var weightMap = new List<Dictionary<int, float>>();
            int vertexOffset = 0;

            var meshes = scene.Meshes;

            int startIndex = 0;
            var parts = new List<MeshPart>();

            // Map Assimp mesh indices to their parent node names (has LOD suffix)
            var meshNodeNames = new Dictionary<int, string>();
            BuildMeshNodeMap(scene.RootNode, meshNodeNames);

            foreach (var mesh in meshes)
            {
                int meshIndexCount = 0;
                int baseVertex = positions.Count;

                for (int i = 0; i < mesh.VertexCount; i++)
                {
                    var pos = mesh.Vertices[i];
                    var norm = mesh.Normals[i];
                    var uv = mesh.HasTextureCoords(0) ? mesh.TextureCoordinateChannels[0][i] : new Vector3D(0, 0, 0);

                    positions.Add(new Vector3(pos.X, pos.Y, pos.Z));
                    normals.Add(new Vector3(norm.X, norm.Y, norm.Z));
                    uvs.Add(new Vector2(uv.X, uv.Y));

                    weightMap.Add(new Dictionary<int, float>());
                }

                foreach (var face in mesh.Faces)
                {
                    if (face.IndexCount == 3)
                    {
                        indices.Add((uint)(face.Indices[0] + vertexOffset));
                        indices.Add((uint)(face.Indices[1] + vertexOffset));
                        indices.Add((uint)(face.Indices[2] + vertexOffset));
                        meshIndexCount += 3;
                    }
                }

                // Extract bone weights
                if (_skeleton.Count > 0 && mesh.HasBones)
                {
                    foreach (Assimp.Bone bone in mesh.Bones)
                    {
                        if (bone.HasVertexWeights)
                        {
                            int boneIndex = _boneNames.IndexOf(bone.Name);
                            if (boneIndex < 0) continue;

                            foreach (Assimp.VertexWeight vw in bone.VertexWeights)
                            {
                                int vertexId = baseVertex + (int)vw.VertexID;
                                if (vertexId < weightMap.Count &&
                                    weightMap[vertexId].Count < 4 &&
                                    !weightMap[vertexId].ContainsKey(boneIndex))
                                {
                                    weightMap[vertexId].Add(boneIndex, vw.Weight);
                                }
                            }
                        }
                    }
                }

                // Per-part bounding box
                Vector3 partMin = new Vector3(float.MaxValue);
                Vector3 partMax = new Vector3(float.MinValue);
                for (int vi = baseVertex; vi < baseVertex + mesh.VertexCount; vi++)
                {
                    partMin = Vector3.Min(partMin, positions[vi]);
                    partMax = Vector3.Max(partMax, positions[vi]);
                }

                var partBB = new Vortice.Mathematics.BoundingBox(partMin, partMax);
                var partCenter = (partMin + partMax) * 0.5f;
                var partRadius = (partMax - partCenter).Length();

                // Use node name (what Blender shows) → fallback to mesh name → fallback to index
                int origIdx = scene.Meshes.IndexOf(mesh);
                string nodeName = meshNodeNames.GetValueOrDefault(origIdx);
                string partName = nodeName ?? mesh.Name ?? $"Part_{meshes.IndexOf(mesh)}";

                // Disambiguate sub-meshes under the same node by appending material name
                if (nodeName != null && mesh.MaterialIndex >= 0 && mesh.MaterialIndex < scene.MaterialCount)
                {
                    var matName = scene.Materials[mesh.MaterialIndex].Name;
                    if (!string.IsNullOrEmpty(matName))
                        partName = $"{nodeName} [{matName}]";
                }

                parts.Add(new MeshPart
                {
                    Name = partName,
                    Enabled = true,
                    BaseIndex = startIndex,
                    NumIndices = meshIndexCount,
                    BaseVertex = 0,
                    BoundingBox = partBB,
                    BoundingSphere = new Vector4(partCenter, partRadius),
                });

                vertexOffset += mesh.VertexCount;
                startIndex += meshIndexCount;
            }

            // Global bounding box
            Vector3 min = new Vector3(float.MaxValue);
            Vector3 max = new Vector3(float.MinValue);
            foreach (var pos in positions)
            {
                min = Vector3.Min(min, pos);
                max = Vector3.Max(max, pos);
            }

            var data = new MeshData
            {
                Positions = positions.ToArray(),
                Normals = normals.ToArray(),
                UVs = uvs.ToArray(),
                Indices = indices.ToArray(),
                Parts = parts,
                BoundingBox = new Vortice.Mathematics.BoundingBox(min, max),
            };

            // ── Sync Part Config ──
            var configByName = new Dictionary<string, MeshPartConfig>();
            foreach (var cfg in Parts)
                if (!string.IsNullOrEmpty(cfg.Name))
                    configByName[cfg.Name] = cfg;

            Parts.Clear();
            for (int i = 0; i < parts.Count; i++)
            {
                var part = parts[i];
                if (configByName.TryGetValue(part.Name, out var existing))
                {
                    Parts.Add(existing);
                }
                else
                {
                    bool isCollision = IsCollisionPart(part.Name);
                    int autoLOD = isCollision ? -1 : ParseLODLevel(part.Name);
                    Parts.Add(new MeshPartConfig
                    {
                        Name = part.Name,
                        Render = !isCollision,
                        Collision = isCollision,
                        LODMin = autoLOD,
                        LODMax = autoLOD,
                    });
                }
                parts[i].Enabled = Parts[i].Render;
            }

            data.Parts = parts;

            // ── LOD Discovery (config-driven) ──
            data.LODs = DiscoverLODsFromConfig(parts, Parts);
            if (data.LODs.Count > 1)
                Debug.Log("ModelImporter", $"Discovered {data.LODs.Count} LOD levels");

            // Ensure every part has a unique MaterialSlot.
            // DiscoverLODsFromConfig assigns slots for LOD-participating parts.
            // Assign sequential slots to any remaining parts (e.g., tiles in a mixed FBX).
            int maxAssigned = -1;
            foreach (var p in parts)
                maxAssigned = Math.Max(maxAssigned, p.MaterialSlot);

            int nextUnusedSlot = maxAssigned + 1;
            // Parts with MaterialSlot = 0 that weren't the first LOD-assigned part need unique slots
            var lodAssigned = new HashSet<int>();
            if (data.LODs.Count > 0)
            {
                foreach (var lod in data.LODs)
                    if (lod.MeshPartIndices != null)
                        foreach (var idx in lod.MeshPartIndices)
                            lodAssigned.Add(idx);
            }
            for (int i = 0; i < parts.Count; i++)
            {
                if (!lodAssigned.Contains(i))
                    parts[i].MaterialSlot = nextUnusedSlot++;
            }

            // Skeleton data
            if (_skeleton.Count > 0)
            {
                data.Bones = _skeleton.ToArray();
                data.BoneWeights = BuildBoneWeights(weightMap);
                Debug.Log("ModelImporter", $"Mesh: {_skeleton.Count} bones, {data.BoneWeights.Length} vertex weights");
            }

            return data;
        }

        // ══════════════════════════════════════════════════════════════
        // Part Config Helpers
        // ══════════════════════════════════════════════════════════════

        private static void BuildMeshNodeMap(Node node, Dictionary<int, string> map)
        {
            if (node.HasMeshes)
                foreach (var meshIdx in node.MeshIndices)
                    if (!map.ContainsKey(meshIdx))
                        map[meshIdx] = node.Name;

            if (node.HasChildren)
                foreach (var child in node.Children)
                    BuildMeshNodeMap(child, map);
        }

        private static bool IsCollisionPart(string name)
        {
            if (string.IsNullOrEmpty(name)) return false;
            if (name.Contains("UCX", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("UBX", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("UCP", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("USP", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("ConvexHull", StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        private static int ParseLODLevel(string name)
        {
            if (string.IsNullOrEmpty(name)) return -1;
            // Strip material suffix: "SM_Tower_LOD0 [Stone]" → "SM_Tower_LOD0"
            int bracket = name.IndexOf(" [");
            var baseName = bracket >= 0 ? name.Substring(0, bracket) : name;

            // Search for "_LOD" or "LOD_" anywhere in the name (case-insensitive)
            var upper = baseName.ToUpperInvariant();

            // Pattern: "_LOD" followed by digit(s) — e.g. Wall_LOD0, SM_House_LOD1_Part
            int idx = upper.IndexOf("_LOD");
            if (idx >= 0)
            {
                int start = idx + 4;
                int end = start;
                while (end < upper.Length && char.IsDigit(upper[end])) end++;
                if (end > start && int.TryParse(upper.AsSpan(start, end - start), out int level))
                    return level;
            }

            // Pattern: "LOD_" followed by digit(s) — e.g. LOD_0_Wall, LOD_1
            idx = upper.IndexOf("LOD_");
            if (idx >= 0)
            {
                int start = idx + 4;
                int end = start;
                while (end < upper.Length && char.IsDigit(upper[end])) end++;
                if (end > start && int.TryParse(upper.AsSpan(start, end - start), out int level))
                    return level;
            }

            return -1;
        }

        private static List<MeshLOD> DiscoverLODsFromConfig(List<MeshPart> parts, List<MeshPartConfig> configs)
        {
            var lodGroups = new SortedDictionary<int, List<int>>();
            for (int i = 0; i < parts.Count && i < configs.Count; i++)
            {
                int lodMin = configs[i].LODMin;
                int lodMax = configs[i].LODMax;
                if (lodMin < 0) continue;
                if (lodMax < lodMin) lodMax = lodMin;
                for (int lod = lodMin; lod <= lodMax; lod++)
                {
                    if (!lodGroups.ContainsKey(lod))
                        lodGroups[lod] = new List<int>();
                    lodGroups[lod].Add(i);
                }
            }
            if (lodGroups.Count <= 1) return new List<MeshLOD>();

            // Assign MaterialSlot to each MeshPart by base name grouping.
            // Parts with same base name (LOD suffix stripped) get the same slot.
            var slotByBaseName = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            int nextSlot = 0;

            foreach (var kvp in lodGroups)
            {
                foreach (var partIdx in kvp.Value)
                {
                    var baseName = StripLODSuffix(parts[partIdx].Name);
                    if (!slotByBaseName.TryGetValue(baseName, out var slot))
                    {
                        slot = nextSlot++;
                        slotByBaseName[baseName] = slot;
                    }
                    parts[partIdx].MaterialSlot = slot;
                }
            }

            var result = new List<MeshLOD>();
            foreach (var kvp in lodGroups)
                result.Add(new MeshLOD { MeshPartIndices = kvp.Value.ToArray() });
            return result;
        }

        /// <summary>
        /// Strip LOD suffix/prefix from a mesh part name to get the base surface name.
        /// "Castle_Wall_LOD0" → "Castle_Wall", "LOD_1_Wall" → "Wall"
        /// </summary>
        public static string StripLODSuffix(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;

            // Strip "_LODN" suffix
            var upper = name.ToUpperInvariant();
            int idx = upper.LastIndexOf("_LOD");
            if (idx >= 0)
            {
                int end = idx + 4;
                while (end < upper.Length && char.IsDigit(upper[end])) end++;
                // Only strip if it's at the end or followed by non-alpha (like " [material]")
                if (end >= upper.Length || !char.IsLetter(upper[end]))
                    return name.Substring(0, idx);
            }

            // Strip "LOD_N_" prefix
            if (upper.StartsWith("LOD_") && upper.Length >= 5)
            {
                int end = 4;
                while (end < upper.Length && char.IsDigit(upper[end])) end++;
                if (end > 4 && end < upper.Length && upper[end] == '_')
                    return name.Substring(end + 1);
            }

            return name;
        }

        private static BoneWeight[] BuildBoneWeights(List<Dictionary<int, float>> weightMap)
        {
            var boneWeights = new BoneWeight[weightMap.Count];
            for (int i = 0; i < weightMap.Count; i++)
            {
                BoneWeight weight = BoneWeight.Default;
                int j = 0;
                foreach (var pair in weightMap[i])
                {
                    if (j >= 4) break;
                    switch (j)
                    {
                        case 0: weight.BoneIDs.X = pair.Key; weight.Weights.X = pair.Value; break;
                        case 1: weight.BoneIDs.Y = pair.Key; weight.Weights.Y = pair.Value; break;
                        case 2: weight.BoneIDs.Z = pair.Key; weight.Weights.Z = pair.Value; break;
                        case 3: weight.BoneIDs.W = pair.Key; weight.Weights.W = pair.Value; break;
                    }
                    j++;
                }
                boneWeights[i] = weight;
            }
            return boneWeights;
        }

        // ══════════════════════════════════════════════════════════════
        // Animation Extraction (via Assimp)
        // ══════════════════════════════════════════════════════════════

        private AnimationClip ExtractAnimation(Assimp.Animation anim, float scale)
        {
            var clip = new AnimationClip();
            clip.Name = anim.Name;
            clip.Duration = (float)anim.DurationInTicks;
            clip.TicksPerSecond = anim.TicksPerSecond > 0 ? (float)anim.TicksPerSecond : 30f;

            foreach (var nodeChannel in anim.NodeAnimationChannels)
            {
                var channel = new AnimationChannel { Target = nodeChannel.NodeName };

                // Position keys
                if (nodeChannel.HasPositionKeys)
                {
                    var posKeys = new List<VectorKey>(nodeChannel.PositionKeyCount);
                    foreach (var key in nodeChannel.PositionKeys)
                    {
                        posKeys.Add(new VectorKey
                        {
                            Time = (float)key.Time,
                            Value = new Vector3(key.Value.X, key.Value.Y, key.Value.Z) * scale
                        });
                    }
                    channel.Position = new VectorKeys(posKeys);
                }

                // Rotation keys
                if (nodeChannel.HasRotationKeys)
                {
                    var rotKeys = new List<QuaternionKey>(nodeChannel.RotationKeyCount);
                    foreach (var key in nodeChannel.RotationKeys)
                    {
                        rotKeys.Add(new QuaternionKey
                        {
                            Time = (float)key.Time,
                            Value = new System.Numerics.Quaternion(key.Value.X, key.Value.Y, key.Value.Z, key.Value.W)
                        });
                    }
                    channel.Rotation = new QuaternionKeys(rotKeys);
                }

                // Scale keys
                if (nodeChannel.HasScalingKeys)
                {
                    var scaleKeys = new List<VectorKey>(nodeChannel.ScalingKeyCount);
                    foreach (var key in nodeChannel.ScalingKeys)
                    {
                        scaleKeys.Add(new VectorKey
                        {
                            Time = (float)key.Time,
                            Value = new Vector3(key.Value.X, key.Value.Y, key.Value.Z)
                        });
                    }
                    channel.Scale = new VectorKeys(scaleKeys);
                }

                clip.AddChannel(channel);
            }

            Debug.Log("ModelImporter", $"Animation '{clip.Name}': {clip.DurationSeconds:F2}s, {clip.Channels.Count} channels");
            return clip;
        }

        // ══════════════════════════════════════════════════════════════
        // Skeleton Generation (ported from MeshImporter)
        // ══════════════════════════════════════════════════════════════

        private void GenerateSkeleton(Scene scene)
        {
            _skeleton.Clear();
            _boneNames.Clear();
            _nodes.Clear();
            _validNodes.Clear();
            _bones.Clear();

            FindBones(scene.RootNode);
            ValidateBones(scene);
            FlattenHierarchy(scene.RootNode);

            if (_skeleton.Count > 0)
            {
                Debug.Log("ModelImporter", $"Skeleton: {_skeleton.Count} bones");
                for (int i = 0; i < Math.Min(5, _skeleton.Count); i++)
                {
                    var bone = _skeleton[i];
                    string parentName = bone.Parent >= 0 ? _skeleton[bone.Parent].Name : "ROOT";
                    Debug.Log($"  [{i}] {bone.Name} → {parentName}");
                }
            }
        }

        private void FindBones(Node node)
        {
            if (!string.IsNullOrEmpty(node.Name))
            {
                _nodes[node.Name] = node;
                _validNodes[node.Name] = false;
            }
            if (!node.HasChildren) return;
            foreach (Node child in node.Children)
                FindBones(child);
        }

        private void ValidateBones(Scene scene)
        {
            int meshIndex = 0;
            foreach (Assimp.Mesh mesh in scene.Meshes)
            {
                if (!mesh.HasBones) continue;
                foreach (Assimp.Bone bone in mesh.Bones)
                {
                    if (!_nodes.ContainsKey(bone.Name)) continue;
                    if (_bones.ContainsKey(bone.Name)) continue;

                    _validNodes[bone.Name] = true;
                    _bones.Add(bone.Name, bone);

                    Node node = _nodes[bone.Name];
                    while (node.Parent != null)
                    {
                        node = node.Parent;
                        if (node.HasMeshes && node.MeshIndices.Contains(meshIndex)) break;
                        if (_validNodes.ContainsKey(node.Name))
                            _validNodes[node.Name] = true;
                    }
                }
                meshIndex++;
            }
        }

        private void FlattenHierarchy(Node node)
        {
            if (!_validNodes.ContainsKey(node.Name)) return;
            if (_validNodes[node.Name] == false) return;

            _boneNames.Add(node.Name);

            var newBone = new Bone { Name = node.Name };
            Matrix4x4 bind = ToMatrix(node.Transform);
            Matrix4x4.Decompose(bind, out var s, out var r, out var t);

            newBone.BindPoseMatrix = bind;
            newBone.BindPose = new BonePose { Position = t, Rotation = r, Scale = s };



            newBone.Parent = node.Parent != null ? _boneNames.IndexOf(node.Parent.Name) : -1;
            _skeleton.Add(newBone);

            if (_bones.TryGetValue(node.Name, out Assimp.Bone? value))
                newBone.OffsetMatrix = ToMatrix(value.OffsetMatrix);
            else
                newBone.OffsetMatrix = Matrix4x4.Identity;

            if (node.HasChildren)
            {
                foreach (Node child in node.Children)
                    FlattenHierarchy(child);
            }
        }

        private static Matrix4x4 ToMatrix(Assimp.Matrix4x4 m)
        {
            // Match Apex: map A1..D4 directly, then Transpose
            Matrix4x4 result = new Matrix4x4(
                m.A1, m.A2, m.A3, m.A4,
                m.B1, m.B2, m.B3, m.B4,
                m.C1, m.C2, m.C3, m.C4,
                m.D1, m.D2, m.D3, m.D4
            );
            return Matrix4x4.Transpose(result);
        }
    }
}
