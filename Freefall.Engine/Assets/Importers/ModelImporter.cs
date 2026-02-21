using System;
using System.Collections.Generic;
using System.Numerics;
using Assimp;
using Freefall.Animation;
using Freefall.Graphics;
using Matrix4x4 = System.Numerics.Matrix4x4;
using Bone = Freefall.Animation.Bone;
using VectorKey = Freefall.Animation.VectorKey;
using QuaternionKey = Freefall.Animation.QuaternionKey;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Unified model importer for FBX, DAE, OBJ files.
    /// Produces all artifacts from a single source: mesh data, skeleton, and animations.
    /// Replaces MeshImporter and AnimationClipImporter for standard model files.
    /// </summary>
    [AssetImporter(".fbx", ".dae", ".obj")]
    public class ModelImporter : IImporter
    {
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
            var result = new ImportResult();
            var scene = LoadScene(filepath, out float scale);
            var name = System.IO.Path.GetFileNameWithoutExtension(filepath);

            // Generate skeleton first (needed by both mesh and animation extraction)
            GenerateSkeleton(scene);

            // ── Mesh ──
            if (scene.HasMeshes)
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
            if (_skeleton.Count > 0)
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
            if (scene.HasAnimations)
            {
                foreach (var anim in scene.Animations)
                {
                    var clip = ExtractAnimation(anim, scale);

                    // Assimp often returns the root bone name (e.g. "Hips") as the animation name
                    // for .dae files. Detect this and fall back to the source file name.
                    var animName = anim.Name;
                    if (string.IsNullOrEmpty(animName) || _boneNames.Contains(animName))
                        animName = name;

                    result.Artifacts.Add(new ImportArtifact
                    {
                        Name = animName,
                        Type = nameof(AnimationClip),
                        Data = clip
                    });
                }
            }

            Debug.Log($"[ModelImporter] '{name}': {result.Artifacts.Count} artifacts " +
                      $"({(scene.HasMeshes ? 1 : 0)} mesh, {scene.AnimationCount} animations, " +
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

            // Match Apex: DAE uses 0.01 scale, FBX uses 1.0
            scale = filepath.EndsWith(".dae", StringComparison.OrdinalIgnoreCase) ? 0.01f : 1f;

            var importer = new AssimpContext();

            importer.SetConfig(new Assimp.Configs.NormalSmoothingAngleConfig(66f));
            importer.SetConfig(new Assimp.Configs.GlobalScaleConfig(scale));
            importer.SetConfig(new Assimp.Configs.KeepSceneHierarchyConfig(true));

            var steps = PostProcessPreset.TargetRealTimeMaximumQuality;
            steps |= PostProcessSteps.FlipWindingOrder;
            steps |= PostProcessSteps.MakeLeftHanded;
            steps |= PostProcessSteps.CalculateTangentSpace;
            steps |= PostProcessSteps.GlobalScale;

            // Apex explicitly removes these
            steps &= ~PostProcessSteps.FindInstances;
            steps &= ~PostProcessSteps.FindDegenerates;
            steps &= ~PostProcessSteps.SplitLargeMeshes;
            steps &= ~PostProcessSteps.RemoveRedundantMaterials;
            steps &= ~PostProcessSteps.OptimizeMeshes;

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

            // Match Apex: reverse mesh order (MeshImporter also reverses)
            var meshes = new List<Assimp.Mesh>(scene.Meshes);
            meshes.Reverse();

            int startIndex = 0;
            var parts = new List<MeshPart>();

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

                parts.Add(new MeshPart
                {
                    Name = mesh.Name ?? $"Part_{meshes.IndexOf(mesh)}",
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

            // Skeleton data
            if (_skeleton.Count > 0)
            {
                data.Bones = _skeleton.ToArray();
                data.BoneWeights = BuildBoneWeights(weightMap);
                Debug.Log("ModelImporter", $"Mesh: {_skeleton.Count} bones, {data.BoneWeights.Length} vertex weights");
            }

            return data;
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
