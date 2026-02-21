using System;
using System.Collections.Generic;
using System.Numerics;
using Assimp;
using Freefall.Graphics;
using Freefall.Animation;
using Matrix4x4 = System.Numerics.Matrix4x4;
using Bone = Freefall.Animation.Bone;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports mesh files (.fbx, .dae, .obj) using Assimp.
    /// Configuration matches Apex's import settings.
    /// Supports skeletal meshes with bone weights.
    /// </summary>
    public class MeshImporter
    {
        private List<Bone> skeleton = new List<Bone>();
        private List<string> boneNames = new List<string>();

        public Graphics.Mesh Load(string filepath)
        {
            var data = ParseRaw(filepath);
            var meshObj = new Graphics.Mesh(Engine.Device, data);
            meshObj.Name = System.IO.Path.GetFileNameWithoutExtension(filepath);
            return meshObj;
        }

        /// <summary>
        /// Parse mesh file into raw CPU arrays. No GPU resources created.
        /// Thread-safe — can be called from background threads.
        /// </summary>
        public MeshData ParseRaw(string filepath)
        {
            skeleton.Clear();
            boneNames.Clear();

            // MATCH APEX EXACTLY: DAE files use 0.01 scale, FBX files use 1.0 scale
            float scale = filepath.EndsWith(".dae", StringComparison.OrdinalIgnoreCase) ? 0.01f : 1f;

            // MATCH APEX EXACTLY - no 'using' statement
            var importer = new AssimpContext();

            // MATCH APEX CONFIGURATION EXACTLY
            Assimp.Configs.NormalSmoothingAngleConfig config = new Assimp.Configs.NormalSmoothingAngleConfig(66f);
            Assimp.Configs.GlobalScaleConfig globalScale = new Assimp.Configs.GlobalScaleConfig(scale);
            Assimp.Configs.KeepSceneHierarchyConfig keepSceneHierarchy = new Assimp.Configs.KeepSceneHierarchyConfig(true);
           
            importer.SetConfig(config);
            importer.SetConfig(keepSceneHierarchy);
            importer.SetConfig(globalScale);

            // MATCH APEX FLAGGING LOGIC
            var steps = PostProcessPreset.TargetRealTimeMaximumQuality;
            
            bool FlipWinding = true;
            bool FlipUVs = false;
            bool LeftHanded = true;
            bool CalculateTangents = true;
            bool PreTransform = false;

            if (FlipWinding) steps |= PostProcessSteps.FlipWindingOrder;
            if (FlipUVs) steps |= PostProcessSteps.FlipUVs;
            if (LeftHanded) steps |= PostProcessSteps.MakeLeftHanded;
            if (CalculateTangents) steps |= PostProcessSteps.CalculateTangentSpace;
            if (PreTransform) steps |= PostProcessSteps.PreTransformVertices;

            steps |= PostProcessSteps.GlobalScale;
            
            // Apex Explicitly Removes these
            steps &= ~PostProcessSteps.FindInstances;
            steps &= ~PostProcessSteps.FindDegenerates;
            steps &= ~PostProcessSteps.SplitLargeMeshes;
            steps &= ~PostProcessSteps.RemoveRedundantMaterials;
            steps &= ~PostProcessSteps.OptimizeMeshes;

            var scene = importer.ImportFile(filepath, steps);
            
            if (scene == null || !scene.HasMeshes)
                throw new Exception($"Failed to load mesh from {filepath}");

            // Generate skeleton if mesh has bones
            GenerateSkeleton(scene);

            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();
            var weightMap = new List<Dictionary<int, float>>();
            int vertexOffset = 0;

            // MATCH APEX REVERSE ORDER
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

                        // Assimp GlobalScale already applied the scale to vertices
                        positions.Add(new Vector3(pos.X, pos.Y, pos.Z));
                        normals.Add(new Vector3(norm.X, norm.Y, norm.Z));
                        uvs.Add(new Vector2(uv.X, uv.Y));
                        
                        // Initialize weight map for each vertex
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
                    if (skeleton.Count > 0 && mesh.HasBones)
                    {
                        foreach (Assimp.Bone bone in mesh.Bones)
                        {
                            if (bone.HasVertexWeights)
                            {
                                int boneIndex = boneNames.IndexOf(bone.Name);
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

                    // Compute per-part bounding box from this part's vertices directly
                    // (vertices are contiguous at [baseVertex, baseVertex+VertexCount) in the combined buffer)
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

                    var part = new MeshPart
                    {
                        Name = mesh.Name ?? $"Part_{meshes.IndexOf(mesh)}",
                        Enabled = true,
                        BaseIndex = startIndex,
                        NumIndices = meshIndexCount,
                        BaseVertex = 0,
                        BoundingBox = partBB,
                        BoundingSphere = new Vector4(partCenter, partRadius),
                    };
                    parts.Add(part);

                    vertexOffset += mesh.VertexCount;
                    startIndex += meshIndexCount;
                }

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

                // Add skeleton if bones were found
                if (skeleton.Count > 0)
                {
                    data.Bones = skeleton.ToArray();

                    // Convert weight map to BoneWeight array
                    BoneWeight[] boneWeights = new BoneWeight[weightMap.Count];
                    for (int i = 0; i < weightMap.Count; i++)
                    {
                        BoneWeight weight = BoneWeight.Default; // Default: bone 0, weight 1
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

                    data.BoneWeights = boneWeights;
                    Debug.Log("MeshImporter", $"Loaded {skeleton.Count} bones, {boneWeights.Length} vertex weights");
                }

            return data;
        }

        // MATCH APEX EXACTLY: tracking dictionaries for skeleton generation
        private Dictionary<string, Node> nodes = new Dictionary<string, Node>();
        private Dictionary<string, bool> validNodes = new Dictionary<string, bool>();
        private Dictionary<string, Assimp.Bone> bones = new Dictionary<string, Assimp.Bone>();

        // MATCH APEX: Phase 1 - Build node dictionary
        private void FindBones(Node node)
        {
            if (!string.IsNullOrEmpty(node.Name))
            {
                nodes[node.Name] = node;
                validNodes[node.Name] = false;
            }

            if (!node.HasChildren) return;

            foreach (Node child in node.Children)
                FindBones(child);
        }

        // MATCH APEX: Phase 2 - Validate which nodes are actual bones
        private void ValidateBones(Scene scene)
        {
            int meshIndex = 0;

            foreach (Assimp.Mesh mesh in scene.Meshes)
            {
                if (!mesh.HasBones) continue;

                foreach (Assimp.Bone bone in mesh.Bones)
                {
                    if (!nodes.ContainsKey(bone.Name))
                        continue;

                    if (bones.ContainsKey(bone.Name)) continue;

                    validNodes[bone.Name] = true;

                    bones.Add(bone.Name, bone);

                    Node node = nodes[bone.Name];

                    while (node.Parent != null)
                    {
                        node = node.Parent;

                        if (node.HasMeshes && node.MeshIndices.Contains(meshIndex))
                            break;

                        if (validNodes.ContainsKey(node.Name))
                            validNodes[node.Name] = true;
                    }
                }

                meshIndex++;
            }
        }

        // MATCH APEX: Phase 3 - Flatten hierarchy into skeleton list
        private void FlattenHierarchy(Node node)
        {
            if (!validNodes.ContainsKey(node.Name)) return;
            if (validNodes[node.Name] == false) return;

            boneNames.Add(node.Name);

            var newBone = new Bone { Name = node.Name };

            Matrix4x4 bind = ToMatrix(node.Transform);
            Matrix4x4.Decompose(bind, out var scale, out var rotate, out var translate);

            newBone.BindPoseMatrix = bind;
            newBone.BindPose = new BonePose { Position = translate, Rotation = rotate, Scale = scale };
            
            // MATCH APEX: Direct parent lookup, not walk-up
            newBone.Parent = node.Parent != null ? boneNames.IndexOf(node.Parent.Name) : -1;
            skeleton.Add(newBone);

            if (bones.TryGetValue(node.Name, out Assimp.Bone? value))
            {
                newBone.OffsetMatrix = ToMatrix(value.OffsetMatrix);
            }
            else
            {
                newBone.OffsetMatrix = Matrix4x4.Identity;
            }

            if (node.HasChildren)
            {
                foreach (Node child in node.Children)
                    FlattenHierarchy(child);
            }
        }

        // MATCH APEX: 3-phase skeleton generation
        private void GenerateSkeleton(Scene scene)
        {
            nodes.Clear();
            validNodes.Clear();
            bones.Clear();

            FindBones(scene.RootNode);
            ValidateBones(scene);
            FlattenHierarchy(scene.RootNode);
            
            // Debug: Print first few bones with parent info
            Debug.Log("MeshImporter", $"Skeleton generated with {skeleton.Count} bones:");
            for (int i = 0; i < Math.Min(10, skeleton.Count); i++)
            {
                var bone = skeleton[i];
                string parentName = bone.Parent >= 0 ? skeleton[bone.Parent].Name : "ROOT";
                Debug.Log($"  [{i}] {bone.Name} -> Parent: {parentName} ({bone.Parent})");
            }
        }

        private static Matrix4x4 ToMatrix(Assimp.Matrix4x4 m)
        {
            // Match Apex's logic: Map A1..D4 directly to M11..M44, then Transpose.
            // This places Translation in Row 4 (Correct for System.Numerics)
            // But Transposes Rotation (Inverted).
            // This requires Animator.cs to Transpose AGAIN before upload.
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
