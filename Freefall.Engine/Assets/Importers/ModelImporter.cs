using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.ComponentModel;
using System.Text;
using Assimp;
using Freefall.Animation;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Serialization;
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
    [AssetImporter(".fbx", ".dae", ".obj", ".x", ImportPriority = 2)]
    public class ModelImporter : IImporter, IPostImporter
    {
        /// <summary>
        /// When set, newly created ModelImporter instances will use these settings
        /// instead of class defaults. Set by UnityImporterWindow before bulk import.
        /// </summary>
        public static ModelImporter OverrideSettings;

        public bool ConvertUnits = true;
        public bool Optimize = false;
        public bool LeftHanded = true;
        public bool FlipUVs = false;
        public bool FlipWinding = true;
        public bool CalculateTangents = true;
        public bool PreTransform = false;

        public bool ImportMesh = true;
        public bool ImportSkeleton = true;
        public bool ImportAnimations = true;
        public bool CreateMaterials = true;


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
        private string _currentFilePath;

        // Transient per-group data for PostImport (not serialized)
        [NonSerialized] private Dictionary<string, (Vector3 pos, System.Numerics.Quaternion rot, Vector3 scale)> _groupTransforms = new();
        [NonSerialized] private Dictionary<string, string[]> _groupMaterialNames = new();

        private bool isDAE;

        /// <summary>
        /// Full import: produce all artifacts (mesh + animations) from a source file.
        /// </summary>
        public ImportResult Import(string filepath)
        {
            // Apply override settings if set (from UnityImporterWindow)
            if (OverrideSettings != null)
            {
                Scale = OverrideSettings.Scale;
                ConvertUnits = OverrideSettings.ConvertUnits;
                Optimize = OverrideSettings.Optimize;
                PreTransform = OverrideSettings.PreTransform;
                FlipUVs = OverrideSettings.FlipUVs;
                FlipWinding = OverrideSettings.FlipWinding;
                CalculateTangents = OverrideSettings.CalculateTangents;
                LeftHanded = OverrideSettings.LeftHanded;
            }

            var result = new ImportResult { Compound = true };
            _currentFilePath = filepath;
            var scene = LoadScene(filepath, out float scale);
            var name = System.IO.Path.GetFileNameWithoutExtension(filepath);

            // Generate skeleton first (needed by both mesh and animation extraction)
            if (ImportSkeleton || ImportMesh)
                GenerateSkeleton(scene);

            // ── Mesh ──
            if (ImportMesh && scene.HasMeshes)
            {
                var groups = ExtractMeshDataPerNode(scene, scale);
                foreach (var (groupName, meshData) in groups)
                {
                    result.Artifacts.Add(new ImportArtifact
                    {
                        Name = groupName,
                        Type = nameof(MeshData),
                        Data = meshData
                    });
                }
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

            // ── Materials ──
            if (CreateMaterials && ImportMesh && scene.HasMeshes)
            {
                ExtractMaterials(scene, filepath, result);
            }

            Debug.Log($"[ModelImporter] '{name}': {result.Artifacts.Count} artifacts " +
                      $"({(ImportMesh && scene.HasMeshes ? 1 : 0)} mesh, {scene.AnimationCount} animations, " +
                      $"{_skeleton.Count} bones)");

            return result;
        }

        public object GetInspectionTarget(MetaFile meta) => this;

        // ══════════════════════════════════════════════════════════════
        // Post-Import: Prefab Generation
        // ══════════════════════════════════════════════════════════════

        /// <summary>
        /// Called after Import() artifacts have been packed and assigned GUIDs.
        /// Generates a Prefab sub-asset per MeshData artifact, wiring Mesh GUID
        /// and material references into MeshRenderer via EntitySerializer.
        /// </summary>
        public List<ImportArtifact> PostImport(string filepath, IReadOnlyList<SubAssetEntry> subAssets)
        {
            var prefabs = new List<ImportArtifact>();

            // Build lookup: material name → GUID from sub-assets + AssetDatabase
            var materialLookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var sub in subAssets)
            {
                if (sub.Type == nameof(AssetDefinitionData))
                    materialLookup[sub.Name] = sub.Guid;
            }

            foreach (var sub in subAssets)
            {
                if (sub.Type != nameof(MeshData)) continue;

                var meshGuid = sub.Guid;
                var meshName = sub.Name;

                // Create temp entity (not registered with EntityManager)
                var entity = new Entity(meshName, register: false);

                // Apply node transform if captured during extraction
                if (_groupTransforms.TryGetValue(meshName, out var xform))
                {
                    entity.Transform.Position = xform.pos;
                    entity.Transform.Rotation = xform.rot;
                    entity.Transform.Scale = xform.scale;
                  //  Debug.Log("ModelImporter", $"GroupTransform '{meshName}': rot=({xform.rot.X:F4}, {xform.rot.Y:F4}, {xform.rot.Z:F4}, {xform.rot.W:F4}) scale=({xform.scale.X:F4}, {xform.scale.Y:F4}, {xform.scale.Z:F4})");
                }

                // Build MeshRenderer with mesh stub + materials
                var renderer = new MeshRenderer();
                renderer.Mesh = new Graphics.Mesh { Guid = meshGuid };

                // Resolve materials per slot
                if (_groupMaterialNames.TryGetValue(meshName, out var matNames) && matNames != null)
                {
                    for (int i = 0; i < matNames.Length; i++)
                    {
                        var matName = matNames[i];
                        if (string.IsNullOrEmpty(matName)) continue;

                        string matGuid = null;
                        // Check sub-assets first, then AssetDatabase
                        if (!materialLookup.TryGetValue(matName, out matGuid))
                        {
                            // Resolve by name, but verify it's actually a material
                            var candidateGuid = AssetDatabase.ResolveGuidByName(matName);
                            if (candidateGuid != null && IsMaterialGuid(candidateGuid))
                                matGuid = candidateGuid;
                        }

                        matGuid ??= InternalAssets.Guids.DefaultMaterial;

                        renderer.Materials.Add(new MaterialOverride
                        {
                            MaterialSlot = i,
                            Material = new Graphics.Material { Guid = matGuid }
                        });
                    }
                }


                entity.AddComponent(renderer);

                // Serialize via EntitySerializer
                var serializer = new EntitySerializer();
                var yaml = serializer.SaveToString(new List<Entity> { entity });

                prefabs.Add(new ImportArtifact
                {
                    Name = meshName,
                    Type = nameof(PrefabData),
                    Data = new PrefabData { Yaml = Encoding.UTF8.GetBytes(yaml) }
                });
            }

            if (prefabs.Count > 0)
                Debug.Log("ModelImporter", $"Created {prefabs.Count} prefab sub-asset(s)");

            return prefabs;
        }

        /// <summary>
        /// Parse mesh data only (CPU arrays, no GPU). Thread-safe.
        /// Returns only the first node group for backwards compatibility.
        /// Use ParseMeshes() to get all groups.
        /// </summary>
        public MeshData ParseMesh(string filepath)
        {
            var groups = ParseMeshes(filepath);
            return groups.Count > 0 ? groups[0].data : new MeshData();
        }

        /// <summary>
        /// Parse all mesh data groups from a model file. One group per node base name.
        /// </summary>
        public List<(string name, MeshData data)> ParseMeshes(string filepath)
        {
            _currentFilePath = filepath;
            var scene = LoadScene(filepath, out float scale);
            GenerateSkeleton(scene);
            return ExtractMeshDataPerNode(scene, scale);
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

            // ASCII FBX 6.1.0 is not supported by Assimp — auto-convert to binary
            string tempBinaryFbx = null;
            if (IsAsciiFbx(filepath))
            {
                tempBinaryFbx = ConvertAsciiFbx(filepath);
                filepath = tempBinaryFbx;
            }

            try
            {
                isDAE = filepath.ToLowerInvariant().EndsWith(".dae");
                // Unit conversion: DAE defaults to 0.01, FBX to 1.0
                // If Scale is explicitly set (non-1), use it directly. Otherwise use convention.
                scale = isDAE ? Scale * 0.01f :  Scale;

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

                // Temp debug: dump node hierarchy transforms
                //var fname = System.IO.Path.GetFileNameWithoutExtension(filepath);
                //DumpNodeHierarchy(scene.RootNode, fname, 0);

                return scene;
            }
            finally
            {
                // Clean up temp binary FBX
                if (tempBinaryFbx != null)
                    try { File.Delete(tempBinaryFbx); } catch { }
            }
        }

        /// <summary>
        /// Detect ASCII FBX format by checking the file header.
        /// ASCII FBX starts with "; FBX" or ";FBX", binary starts with "Kaydara FBX Binary".
        /// </summary>
        private static bool IsAsciiFbx(string filepath)
        {
            if (!filepath.EndsWith(".fbx", StringComparison.OrdinalIgnoreCase))
                return false;

            try
            {
                using var fs = File.OpenRead(filepath);
                var header = new byte[20];
                if (fs.Read(header, 0, header.Length) < 5) return false;

                // ASCII: starts with ";FBX" or "; FBX"
                // Binary: starts with "Kaydara FBX Binary"
                return header[0] == (byte)';';
            }
            catch { return false; }
        }

        /// <summary>
        /// Convert an ASCII FBX file to binary FBX using FbxConverter.exe.
        /// Resolved next to the running executable, same as texconv.exe.
        /// Returns the path to the temporary binary FBX file.
        /// </summary>
        private static string ConvertAsciiFbx(string sourcePath)
        {
            var exeDir = AppDomain.CurrentDomain.BaseDirectory;
            var converterPath = System.IO.Path.Combine(exeDir, "FbxConverter.exe");

            if (!File.Exists(converterPath))
                throw new FileNotFoundException(
                    $"FbxConverter.exe not found at '{converterPath}'. " +
                    "Copy the Autodesk FBX Converter to the editor output directory.");

            // Write to a temp file so we don't modify the original
            var tempPath = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                $"freefall_fbx_{Guid.NewGuid():N}.fbx");

            var args = $"\"{sourcePath}\" \"{tempPath}\" /f /v";

            var psi = new ProcessStartInfo(converterPath)
            {
                Arguments = args,
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };

            using var process = Process.Start(psi);
            var stdout = process.StandardOutput.ReadToEnd();
            var stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0 || !File.Exists(tempPath))
            {
                // Clean up on failure
                try { if (File.Exists(tempPath)) File.Delete(tempPath); } catch { }
                throw new Exception(
                    $"FbxConverter failed (exit {process.ExitCode}):\n{stderr}\n{stdout}");
            }

            var srcName = System.IO.Path.GetFileName(sourcePath);
            Debug.Log("ModelImporter", $"Converted ASCII FBX → binary: {srcName}");

            return tempPath;
        }


        // ══════════════════════════════════════════════════════════════
        // Mesh Extraction (ported from MeshImporter.ParseRaw)
        // ══════════════════════════════════════════════════════════════

        private List<(string name, MeshData data)> ExtractMeshDataPerNode(Scene scene, float scale)
        {
            // ── Phase 1: Build flat part list (identical to old ExtractMeshData) ──
            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();
            var weightMap = new List<Dictionary<int, float>>();
            int vertexOffset = 0;
            var meshes = scene.Meshes;
            int startIndex = 0;
            var parts = new List<MeshPart>();
            var partMaterialNames = new List<string>();

            var meshNodeNames = new Dictionary<int, string>();
            BuildMeshNodeMap(scene.RootNode, meshNodeNames);

            // Build world transform map for each node (for prefab generation)
            var nodeWorldTransforms = new Dictionary<string, Matrix4x4>();
            BuildNodeWorldTransforms(scene.RootNode, Matrix4x4.Identity, nodeWorldTransforms);

            var nodeMeshCounts = new Dictionary<string, int>();
            foreach (var kvp in meshNodeNames)
            {
                if (!nodeMeshCounts.ContainsKey(kvp.Value))
                    nodeMeshCounts[kvp.Value] = 0;
                nodeMeshCounts[kvp.Value]++;
            }

            foreach (var mesh in meshes)
            {
                int meshIndexCount = 0;
                int baseVertex = positions.Count;

                for (int i = 0; i < mesh.VertexCount; i++)
                {
                    var pos = mesh.Vertices[i];
                    var norm = mesh.Normals[i];
                    var uv = mesh.HasTextureCoords(0) ? mesh.TextureCoordinateChannels[0][i] : new Vector3D(0, 0, 0);

                    if (isDAE)
                    {
                        positions.Add(new Vector3(pos.X, pos.Y, pos.Z));
                        normals.Add(new Vector3(norm.X, norm.Y, norm.Z));
                    }
                    else
                    {
                        positions.Add(new Vector3(-pos.X, pos.Y, -pos.Z));
                        normals.Add(new Vector3(-norm.X, norm.Y, -norm.Z));
                    }


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
                                    weightMap[vertexId].Add(boneIndex, vw.Weight);
                            }
                        }
                    }
                }

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

                int origIdx = scene.Meshes.IndexOf(mesh);
                string nodeName = meshNodeNames.GetValueOrDefault(origIdx);
                string partName = nodeName ?? mesh.Name ?? $"Part_{meshes.IndexOf(mesh)}";

                string matName = null;
                if (mesh.MaterialIndex >= 0 && mesh.MaterialIndex < scene.MaterialCount)
                {
                    matName = scene.Materials[mesh.MaterialIndex].Name;
                    if (nodeName != null && nodeMeshCounts.GetValueOrDefault(nodeName) > 1)
                    {
                        if (!string.IsNullOrEmpty(matName))
                            partName = $"{nodeName} [{matName}]";
                    }
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
                partMaterialNames.Add(matName);
                vertexOffset += mesh.VertexCount;
                startIndex += meshIndexCount;
            }

            // ── Phase 2: Sync Part Configs ──
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

            for (int i = 0; i < parts.Count && i < Parts.Count; i++)
            {
                if (!Parts[i].Collision && IsCollisionPart(parts[i].Name))
                {
                    Parts[i].Collision = true;
                    Parts[i].Render = false;
                    parts[i].Enabled = false;
                }
            }

            AutoAssignLOD0ToBaseParts(parts, Parts);

            // ── Phase 3: Group parts by node base name ──
            var fileName = _currentFilePath != null
                ? System.IO.Path.GetFileNameWithoutExtension(_currentFilePath)
                : "mesh";

            var groupOrder = new List<string>();
            var groupPartIndices = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < parts.Count; i++)
            {
                string rawName = parts[i].Name;
                int bracket = rawName.IndexOf(" [");
                string cleanName = bracket >= 0 ? rawName.Substring(0, bracket) : rawName;
                string groupKey = StripLODSuffix(cleanName);

                if (string.IsNullOrEmpty(groupKey))
                    groupKey = fileName;


                if (!groupPartIndices.TryGetValue(groupKey, out var list))
                {
                    list = new List<int>();
                    groupPartIndices[groupKey] = list;
                    groupOrder.Add(groupKey);
                }
                list.Add(i);
            }

            // Single group: use file name, skip split (identical to old single-mesh path)
            if (groupOrder.Count <= 1)
            {
                groupOrder.Clear();
                groupOrder.Add(fileName);
                groupPartIndices.Clear();
                groupPartIndices[fileName] = Enumerable.Range(0, parts.Count).ToList();
            }

            // ── Phase 4: Build MeshData per group ──
            var result = new List<(string name, MeshData data)>();
            var posArr = positions.ToArray();
            var normArr = normals.ToArray();
            var uvArr = uvs.ToArray();
            var idxArr = indices.ToArray();

            foreach (var groupName in groupOrder)
            {
                var pIndices = groupPartIndices[groupName];
                var gParts = new List<MeshPart>();
                var gConfigs = new List<MeshPartConfig>();
                var gPos = new List<Vector3>();
                var gNorm = new List<Vector3>();
                var gUV = new List<Vector2>();
                var gIdx = new List<uint>();
                var gWeights = new List<Dictionary<int, float>>();
                var gMatNames = new List<string>();

                foreach (var pi in pIndices)
                {
                    var srcPart = parts[pi];

                    // Find vertex range used by this part's indices
                    int minV = int.MaxValue, maxV = int.MinValue;
                    for (int ii = srcPart.BaseIndex; ii < srcPart.BaseIndex + srcPart.NumIndices; ii++)
                    {
                        int v = (int)idxArr[ii];
                        minV = Math.Min(minV, v);
                        maxV = Math.Max(maxV, v);
                    }
                    if (minV > maxV) continue;

                    int localBase = gPos.Count;

                    // Copy vertex data for this part's range
                    for (int v = minV; v <= maxV; v++)
                    {
                        gPos.Add(posArr[v]);
                        gNorm.Add(normArr[v]);
                        gUV.Add(uvArr[v]);
                        gWeights.Add(v < weightMap.Count ? weightMap[v] : new Dictionary<int, float>());
                    }

                    // Copy and rebase indices
                    int localBaseIdx = gIdx.Count;
                    for (int ii = srcPart.BaseIndex; ii < srcPart.BaseIndex + srcPart.NumIndices; ii++)
                        gIdx.Add((uint)((int)idxArr[ii] - minV + localBase));

                    // Recompute bounding info
                    Vector3 pMin = new(float.MaxValue), pMax = new(float.MinValue);
                    for (int v = localBase; v < localBase + (maxV - minV + 1); v++)
                    {
                        pMin = Vector3.Min(pMin, gPos[v]);
                        pMax = Vector3.Max(pMax, gPos[v]);
                    }
                    var center = (pMin + pMax) * 0.5f;

                    gParts.Add(new MeshPart
                    {
                        Name = srcPart.Name,
                        Enabled = srcPart.Enabled,
                        BaseIndex = localBaseIdx,
                        NumIndices = srcPart.NumIndices,
                        BaseVertex = 0,
                        BoundingBox = new Vortice.Mathematics.BoundingBox(pMin, pMax),
                        BoundingSphere = new Vector4(center, (pMax - center).Length()),
                    });
                    gConfigs.Add(Parts[pi]);
                    gMatNames.Add(partMaterialNames[pi]);
                }

                if (gParts.Count == 0) continue;

                // Per-group LOD discovery
                AutoAssignLOD0ToBaseParts(gParts, gConfigs);
                var lods = DiscoverLODsFromConfig(gParts, gConfigs);

                // Material slot assignment: LOD parts keep their slots,
                // non-LOD parts get sequential slots after the last LOD slot.
                var lodAssigned = new HashSet<int>();
                foreach (var lod in lods)
                    if (lod.MeshPartIndices != null)
                        foreach (var idx in lod.MeshPartIndices)
                            lodAssigned.Add(idx);

                // Only consider LOD-assigned slots (not struct default 0)
                int maxSlot = -1;
                foreach (var idx in lodAssigned)
                    maxSlot = Math.Max(maxSlot, gParts[idx].MaterialSlot);
                int nextSlot = maxSlot + 1;

                for (int i = 0; i < gParts.Count; i++)
                    if (!lodAssigned.Contains(i))
                        gParts[i].MaterialSlot = nextSlot++;

                // Build MaterialNames keyed by MaterialSlot
                int slotCount = 0;
                foreach (var p in gParts) slotCount = Math.Max(slotCount, p.MaterialSlot + 1);
                var matNames = new string[slotCount];
                for (int i = 0; i < gParts.Count; i++)
                {
                    int slot = gParts[i].MaterialSlot;
                    if (slot >= 0 && slot < matNames.Length && matNames[slot] == null)
                        matNames[slot] = gMatNames[i];
                }


                // Global bounding box (must be after vertex correction)
                Vector3 gMin = new(float.MaxValue), gMax = new(float.MinValue);
                foreach (var pos in gPos) { gMin = Vector3.Min(gMin, pos); gMax = Vector3.Max(gMax, pos); }

                var data = new MeshData
                {
                    Positions = gPos.ToArray(),
                    Normals = gNorm.ToArray(),
                    UVs = gUV.ToArray(),
                    Indices = gIdx.ToArray(),
                    Parts = gParts,
                    BoundingBox = new Vortice.Mathematics.BoundingBox(gMin, gMax),
                    LODs = lods,
                    MaterialNames = matNames,
                };

                if (_skeleton.Count > 0)
                {
                    data.Bones = _skeleton.ToArray();
                    data.BoneWeights = BuildBoneWeights(gWeights);
                }

               // if (lods.Count > 1)
               //     Debug.Log("ModelImporter", $"Group '{groupName}': {lods.Count} LOD levels");

                result.Add((groupName, data));

                // Store per-group data for PostImport prefab generation
                _groupMaterialNames[groupName] = matNames;

                // Find the first mesh index in this group's parts to get its node name → world transform
                if (pIndices.Count > 0)
                {
                    int firstMeshIdx = -1;
                    for (int mi = 0; mi < meshes.Count; mi++)
                    {
                        if (meshNodeNames.TryGetValue(mi, out var nn) &&
                            StripLODSuffix(nn).Equals(groupName, StringComparison.OrdinalIgnoreCase))
                        {
                            firstMeshIdx = mi;
                            break;
                        }
                    }

                    if (firstMeshIdx >= 0 && meshNodeNames.TryGetValue(firstMeshIdx, out var nodeName2) &&
                        nodeWorldTransforms.TryGetValue(nodeName2, out var worldMatrix))
                    {
                        Matrix4x4.Decompose(worldMatrix, out var s, out var r, out var t);
                        // Rotation is identity — vertex data now matches Unity's convention.
                        _groupTransforms[groupName] = (Vector3.Zero, System.Numerics.Quaternion.Identity, s);
                    }
                }
            }

            //Debug.Log("ModelImporter", $"{result.Count} mesh group(s): " +
            //    string.Join(", ", result.Select(r => $"{r.name}({r.data.Parts.Count}p)")));

            return result;
        }

        // ══════════════════════════════════════════════════════════════
        // Material Extraction
        // ══════════════════════════════════════════════════════════════

        // Assimp TextureType → Freefall material slot name
        private static readonly Dictionary<TextureType, string> AssimpSlotMap = new()
        {
            { TextureType.Diffuse,           "Albedo" },
            { TextureType.Normals,           "Normal" },
            { TextureType.Roughness,         "Roughness" },
            { TextureType.Metalness,         "Metallic" },
            { TextureType.Emissive,          "Emissive" },
            { TextureType.AmbientOcclusion,  "AO" },
            { TextureType.Height,            "HeightTex" },
        };

        /// <summary>
        /// Check whether a GUID points to a material-compatible asset
        /// (standalone .mat or AssetDefinitionData sub-asset).
        /// </summary>
        private static bool IsMaterialGuid(string guid)
        {
            var meta = AssetDatabase.GetMeta(guid);
            if (meta == null) return false;

            // Check if it's a sub-asset of type AssetDefinitionData
            var sub = meta.SubAssets?.Find(s => s.Guid == guid);
            if (sub != null)
                return sub.Type == nameof(AssetDefinitionData);

            // Standalone .mat asset
            return meta.MainAssetType == nameof(AssetDefinitionData);
        }

        /// <summary>
        /// Parse animations only from a model file.
        /// </summary>
        public List<AnimationClip> ExtractAnimations(Scene scene, string filepath, ImportResult result, float scale)
        {
            var clips = new List<AnimationClip>();
            if (scene.HasAnimations)
            {
                foreach (var anim in scene.Animations)
                    clips.Add(ExtractAnimation(anim, scale));
            }
            return clips;
        }

        /// <summary>
        /// Extract Assimp materials and emit MaterialDefinition artifacts.
        /// Skips materials that already exist in AssetDatabase.
        /// </summary>
        private void ExtractMaterials(Scene scene, string filepath, ImportResult result)
        {
            if (!scene.HasMaterials) return;

            var fbxDir = Path.GetDirectoryName(filepath) ?? "";
            int created = 0;

            foreach (var assimpMat in scene.Materials)
            {
                var matName = assimpMat.Name;
                if (string.IsNullOrWhiteSpace(matName) || matName == "DefaultMaterial")
                    continue;

                // Skip if an asset with this name already exists in AssetDatabase
                var existingGuid = AssetDatabase.ResolveGuidByName(matName);
                if (existingGuid != null)
                {
                    Debug.Log("ModelImporter", $"Material '{matName}': existing asset found, skipping");
                    continue;
                }

                // Build YAML MaterialDefinition
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("!Material");
                sb.AppendLine($"Effect: {InternalAssets.Guids.DefaultEffect}");
                sb.AppendLine("Textures:");

                bool hasTextures = false;
                foreach (var (texType, slotName) in AssimpSlotMap)
                {
                    if (assimpMat.GetMaterialTexture(texType, 0, out var texSlot))
                    {
                        var texPath = texSlot.FilePath;
                        if (string.IsNullOrEmpty(texPath)) continue;

                        // Resolve texture GUID
                        string texGuid = ResolveTextureGuid(texPath, fbxDir);
                        if (texGuid != null)
                        {
                            sb.AppendLine($"  {slotName}: {texGuid}");
                            hasTextures = true;
                        }
                        else
                        {
                            Debug.Log("ModelImporter", $"Material '{matName}': texture not found for {slotName}: '{texPath}'");
                        }
                    }
                }

                if (!hasTextures)
                {
                    Debug.Log("ModelImporter", $"Material '{matName}': no textures resolved, skipping");
                    continue;
                }

                var yaml = sb.ToString();
                result.Artifacts.Add(new ImportArtifact
                {
                    Name = matName,
                    Type = "Material",
                    Data = new AssetDefinitionData
                    {
                        TypeName = "Material",
                        YamlBytes = System.Text.Encoding.UTF8.GetBytes(yaml)
                    }
                });
                created++;
            }

            if (created > 0)
                Debug.Log("ModelImporter", $"Created {created} material sub-asset(s)");
        }

        /// <summary>
        /// Resolve an Assimp texture file path to an AssetDatabase GUID.
        /// Searches AssetDatabase by filename (textures are unique per asset pack).
        /// </summary>
        private static string ResolveTextureGuid(string texPath, string fbxDir)
        {
            // Handle embedded textures (Assimp uses "*N" notation)
            if (texPath.StartsWith("*")) return null;

            // Search by filename (without extension) — AssetDatabase tracks all imported textures
            var fileName = Path.GetFileNameWithoutExtension(texPath);
            return AssetDatabase.ResolveGuidByName(fileName);
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

        private static void BuildNodeWorldTransforms(Node node, Matrix4x4 parentWorld, Dictionary<string, Matrix4x4> result)
        {
            // Assimp Matrix4x4 is row-major; System.Numerics is column-major → transpose
            var m = node.Transform;
            var local = new Matrix4x4(
                m.A1, m.B1, m.C1, m.D1,
                m.A2, m.B2, m.C2, m.D2,
                m.A3, m.B3, m.C3, m.D3,
                m.A4, m.B4, m.C4, m.D4
            );
            var world = local * parentWorld;
            result[node.Name] = world;

            if (node.HasChildren)
                foreach (var child in node.Children)
                    BuildNodeWorldTransforms(child, world, result);
        }

        private static readonly object _dumpLock = new();
        private static void DumpNodeHierarchy(Node node, string file, int depth)
        {
            var lines = new List<string>();
            CollectNodeHierarchy(node, file, depth, lines);
            lock (_dumpLock)
            {
                var dumpPath = @"d:\Projects\2026\Freefall\.tmp\fbx_nodes.txt";
                Directory.CreateDirectory(Path.GetDirectoryName(dumpPath)!);
                File.AppendAllLines(dumpPath, lines);
            }
        }

        private static void CollectNodeHierarchy(Node node, string file, int depth, List<string> lines)
        {
            var indent = new string(' ', depth * 2);
            var t = node.Transform;
            bool isIdentity = t.A1 == 1 && t.B2 == 1 && t.C3 == 1 && t.D4 == 1 &&
                              t.A2 == 0 && t.A3 == 0 && t.A4 == 0 &&
                              t.B1 == 0 && t.B3 == 0 && t.B4 == 0 &&
                              t.C1 == 0 && t.C2 == 0 && t.C4 == 0 &&
                              t.D1 == 0 && t.D2 == 0 && t.D3 == 0;
            var meshStr = node.HasMeshes ? $" [meshes: {string.Join(",", node.MeshIndices)}]" : "";
            if (!isIdentity)
            {
                lines.Add($"[FBX] {file}: {indent}{node.Name}{meshStr} T=[{t.A4:F4},{t.B4:F4},{t.C4:F4}] S=[{t.A1:F4},{t.B2:F4},{t.C3:F4}] R=[{t.A2:F4},{t.A3:F4},{t.B1:F4},{t.B3:F4},{t.C1:F4},{t.C2:F4}]");
            }
            else
            {
                lines.Add($"[FBX] {file}: {indent}{node.Name}{meshStr} (identity)");
            }
            if (node.HasChildren)
                foreach (var child in node.Children)
                    CollectNodeHierarchy(child, file, depth + 1, lines);
        }

        /// <summary>
        /// Build a map of Assimp mesh index → accumulated world transform.
        /// This bakes the FBX node hierarchy transforms into each mesh's coordinate space.
        /// </summary>
        private static void BuildMeshTransformMap(Node node, Matrix4x4 parentTransform, Dictionary<int, Matrix4x4> map)
        {
            // Convert Assimp's Matrix4x4 to System.Numerics.Matrix4x4
            var at = node.Transform;
            var localTransform = new Matrix4x4(
                at.A1, at.B1, at.C1, at.D1,
                at.A2, at.B2, at.C2, at.D2,
                at.A3, at.B3, at.C3, at.D3,
                at.A4, at.B4, at.C4, at.D4);

            var worldTransform = localTransform * parentTransform;

            if (node.HasMeshes)
                foreach (var meshIdx in node.MeshIndices)
                    if (!map.ContainsKey(meshIdx))
                        map[meshIdx] = worldTransform;

            if (node.HasChildren)
                foreach (var child in node.Children)
                    BuildMeshTransformMap(child, worldTransform, map);
        }

        private static bool IsCollisionPart(string name)
        {
            if (string.IsNullOrEmpty(name)) return false;
            // Unreal conventions
            if (name.Contains("UCX", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("UBX", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("UCP", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("USP", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.Contains("ConvexHull", StringComparison.OrdinalIgnoreCase)) return true;
            // Common suffixes
            if (name.EndsWith("_Coll", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.EndsWith("_Collision", StringComparison.OrdinalIgnoreCase)) return true;
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

            // If LOD groups start at 1+ (no explicit LOD0), synthesize LOD0
            // from non-LOD base parts whose stripped name matches a LOD part.
            if (lodGroups.Count > 0 && !lodGroups.ContainsKey(0))
            {
                // Collect base names present in any LOD group
                var lodBaseNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var kvp in lodGroups)
                    foreach (var idx in kvp.Value)
                        lodBaseNames.Add(StripLODSuffix(parts[idx].Name));

                var lod0 = new List<int>();
                for (int i = 0; i < parts.Count && i < configs.Count; i++)
                {
                    if (configs[i].LODMin >= 0) continue;  // already in a LOD group
                    if (configs[i].Collision) continue;
                    if (!configs[i].Render) continue;

                    var baseName = StripLODSuffix(parts[i].Name);
                    // baseName == parts[i].Name means no LOD suffix (it's a base part)
                    if (baseName == parts[i].Name && lodBaseNames.Contains(baseName))
                        lod0.Add(i);
                }

                if (lod0.Count > 0)
                {
                    lodGroups[0] = lod0;
                    //Debug.Log("ModelImporter", $"Synthesized LOD0 from {lod0.Count} base parts");
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
        /// Auto-assign LOD0 to base parts (no LOD suffix) that have LOD-suffixed siblings.
        /// E.g., "Chapel_Wall_Out" becomes LOD0 when "Chapel_Wall_Out_LOD_01" exists.
        /// </summary>
        private static void AutoAssignLOD0ToBaseParts(List<MeshPart> parts, List<MeshPartConfig> configs)
        {
            // Collect base names that have at least one LOD-suffixed part
            var lodBaseNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < parts.Count && i < configs.Count; i++)
            {
                if (configs[i].LODMin >= 0)
                {
                    var baseName = StripLODSuffix(parts[i].Name);
                    if (baseName != parts[i].Name) // had a LOD suffix that was stripped
                        lodBaseNames.Add(baseName);
                }
            }

            if (lodBaseNames.Count == 0) return;

            // Any non-LOD, non-collision part whose name matches a LOD base name → LOD0
            for (int i = 0; i < parts.Count && i < configs.Count; i++)
            {
                if (configs[i].LODMin >= 0) continue;
                if (configs[i].Collision) continue;

                var baseName = StripLODSuffix(parts[i].Name);
                if (baseName == parts[i].Name && lodBaseNames.Contains(baseName))
                {
                    configs[i].LODMin = 0;
                    configs[i].LODMax = 0;
                }
            }
        }

        /// <summary>
        /// Strip LOD suffix/prefix from a mesh part name to get the base surface name.
        /// "Castle_Wall_LOD0" → "Castle_Wall", "LOD_1_Wall" → "Wall"
        /// </summary>
        public static string StripLODSuffix(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;

            // NOTE: Do NOT strip trailing ".NNN" (Blender disambiguation suffixes like ".001", ".002").
            // These differentiate distinct meshes (Plane.001 vs Plane.002 = different geometry).
            // LOD grouping still works: SM_Roof_LOD1.001 → strip _LOD1 → SM_Roof.001 matches SM_Roof.001.

            // Strip "_LODN" or "_LOD_N" portion but keep any trailing content (e.g. " [MaterialName]")
            var upper = name.ToUpperInvariant();
            int idx = upper.LastIndexOf("_LOD");
            if (idx >= 0)
            {
                int end = idx + 4;
                // Skip optional underscore separator: _LOD_01 vs _LOD01
                if (end < upper.Length && upper[end] == '_') end++;
                while (end < upper.Length && char.IsDigit(upper[end])) end++;
                // Only strip if it's at the end or followed by non-alpha (like " [material]")
                if (end >= upper.Length || !char.IsLetter(upper[end]))
                    return name.Substring(0, idx) + name.Substring(end);
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

            //Debug.Log("ModelImporter", $"Animation '{clip.Name}': {clip.DurationSeconds:F2}s, {clip.Channels.Count} channels");
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
                //for (int i = 0; i < Math.Min(5, _skeleton.Count); i++)
                //{
                //    var bone = _skeleton[i];
                //    string parentName = bone.Parent >= 0 ? _skeleton[bone.Parent].Name : "ROOT";
                //    Debug.Log($"  [{i}] {bone.Name} → {parentName}");
                //}
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
