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
    /// Per-mesh settings for one mesh produced by the import (a model file can produce several).
    /// Keyed by the mesh sub-asset name. Applied by MeshLoader at load, so changing them needs no reimport.
    /// </summary>
    [Serializable]
    public class MeshConfig
    {
        [ReadOnly(true)]
        public string Name = string.Empty;
        [ValueRange(0.1f, 10.0f)]
        public float LODBias = 1.0f;

        public override string ToString() => Name;
    }

    /// <summary>
    /// Per-clip settings for one animation clip produced by the import. Keyed by the clip sub-asset name.
    /// Applied by AnimationClipLoader at load, so changing them needs no reimport, and a reimport keeps them.
    /// </summary>
    [Serializable]
    public class ClipConfig
    {
        [ReadOnly(true)]
        public string Name = string.Empty;

        /// <summary>What is marked on the clip (AnimationClip.Events).</summary>
        public List<Freefall.Animation.AnimationEvent> Events = new();

        /// <summary>
        /// The part of the take that is the clip, in frames of the file (its ticks): a take that holds several
        /// actions one after another, as the clip ranges of a Unity pack cut them. To 0 = the whole take. Unlike
        /// the events it is applied at import: a change needs a reimport.
        /// </summary>
        public float From;
        public float To;

        /// <summary>
        /// The clip plays in place (an animal pack's walk or run): give it the travel its feet imply, so that root
        /// motion carries the creature at the clip's own pace with the surge of every step, and the feet that are
        /// down stay where they are (FootTravel). Applied at import: a change needs a reimport.
        /// </summary>
        public bool RootFromFeet;

        /// <summary>
        /// With <see cref="RootFromFeet"/>, for a clip whose feet are hardly ever seen down (a gallop of a dozen
        /// frames): the even travel to give it instead, in m/s and in the rig's own space as the file has it (an
        /// animal pack's rig walks towards -Z: a gallop at 7 m/s is (0, 0, -7)). Zero: the travel from the feet.
        /// </summary>
        public Vector3 RootVelocity;

        /// <summary>
        /// With <see cref="RootFromFeet"/>: how far the pace may swing either way of its mean over a stride, as a share
        /// of it. The feet's own swing is smoothed to the stride's rhythm and scaled down to this: hand-made in-place
        /// walks stop dead where the weight passes from foot to foot. 1 or more keeps the bursts as the feet show them
        /// (a frog's hop, a rabbit's bound).
        /// </summary>
        public float RootSurge = FootTravel.DefaultSurge;

        /// <summary>
        /// Degrees the clip is turned about the upright through the model's origin: a take captured facing the other way
        /// from the rest of its pack (a rider pack's mount clips, made with the horse turned round, that would end in the
        /// saddle facing its tail). The keys of the nodes at the top of the file are turned; a node there that the clip
        /// does not key is not. Applied at import: a change needs a reimport.
        /// </summary>
        public float Turn;

        public override string ToString() => Name;
    }

    /// <summary>
    /// A prefab the import makes that was edited and saved (a collider added, children, members changed). It has
    /// no file of its own, so it is kept with the file's settings, by the prefab's name, and every import builds the
    /// prefab from it (<see cref="ModelImporter.SavePrefab"/>). Remove the entry and import again to get the prefab
    /// as the file alone makes it.
    /// </summary>
    [Serializable]
    public class PrefabConfig
    {
        [ReadOnly(true)]
        public string Name = string.Empty;

        /// <summary>The saved prefab, as EntitySerializer writes it.</summary>
        [Browsable(false)]
        public string Yaml = string.Empty;

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

        /// <summary>
        /// Turns the file to the engine's axes (Y up, Z ahead) from the ones an FBX says it was saved in, as Unity
        /// imports it: the nodes at the top of the file and their keys are turned, so a Z-up file's clips (3ds Max's)
        /// fit a Y-up rig instead of lying on their backs. Off: the file's numbers are taken as they are. Applied at
        /// import: a change needs a reimport.
        /// </summary>
        public bool ConvertAxes = false;

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
        public List<MeshConfig> Meshes = new();

        public MeshConfig? FindMeshConfig(string meshName)
        {
            foreach (var cfg in Meshes)
                if (string.Equals(cfg.Name, meshName, StringComparison.OrdinalIgnoreCase))
                    return cfg;
            return null;
        }

        public MeshConfig GetOrAddMeshConfig(string meshName)
        {
            var cfg = FindMeshConfig(meshName);
            if (cfg == null)
            {
                cfg = new MeshConfig { Name = meshName };
                Meshes.Add(cfg);
            }
            return cfg;
        }

        /// <summary>Per-clip settings, by clip sub-asset name. Only clips that have any are listed.</summary>
        public List<ClipConfig> Clips = new();

        public ClipConfig? FindClipConfig(string clipName)
        {
            foreach (var cfg in Clips)
                if (string.Equals(cfg.Name, clipName, StringComparison.OrdinalIgnoreCase))
                    return cfg;
            return null;
        }

        /// <summary>
        /// Write the events of a loaded clip into the settings of the file it comes from and save
        /// them (no reimport: they are put on the clip at load). A clip without events is taken
        /// off the list. False for a clip that does not come from a model file.
        /// </summary>
        public static bool SetClipEvents(Freefall.Animation.AnimationClip clip)
        {
            if (clip == null || string.IsNullOrEmpty(clip.Guid)) return false;
            if (AssetDatabase.GetImporter(clip.Guid) is not ModelImporter importer) return false;

            string name = AssetDatabase.ResolveFriendlyName(clip.Guid);
            var config = importer.FindClipConfig(name);

            var events = new List<Freefall.Animation.AnimationEvent>();
            if (clip.Events != null)
                foreach (var e in clip.Events)
                    if (e != null)
                        events.Add(new Freefall.Animation.AnimationEvent { Name = e.Name ?? string.Empty, Time = Math.Clamp(e.Time, 0f, 1f), Bone = e.Bone });
            events.Sort(static (a, b) => a.Time.CompareTo(b.Time));

            if (events.Count == 0)
            {
                if (config == null) return true;
                importer.Clips.Remove(config);
            }
            else
            {
                if (config == null) importer.Clips.Add(config = new ClipConfig { Name = name });
                config.Events = events;
            }

            return AssetDatabase.SaveImporter(clip.Guid, importer) != null;
        }

        /// <summary>Prefabs of this file that were edited and saved, by prefab name.</summary>
        public List<PrefabConfig> Prefabs = new();

        public PrefabConfig? FindPrefabConfig(string prefabName)
        {
            foreach (var cfg in Prefabs)
                if (string.Equals(cfg.Name, prefabName, StringComparison.OrdinalIgnoreCase))
                    return cfg;
            return null;
        }

        /// <summary>
        /// Keep an edited prefab of a model file (Apply to Prefab on an instance of a tree, then save): its YAML goes
        /// into the file's settings and the file is imported again, which builds the prefab from it. What the file
        /// makes of the prefab's root stays the file's (<see cref="FileMembers"/>). False for a prefab that is not a
        /// model file's.
        /// </summary>
        public static bool SavePrefab(Prefab prefab)
        {
            if (prefab == null || string.IsNullOrEmpty(prefab.Guid) || prefab.SourceYaml == null) return false;
            if (AssetDatabase.GuidToPath(prefab.Guid) != null) return false;   // a .prefab file of its own
            if (AssetDatabase.GetImporter(prefab.Guid) is not ModelImporter importer) return false;

            string name = AssetDatabase.ResolveFriendlyName(prefab.Guid);
            var config = importer.FindPrefabConfig(name);
            if (config == null) importer.Prefabs.Add(config = new PrefabConfig { Name = name });
            config.Yaml = prefab.ToYaml();

            AssetDatabase.SaveImporterAndReimport(prefab.Guid, importer);
            prefab.ClearDirty();
            return true;
        }

        /// <summary>
        /// Give meshes without LODs of their own (no _LOD1, _LOD2… objects in the file) a generated chain
        /// (LODGenerator). Authored LODs always win; skinned meshes are left as they are.
        /// </summary>
        [Category("Generated LODs")]
        public bool GenerateLODs = true;

        /// <summary>Share of LOD 0's triangles for each generated level, decreasing.</summary>
        public List<float> LODRatios = new() { 0.5f, 0.25f, 0.125f };

        /// <summary>The chain ends before a level that would have fewer triangles than this.</summary>
        public int LODMinTriangles = 64;

        /// <summary>
        /// Largest shape change of a level, in pixels of a 1080p screen at the size where the level takes over
        /// (meshoptimizer's estimate, well above what shows). A level that cannot reach its ratio within it keeps
        /// more triangles, or is not made. 0 = no limit.
        /// </summary>
        [ValueRange(0f, 30f)]
        public float LODPixelError = 8f;

        /// <summary>
        /// The clips of this file are made on another rig than the one whose clips a graph plays (a pack's rider, for
        /// the Human graph's rig): each is baked onto the skeleton named here, a Skeleton's GUID, as it is loaded
        /// (ClipRetarget). The graph then plays it like its own clips, blended with them on any layer. Empty = the
        /// clips stay as made. Applied at load, like the events: a change needs no reimport.
        /// </summary>
        [Category("Retargeting")]
        public string RetargetTo = string.Empty;

        /// <summary>
        /// With <see cref="RetargetTo"/>: the skeleton the clips were made on (a GUID), for a file of clips that has
        /// none of its own (it is in the rig's model file). Empty = the file's own skeleton.
        /// </summary>
        public string RetargetFrom = string.Empty;

        /// <summary>The members above, for inspectors that show them next to a mesh.</summary>
        public static readonly string[] LODGenerationMembers =
            { nameof(GenerateLODs), nameof(LODRatios), nameof(LODMinTriangles), nameof(LODPixelError) };

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
        private float _daeUnitScale = 1f;

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

            // DAE unit scale: Assimp applies unit conversion to node transforms
            // but NOT to mesh vertices. Extract the scale from the root node
            // so we can apply it to vertices (for correct bounding spheres).
            _daeUnitScale = 1f;
            if (isDAE)
            {
                var rootMat = ToMatrix(scene.RootNode.Transform);
                Matrix4x4.Decompose(rootMat, out var rs, out _, out _);
                _daeUnitScale = rs.X; // e.g. 0.01 for cm→m
            }

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
                    BoneNames = _boneNames.ToArray(),
                    FlipXZ = !isDAE
                };

                result.Artifacts.Add(new ImportArtifact
                {
                    Name = name,
                    Type = nameof(Skeleton),
                    Data = skeleton
                });
            }

            // ── Animations ──
            if (ImportAnimations && scene.HasAnimations)
            {
                foreach (var anim in scene.Animations)
                {
                    var clip = ExtractAnimation(scene, anim, scale);
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
                            // Resolve by name among materials first: an untyped lookup returns whatever asset
                            // carries the name, and a mesh or model file of the same name ("Castle Wall 1A1.fbx"
                            // next to "Castle Wall 1A1.mat") then hid the material and the slot fell back to
                            // DefaultMaterial.
                            var candidateGuid = AssetDatabase.ResolveGuidByName(matName, nameof(Graphics.Material))
                                                ?? AssetDatabase.ResolveGuidByName(matName);
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

                // Edited and saved: that prefab, with what comes from the file as this import made it
                if (FindPrefabConfig(meshName) is { Yaml.Length: > 0 } saved)
                    yaml = saved.Yaml = WithFileMembers(saved.Yaml, yaml);

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

        // ── Saved prefabs ──

        /// <summary>
        /// What of a saved prefab's root stays the file's, by document: its name, the place its node gives it, its mesh
        /// and the materials the file gives the mesh. A model changed in Blender reaches its saved prefabs that way;
        /// everything else (colliders, children, the renderer's other members) is the saved prefab's.
        /// </summary>
        private static readonly (string Document, string[] Members)[] FileMembers =
        {
            ("Entity", new[] { "Name" }),
            ("Transform", new[] { "Position", "Rotation", "Scale" }),
            ("MeshRenderer", new[] { "Mesh", "Materials" }),
        };

        /// <summary>
        /// <paramref name="saved"/> with the root's <see cref="FileMembers"/> as in <paramref name="made"/>, the prefab
        /// this import makes. Works on the text as EntitySerializer writes it (Import may run on a worker, where no
        /// entity may be made): a document per entity and component after a "---" line, the members two spaces in.
        /// A member the file's prefab lacks is taken out, one the saved prefab lacks is added; a document the saved
        /// prefab took off (its renderer) stays off. UIDs stay the saved prefab's, which its children refer to.
        /// </summary>
        internal static string WithFileMembers(string saved, string made)
        {
            var savedDocuments = SplitDocuments(saved);
            var madeDocuments = SplitDocuments(made);

            foreach (var (document, members) in FileMembers)
            {
                var into = RootDocument(savedDocuments, document);
                var from = RootDocument(madeDocuments, document);
                if (into == null || from == null) continue;

                foreach (var member in members)
                {
                    var (start, count) = FindMember(into, member);
                    if (start >= 0) into.RemoveRange(start, count);

                    var (fromStart, fromCount) = FindMember(from, member);
                    if (fromStart >= 0)
                        into.InsertRange(start >= 0 ? start : into.Count, from.GetRange(fromStart, fromCount));
                }
            }

            var text = new StringBuilder();
            foreach (var document in savedDocuments)
            {
                text.Append("---\n");
                foreach (var line in document) text.Append(line).Append('\n');
            }
            return text.ToString();
        }

        // The documents of a prefab's YAML, each as its lines without the "---" before it
        private static List<List<string>> SplitDocuments(string yaml)
        {
            var documents = new List<List<string>>();
            List<string>? document = null;
            foreach (var raw in yaml.TrimStart('﻿').Split('\n'))
            {
                var line = raw.TrimEnd('\r');
                if (line.Trim() == "---")
                {
                    documents.Add(document = new List<string>());
                    continue;
                }
                if (document == null)
                {
                    if (line.Trim().Length == 0) continue;
                    documents.Add(document = new List<string>());
                }
                document.Add(line);
            }

            foreach (var d in documents)
                while (d.Count > 0 && d[^1].Trim().Length == 0) d.RemoveAt(d.Count - 1);
            documents.RemoveAll(d => d.Count == 0);
            return documents;
        }

        // The document of that type among the root entity's: those from the first Entity document up to the next one
        private static List<string>? RootDocument(List<List<string>> documents, string type)
        {
            int entities = 0;
            foreach (var document in documents)
            {
                var header = document[0].Trim().TrimEnd(':');
                if (header == "Entity" && ++entities > 1) break;
                if (entities == 1 && header == type) return document;
            }
            return null;
        }

        // A member's lines in a document: its own and those below it up to the next member. (-1, 0) when it has none.
        private static (int Start, int Count) FindMember(List<string> document, string member)
        {
            for (int i = 1; i < document.Count; i++)
            {
                if (!IsMemberLine(document[i])) continue;
                int colon = document[i].IndexOf(':');
                if (colon < 0 || document[i][..colon].Trim() != member) continue;

                int end = i + 1;
                while (end < document.Count && !IsMemberLine(document[end])) end++;
                return (i, end - i);
            }
            return (-1, 0);
        }

        // A key of the document's mapping: two spaces in. A list's items ("  - ...") and anything further in belong
        // to the key above them.
        private static bool IsMemberLine(string line) =>
            line.Length > 2 && line[0] == ' ' && line[1] == ' ' && line[2] != ' ' && line[2] != '-';

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

                if (ConvertAxes) ToEngineAxes(scene);

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
        /// The file's axes (an FBX's CoordAxis, UpAxis and FrontAxis, each with a sign) turned to X right, Y up and Z
        /// ahead (<see cref="ConvertAxes"/>): the turn is put on the nodes right under the scene's root and on their
        /// keys, as Unity's FBX import does, so a clip's keys fit another file's rig that is already Y up. Meshes and
        /// bones below come along; a bone's offset matrix is unchanged, as mesh and bones turn together.
        /// </summary>
        private void ToEngineAxes(Scene scene)
        {
            Vector3? Axis(string name)
            {
                if (scene.Metadata == null
                    || !scene.Metadata.TryGetValue(name, out var axis) || !scene.Metadata.TryGetValue(name + "Sign", out var sign))
                    return null;
                int index = Convert.ToInt32(axis.Data);
                if (index < 0 || index > 2) return null;
                var v = Vector3.Zero;
                if (index == 0) v.X = 1; else if (index == 1) v.Y = 1; else v.Z = 1;
                return v * Convert.ToInt32(sign.Data);
            }

            var right = Axis("CoordAxis");
            var up = Axis("UpAxis");
            var ahead = Axis("FrontAxis");
            if (right == null || up == null || ahead == null) return;

            // v' = v * turn: each column is one of the file's axes, so v' = (v · right, v · up, v · ahead)
            var turn = new Matrix4x4(
                right.Value.X, up.Value.X, ahead.Value.X, 0,
                right.Value.Y, up.Value.Y, ahead.Value.Y, 0,
                right.Value.Z, up.Value.Z, ahead.Value.Z, 0,
                0, 0, 0, 1);
            if (turn.IsIdentity) return;
            if (turn.GetDeterminant() < 0)
            {
                Debug.LogWarning("ModelImporter", $"'{System.IO.Path.GetFileName(_currentFilePath)}': its axes mirror the engine's; ConvertAxes leaves them");
                return;
            }

            // The scene is in the engine's hand already (MakeLeftHanded mirrors Z): the turn as it is there
            if (LeftHanded)
            {
                var mirror = Matrix4x4.CreateScale(1, 1, -1);
                turn = mirror * turn * mirror;
            }
            var rotation = System.Numerics.Quaternion.CreateFromRotationMatrix(turn);

            var top = new HashSet<string>();
            foreach (var child in scene.RootNode.Children)
            {
                top.Add(child.Name);
                child.Transform = ToAssimp(ToMatrix(child.Transform) * turn);
            }

            foreach (var anim in scene.Animations)
                foreach (var channel in anim.NodeAnimationChannels)
                {
                    if (!top.Contains(channel.NodeName)) continue;
                    var positions = channel.PositionKeys;
                    for (int i = 0; i < positions.Count; i++)
                    {
                        var p = positions[i].Value;
                        var t = Vector3.Transform(new Vector3(p.X, p.Y, p.Z), rotation);
                        positions[i] = new Assimp.VectorKey { Time = positions[i].Time, Value = new Vector3D { X = t.X, Y = t.Y, Z = t.Z } };
                    }
                    var rotations = channel.RotationKeys;
                    for (int i = 0; i < rotations.Count; i++)
                    {
                        var q = rotations[i].Value;
                        // The key's own turn first, then the file's
                        var r = rotation * new System.Numerics.Quaternion(q.X, q.Y, q.Z, q.W);
                        rotations[i] = new Assimp.QuaternionKey { Time = rotations[i].Time, Value = new Assimp.Quaternion { W = r.W, X = r.X, Y = r.Y, Z = r.Z } };
                    }
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

            // Build mesh-to-world-transform map for skinned mesh vertex baking
            var meshWorldTransforms = new Dictionary<int, Matrix4x4>();
            if (_skeleton.Count > 0)
                BuildMeshTransformMap(scene.RootNode, Matrix4x4.Identity, meshWorldTransforms);

            var nodeMeshCounts = new Dictionary<string, int>();
            foreach (var kvp in meshNodeNames)
            {
                if (!nodeMeshCounts.ContainsKey(kvp.Value))
                    nodeMeshCounts[kvp.Value] = 0;
                nodeMeshCounts[kvp.Value]++;
            }

            for (int meshIdx = 0; meshIdx < meshes.Count; meshIdx++)
            {
                var mesh = meshes[meshIdx];
                int meshIndexCount = 0;
                int baseVertex = positions.Count;

                // For skinned FBX meshes, bake node world transform into vertices
                // so all parts share the same coordinate space (matching OffsetMatrix expectations).
                // Without this, each part's vertices are in its own node-local space.
                bool bakeNodeTransform = meshWorldTransforms.TryGetValue(meshIdx, out var meshWorldXform) && !isDAE && _skeleton.Count > 0;                            ;
                if (!bakeNodeTransform) meshWorldXform = Matrix4x4.Identity;

                for (int i = 0; i < mesh.VertexCount; i++)
                {
                    var pos = mesh.Vertices[i];
                    var norm = mesh.Normals[i];
                    var uv = mesh.HasTextureCoords(0) ? mesh.TextureCoordinateChannels[0][i] : new Vector3D(0, 0, 0);

                    if (isDAE)
                    {
                        positions.Add(new Vector3(pos.X, pos.Y, pos.Z) * _daeUnitScale);
                        normals.Add(new Vector3(norm.X, norm.Y, norm.Z));
                    }
                    else
                    {
                        var p = new Vector3(pos.X, pos.Y, pos.Z);
                        var n = new Vector3(norm.X, norm.Y, norm.Z);

                        if (bakeNodeTransform)
                        {
                            p = Vector3.Transform(p, meshWorldXform);
                            n = Vector3.Normalize(Vector3.TransformNormal(n, meshWorldXform));
                        }

                        positions.Add(new Vector3(-p.X, p.Y, -p.Z));
                        normals.Add(new Vector3(-n.X, n.Y, -n.Z));
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
                            if (boneIndex < 0)
                            {
                                Debug.Log($"[ModelImporter] WARNING: mesh bone '{bone.Name}' not found in skeleton ({bone.VertexWeightCount} weights dropped)");
                                continue;
                            }
                            foreach (Assimp.VertexWeight vw in bone.VertexWeights)
                            {
                                int vertexId = baseVertex + (int)vw.VertexID;
                                if (vertexId < weightMap.Count &&
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

                string nodeName = meshNodeNames.GetValueOrDefault(meshIdx);
                string partName = nodeName ?? mesh.Name ?? $"Part_{meshIdx}";

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

                // No LODs in the file: generate a chain. The new parts take their LOD 0 part's material slot.
                if (lods.Count == 0 && GenerateLODs && _skeleton.Count == 0)
                    lods = LODGenerator.Generate(gParts, gPos, gNorm, gIdx, LODRatios, LODMinTriangles, LODPixelError, groupName);


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

                // For skinned meshes, force identity — skinning pipeline handles positioning
                // via OffsetMatrix * bonePose. Node transforms would double-apply the
                // FBX geometric correction (scale 100, rot -90x).
                bool isSkinned = _skeleton.Count > 0 && data.BoneWeights != null;
                if (isSkinned)
                {
                    _groupTransforms[groupName] = (Vector3.Zero, System.Numerics.Quaternion.Identity, Vector3.One);
                }
                else if (pIndices.Count > 0)
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
                        _groupTransforms[groupName] = (Vector3.Zero, System.Numerics.Quaternion.Identity, s);
                    }
                }
            }

            //Debug.Log("ModelImporter", $"{result.Count} mesh group(s): " +
            //    string.Join(", ", result.Select(r => $"{r.name}({r.data.Parts.Count}p)")));

            // Sync mesh configs: one per produced mesh, keeping existing settings by name
            var meshConfigs = new List<MeshConfig>();
            foreach (var (groupName, _) in result)
                meshConfigs.Add(FindMeshConfig(groupName) ?? new MeshConfig { Name = groupName });
            Meshes = meshConfigs;

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
        /// Parse animations only from an Assimp scene.
        /// </summary>
        public List<AnimationClip> ExtractAnimations(Scene scene, float scale)
        {
            var clips = new List<AnimationClip>();
            if (scene.HasAnimations)
            {
                foreach (var anim in scene.Animations)
                    clips.Add(ExtractAnimation(scene, anim, scale));
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

                // Skip if an asset with this name already exists in AssetDatabase, unless it is this file's own
                // material from its last import: emitted again, it keeps its GUID (ImportAsset reuses sub-asset GUIDs
                // by name and type). Skipping that one as well dropped it from the meta on every reimport and left
                // the file's prefabs on DefaultMaterial (Cliff_Rocks_Sets_3A1.fbx, 2026-10-18).
                bool ownMaterial = AssetDatabase.HadSubAsset(filepath, matName, nameof(AssetDefinitionData));
                var existingGuid = ownMaterial ? null : AssetDatabase.ResolveGuidByName(matName);
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
            int idx = upper.IndexOf("LOD");
            if (idx >= 0)
            {
                int start = idx + 3;
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
            int idx = upper.LastIndexOf("LOD");
            if (idx >= 0)
            {
                int end = idx + 3;
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
            // Reusable scratch list to sort by weight descending
            var sorted = new List<KeyValuePair<int, float>>(8);

            for (int i = 0; i < weightMap.Count; i++)
            {
                BoneWeight weight = BoneWeight.Default;
                var map = weightMap[i];
                if (map.Count == 0)
                {
                    boneWeights[i] = weight;
                    continue;
                }

                // Sort by weight descending, keep strongest 4
                sorted.Clear();
                sorted.AddRange(map);
                sorted.Sort((a, b) => b.Value.CompareTo(a.Value));

                int count = Math.Min(sorted.Count, 4);
                if (count >= 1) { weight.BoneIDs.X = sorted[0].Key; weight.Weights.X = sorted[0].Value; }
                if (count >= 2) { weight.BoneIDs.Y = sorted[1].Key; weight.Weights.Y = sorted[1].Value; }
                if (count >= 3) { weight.BoneIDs.Z = sorted[2].Key; weight.Weights.Z = sorted[2].Value; }
                if (count >= 4) { weight.BoneIDs.W = sorted[3].Key; weight.Weights.W = sorted[3].Value; }

                // Normalize weights to sum to 1.0
                float sum = weight.Weights.X + weight.Weights.Y + weight.Weights.Z + weight.Weights.W;
                if (sum > 0 && MathF.Abs(sum - 1f) > 1e-6f)
                {
                    float inv = 1f / sum;
                    weight.Weights *= inv;
                }

                boneWeights[i] = weight;
            }
            return boneWeights;
        }

        // ══════════════════════════════════════════════════════════════
        // Animation Extraction (via Assimp)
        // ══════════════════════════════════════════════════════════════

        private AnimationClip ExtractAnimation(Scene scene, Assimp.Animation anim, float scale)
        {
            var clip = new AnimationClip();
            clip.Name = anim.Name;
            clip.Duration = (float)anim.DurationInTicks;
            clip.TicksPerSecond = anim.TicksPerSecond > 0 ? (float)anim.TicksPerSecond : 30f;

            // The skeleton has no pivot nodes (FlattenHierarchy folds them into the bone below), so
            // every channel has to give a bone's pose relative to the bone above it.
            //
            // A channel on a node itself already does: Assimp bakes the node's whole FBX transform
            // chain (translation, pre-rotation, ...) into it, and the only pivot nodes between that
            // chain and the bone above are geometric ones, which bones leave out.
            //
            // An FBX node with real pivots or geometric transforms gets its channels on its pivot
            // nodes instead. Those are composed into one channel for the bone afterwards.
            Dictionary<string, List<AnimationChannel>> pivotChannels = null;

            foreach (var nodeChannel in anim.NodeAnimationChannels)
            {
                var channel = new AnimationChannel { Target = nodeChannel.NodeName };

                int pivot = nodeChannel.NodeName.IndexOf(PivotMarker, StringComparison.Ordinal);

                // Position keys
                if (nodeChannel.HasPositionKeys)
                {
                    var posKeys = new List<VectorKey>(nodeChannel.PositionKeyCount);
                    foreach (var key in nodeChannel.PositionKeys)
                    {
                        var p = new Vector3(key.Value.X, key.Value.Y, key.Value.Z);
                        // Pivot channels are composed in file units and scaled then
                        posKeys.Add(new VectorKey { Time = (float)key.Time, Value = pivot < 0 ? p * scale : p });
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

                if (pivot < 0)
                {
                    clip.AddChannel(channel);
                    continue;
                }

                string bone = nodeChannel.NodeName.Substring(0, pivot);
                pivotChannels ??= new Dictionary<string, List<AnimationChannel>>();
                if (!pivotChannels.TryGetValue(bone, out var list))
                    pivotChannels[bone] = list = new List<AnimationChannel>();
                list.Add(channel);
            }

            if (pivotChannels != null)
            {
                foreach (var (bone, channels) in pivotChannels)
                    ComposePivotChannels(scene, clip, bone, channels, scale);
            }

            // Only a part of the take is the clip (ClipConfig.From / To)
            var config = FindClipConfig(clip.Name);
            if (config != null && config.To > config.From)
                clip.Cut(config.From, config.To);

            if (config != null && config.Turn != 0f)
                TurnClip(scene, clip, config.Turn);

            // In place: the travel its feet imply, on the root (ClipConfig.RootFromFeet). A file of clips alone has no
            // mesh to take a skeleton from: the nodes the clip moves make one.
            if (config != null && config.RootFromFeet)
            {
                var skeleton = _skeleton.Count > 0
                    ? new Skeleton { Bones = _skeleton.ToArray(), BoneNames = _boneNames.ToArray() }
                    : SkeletonOfNodes(scene, anim);
                if (!FootTravel.Add(clip, skeleton, config.RootVelocity, config.RootSurge))
                    Debug.LogWarning("ModelImporter", $"'{System.IO.Path.GetFileName(_currentFilePath)}' '{clip.Name}': RootFromFeet found no feet that slide; the clip stays in place");
            }

            return clip;
        }

        /// <summary>
        /// The clip turned <paramref name="degrees"/> about the upright through the model's origin
        /// (<see cref="ClipConfig.Turn"/>): the keys of every node right under the scene's root, pivot nodes passed over.
        /// A clip made around another spot than its neighbours is moved by its state instead (AnimationState.Offset).
        /// </summary>
        private static void TurnClip(Scene scene, AnimationClip clip, float degrees)
        {
            var turn = System.Numerics.Quaternion.CreateFromAxisAngle(Vector3.UnitY, degrees * MathF.PI / 180f);
            foreach (var channel in clip.Channels)
            {
                var parent = scene.RootNode.FindNode(channel.Target)?.Parent;
                while (parent != null && IsPivotNode(parent)) parent = parent.Parent;
                if (parent != scene.RootNode) continue;

                if (channel.Position != null)
                    for (int i = 0; i < channel.Position.Count; i++)
                        channel.Position[i].Value = Vector3.Transform(channel.Position[i].Value, turn);
                if (channel.Rotation != null)
                    for (int i = 0; i < channel.Rotation.Count; i++)
                        // The key's own turn first, then the clip's
                        channel.Rotation[i].Value = turn * channel.Rotation[i].Value;
            }
        }

        /// <summary>
        /// The nodes a clip moves and those above them as a skeleton (pivot nodes folded in, as in
        /// <see cref="FlattenHierarchy"/>): what FootTravel reads the feet off in a file that holds clips only. Not
        /// imported.
        /// </summary>
        private static Skeleton SkeletonOfNodes(Scene scene, Assimp.Animation anim)
        {
            var moved = new HashSet<string>();
            foreach (var channel in anim.NodeAnimationChannels)
            {
                string name = channel.NodeName;
                int pivot = name.IndexOf(PivotMarker, StringComparison.Ordinal);
                moved.Add(pivot >= 0 ? name.Substring(0, pivot) : name);
            }

            var kept = new HashSet<Node>();
            void Find(Node node)
            {
                if (moved.Contains(node.Name))
                    for (var n = node; n != null && kept.Add(n); n = n.Parent) { }
                if (node.HasChildren)
                    foreach (Node child in node.Children) Find(child);
            }
            Find(scene.RootNode);

            var bones = new List<Bone>();
            var names = new List<string>();
            void Add(Node node)
            {
                if (!kept.Contains(node)) return;
                if (!IsPivotNode(node))
                {
                    var bind = CollapsedLocal(node, out Node parent);
                    Matrix4x4.Decompose(bind, out var s, out var r, out var t);
                    names.Add(node.Name);
                    bones.Add(new Bone
                    {
                        Name = node.Name,
                        BindPoseMatrix = bind,
                        BindPose = new BonePose { Position = t, Rotation = r, Scale = s },
                        Parent = parent != null ? names.IndexOf(parent.Name) : -1,
                        OffsetMatrix = Matrix4x4.Identity,
                    });
                }
                if (node.HasChildren)
                    foreach (Node child in node.Children) Add(child);
            }
            Add(scene.RootNode);
            return new Skeleton { Bones = bones.ToArray(), BoneNames = names.ToArray() };
        }

        private const string PivotMarker = "_$AssimpFbx$_";
        private const string GeometricPivotMarker = "_$AssimpFbx$_Geometric";

        /// <summary>
        /// One channel for a bone whose FBX node is animated through its pivot nodes: at every key
        /// time of those, the bone's pose is its node times each pivot node above it, animated or
        /// not, as in <see cref="CollapsedLocal"/>.
        /// </summary>
        private static void ComposePivotChannels(Scene scene, AnimationClip clip, string boneName,
            List<AnimationChannel> pivotChannels, float scale)
        {
            var node = scene.RootNode.FindNode(boneName);
            if (node == null) return;

            foreach (var existing in clip.Channels)
                if (existing.Target == boneName) return; // keyed on the node as well: that channel stands

            var byNode = new Dictionary<string, AnimationChannel>();
            var times = new SortedSet<float>();
            foreach (var channel in pivotChannels)
            {
                byNode[channel.Target] = channel;
                for (int i = 0; i < (channel.Position?.Count ?? 0); i++) times.Add(channel.Position[i].Time);
                for (int i = 0; i < (channel.Rotation?.Count ?? 0); i++) times.Add(channel.Rotation[i].Time);
                for (int i = 0; i < (channel.Scale?.Count ?? 0); i++) times.Add(channel.Scale[i].Time);
            }
            if (times.Count == 0) return;

            var posKeys = new List<VectorKey>(times.Count);
            var rotKeys = new List<QuaternionKey>(times.Count);
            var scaleKeys = new List<VectorKey>(times.Count);

            foreach (float time in times)
            {
                var local = ToMatrix(node.Transform);
                for (var parent = node.Parent; parent != null && IsPivotNode(parent); parent = parent.Parent)
                {
                    if (IsGeometricPivot(parent)) continue;

                    if (byNode.TryGetValue(parent.Name, out var channel))
                    {
                        var pose = channel.GetBoneTransform(null, time);
                        local *= Matrix4x4.CreateScale(pose.Scale)
                               * Matrix4x4.CreateFromQuaternion(pose.Rotation)
                               * Matrix4x4.CreateTranslation(pose.Position);
                    }
                    else
                        local *= ToMatrix(parent.Transform);
                }

                if (!Matrix4x4.Decompose(local, out var s, out var r, out var t)) continue;

                posKeys.Add(new VectorKey { Time = time, Value = t * scale });
                rotKeys.Add(new QuaternionKey { Time = time, Value = r });
                scaleKeys.Add(new VectorKey { Time = time, Value = s });
            }
            if (rotKeys.Count == 0) return;

            clip.AddChannel(new AnimationChannel
            {
                Target = boneName,
                Position = new VectorKeys(posKeys),
                Rotation = new QuaternionKeys(rotKeys),
                Scale = new VectorKeys(scaleKeys),
            });
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
            CorrectOffsetMatrices();

        }

        /// <summary>
        /// Assimp may inject structural nodes ($AssimpFbx$_GeometricScaling, _PreRotation, etc.)
        /// that carry transforms not accounted for in the original FBX bone OffsetMatrix.
        /// For each skinning bone, verify that OffsetMatrix * bindWorld ≈ Identity.
        /// If not, recompute OffsetMatrix = Inv(bindWorld) from the hierarchy.
        /// </summary>
        private void CorrectOffsetMatrices()
        {
            // Build bind world matrices from hierarchy
            var worldMatrices = new Matrix4x4[_skeleton.Count];
            for (int i = 0; i < _skeleton.Count; i++)
            {
                worldMatrices[i] = _skeleton[i].BindPoseMatrix;
                if (_skeleton[i].Parent >= 0)
                    worldMatrices[i] = worldMatrices[i] * worldMatrices[_skeleton[i].Parent];
            }

            for (int i = 0; i < _skeleton.Count; i++)
            {
                if (_skeleton[i].OffsetMatrix == Matrix4x4.Identity)
                    continue; // structural node, no offset to correct

                var product = _skeleton[i].OffsetMatrix * worldMatrices[i];

                // Check if offset * bindWorld ≈ Identity (diagonal ≈ 1, off-diagonal ≈ 0)
                float diagError = MathF.Abs(product.M11 - 1) + MathF.Abs(product.M22 - 1)
                                + MathF.Abs(product.M33 - 1) + MathF.Abs(product.M44 - 1);
                float offDiag = MathF.Abs(product.M12) + MathF.Abs(product.M13)
                              + MathF.Abs(product.M21) + MathF.Abs(product.M23)
                              + MathF.Abs(product.M31) + MathF.Abs(product.M32);

                if (diagError + offDiag > 0.1f)
                {
                    if (Matrix4x4.Invert(worldMatrices[i], out var corrected))
                        _skeleton[i].OffsetMatrix = corrected;
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

        /// <summary>
        /// Assimp splits an FBX node's transform chain into extra nodes named
        /// "&lt;node&gt;_$AssimpFbx$_&lt;part&gt;" (translation, pre-rotation, geometric transforms and their
        /// inverses). They are not bones: nothing is skinned to them and no clip moves them on their own.
        /// </summary>
        private static bool IsPivotNode(Node node) => node.Name.Contains(PivotMarker, StringComparison.Ordinal);

        /// <summary>
        /// A pivot node for an FBX geometric transform, or for the inverse of one, which Assimp puts
        /// below the node to keep the transform from its children. A geometric transform moves the
        /// geometry attached to a node, not the node: bones leave both out. Rigs carry unit scales
        /// there (×100 on a bone, ×0.01 on the one below), which must not end up in a bind pose.
        /// </summary>
        private static bool IsGeometricPivot(Node node) => node.Name.Contains(GeometricPivotMarker, StringComparison.Ordinal);

        /// <summary>
        /// A node's transform relative to the nearest ancestor that is not a pivot node, with the
        /// pivot nodes in between folded in, except geometric ones.
        /// world(node) = local(node) * local(parent) * ...
        /// </summary>
        private static Matrix4x4 CollapsedLocal(Node node, out Node parent)
        {
            var local = ToMatrix(node.Transform);
            for (parent = node.Parent; parent != null && IsPivotNode(parent); parent = parent.Parent)
            {
                if (!IsGeometricPivot(parent))
                    local *= ToMatrix(parent.Transform);
            }
            return local;
        }

        private void FlattenHierarchy(Node node)
        {
            if (!_validNodes.ContainsKey(node.Name)) return;
            if (_validNodes[node.Name] == false) return;

            // Pivot nodes are folded into the bone below them: they made up more than half of some
            // skeletons, and a clip that keys a bone already includes them (see ExtractAnimation).
            if (IsPivotNode(node))
            {
                if (node.HasChildren)
                {
                    foreach (Node child in node.Children)
                        FlattenHierarchy(child);
                }
                return;
            }

            _boneNames.Add(node.Name);

            var newBone = new Bone { Name = node.Name };
            Matrix4x4 bind = CollapsedLocal(node, out Node parent);
            Matrix4x4.Decompose(bind, out var s, out var r, out var t);


            newBone.BindPoseMatrix = bind;
            newBone.BindPose = new BonePose { Position = t, Rotation = r, Scale = s };

            newBone.Parent = parent != null ? _boneNames.IndexOf(parent.Name) : -1;
            _skeleton.Add(newBone);

            if (_bones.TryGetValue(node.Name, out Assimp.Bone? value))
            {
                var om = ToMatrix(value.OffsetMatrix);
                // Compensate for DAE vertex unit scaling: OM_new = S(1/unit) * OM
                // so that: vertex_m * OM_new = vertex_cm * OM (skinning unchanged)
                if (isDAE && MathF.Abs(_daeUnitScale) > 1e-6f)
                {
                    float inv = 1f / _daeUnitScale;
                    om.M11 *= inv; om.M12 *= inv; om.M13 *= inv; om.M14 *= inv;
                    om.M21 *= inv; om.M22 *= inv; om.M23 *= inv; om.M24 *= inv;
                    om.M31 *= inv; om.M32 *= inv; om.M33 *= inv; om.M34 *= inv;
                }
                newBone.OffsetMatrix = om;
            }
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

        /// <summary>The way back from <see cref="ToMatrix"/>.</summary>
        private static Assimp.Matrix4x4 ToAssimp(Matrix4x4 m)
        {
            var t = Matrix4x4.Transpose(m);
            return new Assimp.Matrix4x4(
                t.M11, t.M12, t.M13, t.M14,
                t.M21, t.M22, t.M23, t.M24,
                t.M31, t.M32, t.M33, t.M34,
                t.M41, t.M42, t.M43, t.M44);
        }

        /// <summary>
        /// Conjugate a matrix by the -X/-Z flip: F * M * F where F = diag(-1,1,-1,1).
        /// Makes skeleton/animation data consistent with FBX mesh vertex flip.
        /// </summary>
        private static Matrix4x4 FlipXZ(Matrix4x4 m)
        {
            return new Matrix4x4(
                 m.M11, -m.M12,  m.M13, -m.M14,
                -m.M21,  m.M22, -m.M23,  m.M24,
                 m.M31, -m.M32,  m.M33, -m.M34,
                -m.M41,  m.M42, -m.M43,  m.M44
            );
        }
    }
}
