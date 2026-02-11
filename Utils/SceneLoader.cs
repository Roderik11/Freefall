using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Freefall.Assets;
using static Freefall.Assets.InternalAssets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Loaders
{
    public class SceneLoader
    {
        public Dictionary<string, StaticMesh> staticMeshLookup = new Dictionary<string, StaticMesh>();
        private Dictionary<string, string> meshLookup = new Dictionary<string, string>();
        private string assetsDirectory;


        public class SceneExport
        {
            public List<SceneObject> Objects { get; set; } = new List<SceneObject>();
            public List<SceneObject> Trees { get; set; } = new List<SceneObject>();
            public List<SceneLight> Lights { get; set; } = new List<SceneLight>();
            public List<SceneAudioSource> AudioSources { get; set; } = new List<SceneAudioSource>();
        }

        [Serializable]
        public class SceneLight
        {
            public string Type { get; set; } = "Point";
            public string Position { get; set; } = "";
            public string Color { get; set; } = "1;1;1";
            public float Intensity { get; set; } = 1;
            public float Range { get; set; } = 10;
        }

        [Serializable]
        public class SceneAudioSource
        {
            public string Name { get; set; } = "AudioSource";
            public string Position { get; set; } = "";
            public string Rotation { get; set; } = "";
            public string Scale { get; set; } = "";
            public string ClipName { get; set; } = "";
            public float Volume { get; set; } = 1;
            public float Pitch { get; set; } = 1;
            public float MinDistance { get; set; } = 0;
            public float MaxDistance { get; set; } = 10;
            public bool Loop { get; set; }
            public bool PlayOnAwake { get; set; } = true;
        }

        [Serializable]
        public class SceneObject
        {
            public string PrefabName { get; set; } = "";
            public string Position { get; set; } = "";
            public string Rotation { get; set; } = "";
            public string Scale { get; set; } = "";
            public bool Collision { get; set; }
        }

        public SceneLoader(string assetsDir)
        {
            assetsDirectory = assetsDir;
            var root = new DirectoryInfo(assetsDirectory);
            var files = Directory.GetFiles(assetsDirectory, "*.fbx", SearchOption.AllDirectories);
            var relativePaths = files.Select(f => f.Replace(root.FullName, string.Empty)).ToHashSet();

            foreach (var path in relativePaths)
            {
                var fileName = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
                if (!meshLookup.ContainsKey(fileName))
                    meshLookup.Add(fileName, path);
            }

            Debug.Log($"[SceneLoader] Found {meshLookup.Count} unique meshes in {assetsDirectory}");
        }

        public void Load(string path, int maxcount = int.MaxValue)
        {
            Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

            var text = File.ReadAllText(path);
            var scene = JsonSerializer.Deserialize<SceneExport>(text, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (scene == null) return;

            // Load Objects
            int count = 0;
            foreach (var obj in scene.Objects)
            {
                if (!staticMeshLookup.TryGetValue(obj.PrefabName, out var staticMesh))
                {
                    if (!meshLookup.TryGetValue(obj.PrefabName.ToLowerInvariant(), out var relativePath))
                        continue;

                    var fullPath = Path.Combine(assetsDirectory, relativePath.TrimStart('\\'));
                    var mesh = Engine.Assets.Load<Mesh>(fullPath);
                    staticMesh = CreateStaticMesh(mesh, fullPath);
                    staticMeshLookup.Add(obj.PrefabName, staticMesh);
                }

                var entity = new Entity(obj.PrefabName);
                var pos = StringToVector3(obj.Position);
                var rot = StringToQuaternion(obj.Rotation) * Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)Math.PI);
                var scale = StringToVector3(obj.Scale);
                
                entity.Transform.Position = pos;
                entity.Transform.Rotation = rot;
                entity.Transform.Scale = scale;

                var renderer = entity.AddComponent<StaticMeshRenderer>();
                renderer.StaticMesh = staticMesh;

                count++;
                if (count >= maxcount)
                    break;
            }

            // Load Lights
            foreach (var light in scene.Lights)
            {
                if (string.Equals(light.Type, "Point", StringComparison.OrdinalIgnoreCase))
                {
                    var entity = new Entity("PointLight");
                    entity.Transform.Position = StringToVector3(light.Position);

                    var pointLight = entity.AddComponent<PointLight>();
                    pointLight.Color = StringToColor3(light.Color);
                    pointLight.Intensity = light.Intensity;
                    pointLight.Range = light.Range;
                }
            }

            // Load Audio Sources
            foreach (var audio in scene.AudioSources)
            {
                if (string.IsNullOrEmpty(audio.ClipName)) continue;

                var clipPath = Path.Combine(assetsDirectory, "Sounds", audio.ClipName + ".wav");
                if (!File.Exists(clipPath))
                {
                    Debug.Log($"[SceneLoader] Audio clip not found: {clipPath}");
                    continue;
                }

                var entity = new Entity(audio.Name);
                entity.Transform.Position = StringToVector3(audio.Position);

                var src = entity.AddComponent<AudioSource>();
                src.AudioClip = Engine.Assets.Load<AudioClip>(clipPath);
                src.Volume = audio.Volume;
                src.Range = audio.MaxDistance;
                src.MinDistance = audio.MinDistance;
                src.Loop = audio.Loop;
                src.PlayOnAwake = audio.PlayOnAwake;

                Debug.Log($"[SceneLoader] Audio: {audio.Name} clip={audio.ClipName} pos={audio.Position} range={audio.MaxDistance}");
            }

            // Load Trees
            count = 0;
            foreach (var data in scene.Trees)
            {
                if (!staticMeshLookup.TryGetValue(data.PrefabName, out var staticMesh))
                {
                    if (!meshLookup.TryGetValue(data.PrefabName.ToLowerInvariant(), out var relativePath))
                        continue;

                    var fullPath = Path.Combine(assetsDirectory, relativePath.TrimStart('\\'));
                    var mesh = Engine.Assets.Load<Mesh>(fullPath);
                    staticMesh = CreateStaticMesh(mesh, fullPath, false);
                    staticMeshLookup.Add(data.PrefabName, staticMesh);
                }

                var entity = new Entity(data.PrefabName);
                entity.Transform.Position = StringToVector3(data.Position);
                entity.Transform.Rotation = StringToQuaternion(data.Rotation) * Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)Math.PI);
                entity.Transform.Scale = StringToVector3(data.Scale);

                var renderer = entity.AddComponent<StaticMeshRenderer>();
                renderer.StaticMesh = staticMesh;

                count++;
                if (count >= maxcount)
                    break;
            }
        }

        public async Task LoadAsync(string path, IProgress<string> progress = null, CancellationToken cancellationToken = default)
        {
            progress?.Report($"Reading {Path.GetFileName(path)}...");
            cancellationToken.ThrowIfCancellationRequested();

            string text = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);

            progress?.Report("Parsing JSON...");
            var scene = JsonSerializer.Deserialize<SceneExport>(text, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (scene == null) return;

            // Collect all items with their mesh paths
            var allItems = new List<(SceneObject obj, bool isTree)>();
            foreach (var obj in scene.Objects)
            {
                if (!meshLookup.TryGetValue(obj.PrefabName.ToLowerInvariant(), out _)) continue;
                allItems.Add((obj, false));
            }
            foreach (var data in scene.Trees)
            {
                if (!meshLookup.TryGetValue(data.PrefabName.ToLowerInvariant(), out _)) continue;
                allItems.Add((data, true));
            }

            // Group objects by their mesh path for batch loading
            var objectsByMesh = new Dictionary<string, List<(SceneObject obj, bool isTree)>>();
            foreach (var (obj, isTree) in allItems)
            {
                var relativePath = meshLookup[obj.PrefabName.ToLowerInvariant()];
                var fullPath = Path.Combine(assetsDirectory, relativePath.TrimStart('\\'));
                if (!objectsByMesh.TryGetValue(fullPath, out var list))
                {
                    list = new List<(SceneObject, bool)>();
                    objectsByMesh[fullPath] = list;
                }
                list.Add((obj, isTree));
            }

            int totalMeshes = objectsByMesh.Count;
            int meshesParsed = 0;
            int created = 0;

            TextureLibrary.Initialize();
            var wicLock = new object();

            // ═══════════════════════════════════════════════════════════════════
            // PHASE 1: CPU-only parsing on background threads
            // ═══════════════════════════════════════════════════════════════════

            var parsedEntries = new System.Collections.Concurrent.ConcurrentBag<(
                string meshPath,
                string meshName,
                MeshData meshData,
                Assets.CpuTextureData? albedo,
                Assets.CpuTextureData? normal,
                List<(SceneObject obj, bool isTree)> instances
            )>();

            await Task.Run(() =>
            {
                var parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount / 2,
                    CancellationToken = cancellationToken
                };

                Parallel.ForEach(objectsByMesh, parallelOptions, (entry) =>
                {
                    var (fullPath, objects) = entry;
                    Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

                    int currentMesh = Interlocked.Increment(ref meshesParsed);
                    if (currentMesh % 10 == 0)
                        progress?.Report($"[Phase 1] Parsing [{currentMesh}/{totalMeshes}]...");

                    try
                    {
                        // Parse FBX → MeshData
                        var importer = new Assets.Importers.MeshImporter();
                        var meshData = importer.ParseRaw(fullPath);
                        var meshName = Path.GetFileNameWithoutExtension(fullPath);

                        // Parse textures
                        Assets.CpuTextureData albedo = null;
                        Assets.CpuTextureData normal = null;

                        var texturePath = Path.GetDirectoryName(fullPath) + @"\Textures\";
                        if (Directory.Exists(texturePath))
                        {
                            foreach (var file in Directory.EnumerateFiles(texturePath))
                            {
                                if (file.EndsWith(".psd", StringComparison.OrdinalIgnoreCase)) continue;
                                var name = Path.GetFileNameWithoutExtension(file);
                                try
                                {
                                    var loadPath = TextureLibrary.ResolvePackedDDS(file) ?? file;
                                    bool isDDS = loadPath.EndsWith(".dds", StringComparison.OrdinalIgnoreCase);

                                    if (name.EndsWith("_Dif") || name.EndsWith("_diffuse") || name.EndsWith("_Diffuse"))
                                    {
                                        if (isDDS) albedo = Texture.ParseFromFile(Engine.Device, loadPath);
                                        else lock (wicLock) { albedo = Texture.ParseFromFile(Engine.Device, loadPath); }
                                    }
                                    else if (name.EndsWith("_Nor") || name.EndsWith("_normal") || name.EndsWith("_Normal"))
                                    {
                                        if (isDDS) normal = Texture.ParseFromFile(Engine.Device, loadPath);
                                        else lock (wicLock) { normal = Texture.ParseFromFile(Engine.Device, loadPath); }
                                    }
                                }
                                catch (Exception) { /* log */ }
                            }
                        }

                        parsedEntries.Add((fullPath, meshName, meshData, albedo, normal, objects));
                    }
                    catch (Exception ex)
                    {
                        Debug.Log($"[SceneLoader] Failed to parse {Path.GetFileNameWithoutExtension(fullPath)}: {ex.Message}");
                    }
                });
            }, cancellationToken);

            progress?.Report($"[Phase 1] Done. Streaming to GPU...");

            // ═══════════════════════════════════════════════════════════════════
            // PHASE 2: GPU Asset Creation (Main Thread - Non-Blocking)
            // ═══════════════════════════════════════════════════════════════════

            int uploaded = 0;
            var createdTasks = new List<Task>();

            foreach (var entry in parsedEntries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Dispatch to main thread to Create Commited Resources + Entities
                // Since we use CreateAsync, this will be very fast
                var task = Engine.RunOnMainThreadAsync(() =>
                {
                    var (meshPath, meshName, meshData, albedoCpu, normalCpu, instances) = entry;

                    // 1. Create Mesh (Buffers created, upload queued)
                    var mesh = Mesh.CreateAsync(Engine.Device, meshData);
                    mesh.Name = meshName;
                    mesh.RegisterMeshParts();

                    // 2. Create Textures (Resources created, upload queued)
                    Texture albedoTex = albedoCpu != null ? Texture.CreateAsync(Engine.Device, albedoCpu) : null;
                    Texture normalTex = normalCpu != null ? Texture.CreateAsync(Engine.Device, normalCpu) : null;

                    // 3. Build Material
                    Material material;
                    if (albedoTex != null || normalTex != null)
                    {
                        material = new Material(DefaultEffect);
                        material.SetTexture("AlbedoTex", albedoTex ?? White);
                        material.SetTexture("NormalTex", normalTex ?? FlatNormal);
                    }
                    else
                    {
                        material = DefaultMaterial;
                    }

                    // 4. Create Entities
                    foreach (var (obj, isTree) in instances)
                    {
                        var key = obj.PrefabName;

                        if (!staticMeshLookup.TryGetValue(key, out var staticMesh))
                        {
                            staticMesh = new StaticMesh { Name = meshName, Mesh = mesh, LODGroup = LODGroups.LargeProps };
                            // Populate StaticMesh Parts
                            var allLODs = mesh.MeshParts.FindAll(m => m.Name.Contains("LOD_"));
                            bool allAreLODs = (mesh.MeshParts.Count - allLODs.Count) <= 0;

                             if (allAreLODs)
                            {
                                for (int p = 0; p < mesh.MeshParts.Count; p++)
                                    staticMesh.MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                            }
                            else
                            {
                                for (int p = 0; p < mesh.MeshParts.Count; p++)
                                {
                                    var part = mesh.MeshParts[p];
                                    int index = part.Name.IndexOf("_LOD_");
                                    if (index < 0)
                                    {
                                        staticMesh.MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                                    }
                                    else
                                    {
                                        int endIndex = index + 6 < part.Name.Length ? 2 : 1;
                                        int lvl = Convert.ToInt32(part.Name.Substring(index + 5, endIndex)) - 1;
                                        while (staticMesh.LODs.Count < lvl + 1)
                                            staticMesh.LODs.Add(new StaticMeshLOD { Mesh = mesh });
                                        staticMesh.LODs[lvl].MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                                    }
                                }
                            }
                            staticMeshLookup[key] = staticMesh;
                        }

                        var entity = new Entity(key);
                        entity.Transform.Position = StringToVector3(obj.Position);
                        entity.Transform.Rotation = StringToQuaternion(obj.Rotation) * Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)Math.PI);
                        entity.Transform.Scale = StringToVector3(obj.Scale);

                        var renderer = entity.AddComponent<StaticMeshRenderer>();
                        renderer.StaticMesh = staticMesh;
                        
                        created++;
                    }
                    
                    int cur = Interlocked.Increment(ref uploaded);
                    if (cur % 50 == 0) 
                        progress?.Report($"[Phase 2] Created {cur}/{parsedEntries.Count} mesh assets...");
                });
                createdTasks.Add(task);
            }

            await Task.WhenAll(createdTasks);

            // Load Lights (lightweight — no mesh/texture loading needed)
            int lightCount = 0;
            foreach (var light in scene.Lights)
            {
                if (string.Equals(light.Type, "Point", StringComparison.OrdinalIgnoreCase))
                {
                    await Engine.RunOnMainThreadAsync(() =>
                    {
                        var entity = new Entity("PointLight");
                        entity.Transform.Position = StringToVector3(light.Position);

                        var pointLight = entity.AddComponent<PointLight>();
                        pointLight.Color = StringToColor3(light.Color);
                        pointLight.Intensity = light.Intensity;
                        pointLight.Range = light.Range;
                    });
                    lightCount++;
                }
            }

            // Load Audio Sources
            int audioCount = 0;
            foreach (var audio in scene.AudioSources)
            {
                if (string.IsNullOrEmpty(audio.ClipName)) continue;

                var clipPath = Path.Combine(assetsDirectory, "Sounds", audio.ClipName + ".wav");
                if (!File.Exists(clipPath))
                {
                    Debug.Log($"[SceneLoader] Audio clip not found: {clipPath}");
                    continue;
                }

                await Engine.RunOnMainThreadAsync(() =>
                {
                    var entity = new Entity(audio.Name);
                    entity.Transform.Position = StringToVector3(audio.Position);

                    var src = entity.AddComponent<AudioSource>();
                    src.AudioClip = Engine.Assets.Load<AudioClip>(clipPath);
                    src.Volume = audio.Volume;
                    src.Range = audio.MaxDistance;
                    src.MinDistance = audio.MinDistance;
                    src.Loop = audio.Loop;
                    src.PlayOnAwake = audio.PlayOnAwake;

                    Debug.Log($"[SceneLoader] Audio: {audio.Name} clip={audio.ClipName} range={audio.MaxDistance}");
                });
                audioCount++;
            }

            progress?.Report($"Done — {created} entities, {lightCount} lights, {audioCount} audio sources in scene.");
        }

        private StaticMesh CreateStaticMesh(Mesh mesh, string meshPath, bool tree = false)
        {
            var staticMesh = new StaticMesh { Name = mesh.Name, Mesh = mesh, LODGroup = LODGroups.LargeProps };
            var allLODs = mesh.MeshParts.FindAll((m) => m.Name.Contains("LOD_"));
            var material = DefaultMaterial;

            if (mesh.MeshParts.Count - allLODs.Count <= 0)
            {
                for (int i = 0; i < mesh.MeshParts.Count; i++)
                {
                    staticMesh.MeshParts.Add(new MeshElement
                    {
                        Mesh = mesh,
                        Material = material,
                        MeshPartIndex = i
                    });
                }
                return staticMesh;
            }

            for (int i = 0; i < mesh.MeshParts.Count; i++)
            {
                // ... [Same LOD logic preserved]
                var part = mesh.MeshParts[i];
                var name = part.Name;
                int index = name.IndexOf("_LOD_");
                if (index < 0)
                {
                    staticMesh.MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = i });
                }
                else
                {
                    int endIndex = index + 6 < name.Length ? 2 : 1;
                    int lvl = Convert.ToInt32(name.Substring(index + 5, endIndex)) - 1;
                    while (staticMesh.LODs.Count < lvl + 1) staticMesh.LODs.Add(new StaticMeshLOD { Mesh = mesh });
                    staticMesh.LODs[lvl].MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = i });
                }
            }
            return staticMesh;
        }

        private static Vector3 StringToVector3(string str)
        {
            if (string.IsNullOrEmpty(str)) return Vector3.Zero;
            var parts = str.Split(';');
            if (parts.Length != 3) return Vector3.Zero;
            return new Vector3(
                float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture),
                float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture)
            );
        }

        private static Color3 StringToColor3(string str)
        {
            var v = StringToVector3(str);
            return new Color3(v.X, v.Y, v.Z);
        }

        private static Quaternion StringToQuaternion(string str)
        {
            if (string.IsNullOrEmpty(str)) return Quaternion.Identity;
            var parts = str.Split(';');
            if (parts.Length != 4) return Quaternion.Identity;
            return new Quaternion(
                float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture),
                float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture),
                float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture)
            );
        }
    }
}
