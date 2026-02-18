using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Freefall.Assets;
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

        #region JSON Schema

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

        #endregion

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

        /// <summary>
        /// Synchronous load — used by editor and small scenes.
        /// </summary>
        public void Load(string path, int maxcount = int.MaxValue)
        {
            Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

            var scene = ParseJson(File.ReadAllText(path));
            if (scene == null) return;

            int count = 0;
            foreach (var obj in scene.Objects.Concat(scene.Trees))
            {
                var staticMesh = ResolveStaticMesh(obj.PrefabName);
                if (staticMesh == null) continue;

                SpawnEntity(obj, staticMesh);
                if (++count >= maxcount) break;
            }

            SpawnLights(scene.Lights);
            SpawnAudio(scene.AudioSources);
        }

        /// <summary>
        /// Async load — loads all unique meshes in parallel via Assets.LoadAsync&lt;StaticMesh&gt;,
        /// then spawns entities on the main thread in batches.
        /// </summary>
        public async Task LoadAsync(string path, IProgress<string> progress = null, CancellationToken cancellationToken = default)
        {
            progress?.Report($"Reading {Path.GetFileName(path)}...");
            string text = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);

            var scene = ParseJson(text);
            if (scene == null) return;

            // Merge Objects + Trees into a single list
            var allItems = scene.Objects.Concat(scene.Trees)
                .Where(o => meshLookup.ContainsKey(o.PrefabName.ToLowerInvariant()))
                .ToList();

            // Group by mesh path → only load each mesh once
            var groups = allItems
                .GroupBy(o => meshLookup[o.PrefabName.ToLowerInvariant()])
                .ToList();

            int totalMeshes = groups.Count;
            int loaded = 0;

            TextureLibrary.Initialize();

            // ── Stream: load mesh → immediately spawn its instances in small batches ──
            progress?.Report($"Loading {totalMeshes} unique meshes...");

            int totalSpawned = 0;
            int totalInstances = allItems.Count;
            const int spawnBatchSize = 256;
            var throttle = new SemaphoreSlim(2); // limit concurrent GPU resource creation

            var loadTasks = groups.Select(async group =>
            {
                await throttle.WaitAsync(cancellationToken);
                StaticMesh staticMesh;
                try
                {
                    var fullPath = Path.Combine(assetsDirectory, group.Key.TrimStart('\\'));
                    staticMesh = await Engine.Assets.LoadAsync<StaticMesh>(fullPath);
                }
                finally
                {
                    throttle.Release();
                }

                // Register for lookup
                foreach (var obj in group)
                {
                    lock (staticMeshLookup)
                        staticMeshLookup[obj.PrefabName] = staticMesh;
                }

                int current = Interlocked.Increment(ref loaded);
                progress?.Report($"Loaded [{current}/{totalMeshes}] — spawning instances...");

                // Immediately spawn this mesh's instances in small batches
                var instances = group.ToList();
                for (int i = 0; i < instances.Count; i += spawnBatchSize)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int end = Math.Min(i + spawnBatchSize, instances.Count);
                    var batch = instances.GetRange(i, end - i);
                    var sm = staticMesh; // capture for closure

                    await Engine.RunOnMainThreadAsync(() =>
                    {
                        foreach (var obj in batch)
                            SpawnEntity(obj, sm);
                    });

                    int spawned = Interlocked.Add(ref totalSpawned, batch.Count);
                    if (spawned % 100 < spawnBatchSize)
                        progress?.Report($"Spawned {spawned}/{totalInstances} entities...");
                }
            }).ToList();

            await Task.WhenAll(loadTasks);

            // ── Lights (lightweight) ──
            if (scene.Lights.Count > 0)
            {
                await Engine.RunOnMainThreadAsync(() => SpawnLights(scene.Lights));
            }

            // ── Audio: pre-load clips on background thread, then spawn ──
            if (scene.AudioSources.Count > 0)
            {
                // Pre-load all audio clips off the main thread
                var audioClips = new Dictionary<string, AudioClip>();
                await Task.Run(() =>
                {
                    foreach (var audio in scene.AudioSources)
                    {
                        if (string.IsNullOrEmpty(audio.ClipName) || audioClips.ContainsKey(audio.ClipName)) continue;
                        var clipPath = Path.Combine(assetsDirectory, "Sounds", audio.ClipName + ".wav");
                        if (!File.Exists(clipPath)) continue;
                        audioClips[audio.ClipName] = Engine.Assets.Load<AudioClip>(clipPath);
                    }
                });

                // Spawn audio entities with pre-loaded clips
                await Engine.RunOnMainThreadAsync(() => SpawnAudio(scene.AudioSources, audioClips));
            }

            progress?.Report($"Done — {totalInstances} entities, {scene.Lights.Count} lights, {scene.AudioSources.Count} audio sources.");
        }

        #region Helpers

        private SceneExport ParseJson(string text)
        {
            return JsonSerializer.Deserialize<SceneExport>(text, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        }

        /// <summary>
        /// Resolves prefab name → StaticMesh. Loads via AssetManager if not cached.
        /// </summary>
        private StaticMesh ResolveStaticMesh(string prefabName)
        {
            if (staticMeshLookup.TryGetValue(prefabName, out var cached))
                return cached;

            if (!meshLookup.TryGetValue(prefabName.ToLowerInvariant(), out var relativePath))
                return null;

            var fullPath = Path.Combine(assetsDirectory, relativePath.TrimStart('\\'));
            var staticMesh = Engine.Assets.Load<StaticMesh>(fullPath);
            staticMeshLookup[prefabName] = staticMesh;
            return staticMesh;
        }

        private static void SpawnEntity(SceneObject obj, StaticMesh staticMesh)
        {
            var entity = new Entity(obj.PrefabName);
            entity.Transform.Position = StringToVector3(obj.Position);
            entity.Transform.Rotation = StringToQuaternion(obj.Rotation) * Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)Math.PI);
            entity.Transform.Scale = StringToVector3(obj.Scale);

            var renderer = entity.AddComponent<StaticMeshRenderer>();
            renderer.StaticMesh = staticMesh;

            if (obj.Collision)
            {
                var body = entity.AddComponent<RigidBody>();
                body.StaticMesh = staticMesh;
                body.Type = ShapeType.StaticMesh;
                body.IsStatic = true;
            }
        }

        private void SpawnLights(List<SceneLight> lights)
        {
            foreach (var light in lights)
            {
                if (!string.Equals(light.Type, "Point", StringComparison.OrdinalIgnoreCase)) continue;

                var entity = new Entity("PointLight");
                entity.Transform.Position = StringToVector3(light.Position);

                var pointLight = entity.AddComponent<PointLight>();
                pointLight.Color = StringToColor3(light.Color);
                pointLight.Intensity = light.Intensity;
                pointLight.Range = light.Range;
            }
        }

        private void SpawnAudio(List<SceneAudioSource> audioSources, Dictionary<string, AudioClip> preloadedClips = null)
        {
            foreach (var audio in audioSources)
            {
                if (string.IsNullOrEmpty(audio.ClipName)) continue;

                AudioClip clip;
                if (preloadedClips != null && preloadedClips.TryGetValue(audio.ClipName, out clip))
                {
                    // Use pre-loaded clip (async path)
                }
                else
                {
                    // Fallback: load synchronously (sync Load path)
                    var clipPath = Path.Combine(assetsDirectory, "Sounds", audio.ClipName + ".wav");
                    if (!File.Exists(clipPath))
                    {
                        Debug.Log($"[SceneLoader] Audio clip not found: {clipPath}");
                        continue;
                    }
                    clip = Engine.Assets.Load<AudioClip>(clipPath);
                }

                var entity = new Entity(audio.Name);
                entity.Transform.Position = StringToVector3(audio.Position);

                var src = entity.AddComponent<AudioSource>();
                src.AudioClip = clip;
                src.Volume = audio.Volume;
                src.Range = audio.MaxDistance;
                src.MinDistance = audio.MinDistance;
                src.Loop = audio.Loop;
                src.PlayOnAwake = audio.PlayOnAwake;
            }
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

        #endregion
    }
}
