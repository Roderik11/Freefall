using System;
using System.IO;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads StaticMesh assets from cache (.asset files).
    /// Unpacks AssetDefinitionData (YAML), deserializes the StaticMesh definition,
    /// then resolves all GUID references (Mesh, Material, Texture) via AssetManager.
    /// Finally cooks the physics mesh.
    /// </summary>
    [AssetLoader(typeof(StaticMesh))]
    public class StaticMeshLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for static mesh '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager)
        {
            try
            {
                Debug.Log($"[StaticMeshLoader] Loading '{name}' from {cachePath}");
                Console.Out.Flush();

                AssetDefinitionData defData;
                using (var stream = File.OpenRead(cachePath))
                    defData = _packer.Read(stream);

                // Deserialize YAML → StaticMesh
                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                Debug.Log($"[StaticMeshLoader] Deserializing YAML for '{name}' ({yaml.Length} chars)");
                Console.Out.Flush();

                StaticMesh staticMesh;
                try
                {
                    staticMesh = NativeImporter.LoadFromString(yaml) as StaticMesh;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"Failed to deserialize StaticMesh from cache: {name} — {ex.GetType().Name}: {ex.Message}", ex);
                }

                if (staticMesh == null)
                {
                    var preview = yaml.Length > 200 ? yaml[..200] + "..." : yaml;
                    throw new InvalidDataException(
                        $"Failed to deserialize StaticMesh from cache: {name}. " +
                        $"TypeName in cache: '{defData.TypeName}'. YAML preview: {preview}");
                }

                staticMesh.Name = name;

                // ── Resolve GUID references ──

                // Resolve the main Mesh
                if (staticMesh.Mesh != null && !string.IsNullOrEmpty(staticMesh.Mesh.Guid))
                {
                    Debug.Log($"[StaticMeshLoader] Resolving Mesh GUID '{staticMesh.Mesh.Guid}' for '{name}'");
                    Console.Out.Flush();
                    staticMesh.Mesh = manager.LoadByGuid<Mesh>(staticMesh.Mesh.Guid);
                    Debug.Log($"[StaticMeshLoader] Mesh resolved: {(staticMesh.Mesh != null ? "OK" : "NULL")}");
                    Console.Out.Flush();
                }

                // Resolve MeshPart references
                if (staticMesh.MeshParts != null)
                {
                    foreach (var element in staticMesh.MeshParts)
                    {
                        if (element.Mesh != null && !string.IsNullOrEmpty(element.Mesh.Guid))
                            element.Mesh = manager.LoadByGuid<Mesh>(element.Mesh.Guid);
                        else
                            element.Mesh ??= staticMesh.Mesh;

                        if (element.Material != null && !string.IsNullOrEmpty(element.Material.Guid))
                            element.Material = manager.LoadByGuid<Material>(element.Material.Guid);
                    }
                }

                Debug.Log($"[StaticMeshLoader] MeshParts resolved for '{name}'");
                Console.Out.Flush();

                // Resolve LOD references
                if (staticMesh.LODs != null)
                {
                    foreach (var lod in staticMesh.LODs)
                    {
                        if (lod.Mesh != null && !string.IsNullOrEmpty(lod.Mesh.Guid))
                            lod.Mesh = manager.LoadByGuid<Mesh>(lod.Mesh.Guid);
                        else
                            lod.Mesh ??= staticMesh.Mesh;

                        if (lod.MeshParts != null)
                        {
                            foreach (var element in lod.MeshParts)
                            {
                                if (element.Mesh != null && !string.IsNullOrEmpty(element.Mesh.Guid))
                                    element.Mesh = manager.LoadByGuid<Mesh>(element.Mesh.Guid);
                                else
                                    element.Mesh ??= lod.Mesh;

                                if (element.Material != null && !string.IsNullOrEmpty(element.Material.Guid))
                                    element.Material = manager.LoadByGuid<Material>(element.Material.Guid);
                            }
                        }
                    }
                }

                Debug.Log($"[StaticMeshLoader] LODs resolved for '{name}'");
                Console.Out.Flush();

                // Register mesh parts for GPU draw
                staticMesh.Mesh?.RegisterMeshParts();

                Console.Out.Flush();

                // TODO: re-enable once crash is resolved
                // staticMesh.CookPhysicsMesh();

                return staticMesh;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("StaticMeshLoader", $"FAILED to load '{name}': {ex}");
                Console.Out.Flush();
                return null;
            }
        }
    }
}
