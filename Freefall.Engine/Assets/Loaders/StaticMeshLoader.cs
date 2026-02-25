using System;
using System.IO;
using System.Linq;
using System.Text;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads StaticMesh assets from cache (.asset files).
    /// Unpacks AssetDefinitionData (YAML), deserializes the StaticMesh definition,
    /// then resolves all GUID references (Mesh, Material, Texture) via AssetManager.
    /// Loads pre-cooked PhysX TriangleMesh if available.
    /// </summary>
    [AssetLoader(typeof(StaticMesh))]
    public class StaticMeshLoader : IAssetLoader
    {
        private readonly AssetDefinitionPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AssetDefinitionData");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for static mesh '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            try
            {
                Debug.Log($"[StaticMeshLoader] Loading '{name}' from {cachePath}");

                AssetDefinitionData defData;
                using (var stream = File.OpenRead(cachePath))
                    defData = _packer.Read(stream);

                var yaml = Encoding.UTF8.GetString(defData.YamlBytes);
                StaticMesh staticMesh;
                try
                {
                    staticMesh = NativeImporter.LoadFromString(yaml) as StaticMesh;
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"Failed to deserialize StaticMesh from cache: {name} - {ex.GetType().Name}: {ex.Message}", ex);
                }

                if (staticMesh == null)
                {
                    var preview = yaml.Length > 200 ? yaml[..200] + "..." : yaml;
                    throw new InvalidDataException(
                        $"Failed to deserialize StaticMesh from cache: {name}. " +
                        $"TypeName in cache: '{defData.TypeName}'. YAML preview: {preview}");
                }

                staticMesh.Name = name;

                // -- Resolve GUID references --

                if (staticMesh.Mesh != null && !string.IsNullOrEmpty(staticMesh.Mesh.Guid))
                    staticMesh.Mesh = manager.LoadByGuid<Mesh>(staticMesh.Mesh.Guid);

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

                staticMesh.Mesh?.RegisterMeshParts();

                // -- Load pre-cooked PhysX TriangleMesh --
                LoadCookedCollisionMesh(staticMesh, sourceGuid, name);

                return staticMesh;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("StaticMeshLoader", $"FAILED to load '{name}': {ex}");
                return null;
            }
        }

        /// <summary>
        /// Load CollisionMeshData subasset directly via source GUID -> meta -> subasset GUID.
        /// No name-based resolution needed.
        /// </summary>
        private static void LoadCookedCollisionMesh(StaticMesh staticMesh, string sourceGuid, string name)
        {
            try
            {
                if (string.IsNullOrEmpty(sourceGuid))
                    return;

                var meta = AssetDatabase.GetMeta(sourceGuid);
                if (meta == null)
                    return;

                var collisionSub = meta.SubAssets.FirstOrDefault(
                    s => s.Type == nameof(CollisionMeshData));
                if (collisionSub == null)
                    return;

                var physxPath = AssetDatabase.ResolveCachePathByGuid(collisionSub.Guid);
                if (physxPath == null || !File.Exists(physxPath))
                    return;

                var packer = new CollisionMeshPacker();
                using var stream = File.OpenRead(physxPath);
                var cooked = packer.Read(stream);

                staticMesh.CookedTriMesh = PhysicsWorld.Physics.CreateTriangleMesh(
                    new MemoryStream(cooked.CookedBytes));
            }
            catch (Exception ex)
            {
                Debug.LogWarning("StaticMeshLoader",
                    $"Failed to load cooked collision mesh for '{name}': {ex.Message}");
            }
        }
    }
}
