using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using Freefall.Assets.Packers;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports .staticmesh files (YAML-serialized StaticMesh definitions).
    /// Produces the StaticMesh definition artifact plus a hidden CollisionMeshData
    /// subasset containing a pre-cooked PhysX TriangleMesh.
    ///
    /// Collision geometry is selected by MeshElement.Collision flags in the YAML.
    /// Falls back to lowest LOD meshparts, then all meshparts if none are flagged.
    /// </summary>
    [AssetImporter(".staticmesh")]
    public class StaticMeshImporter : IImporter
    {
        /// <summary>
        /// .staticmesh files ARE the asset definition — inspect the loaded StaticMesh, not the importer.
        /// </summary>
        public object GetInspectionTarget(MetaFile meta)
        {
            return Engine.Assets.LoadByGuid<StaticMesh>(meta.Guid);
        }

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var yaml = File.ReadAllText(filepath, Encoding.UTF8);

            // ── 1. Main artifact: StaticMesh definition (YAML) ──
            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = "StaticMesh",
                Data = new AssetDefinitionData
                {
                    TypeName = "StaticMesh",
                    YamlBytes = Encoding.UTF8.GetBytes(yaml)
                }
            });

            // ── 2. Cook PhysX collision mesh ──
            try
            {
                var cookedBytes = CookCollisionMesh(yaml, name);
                if (cookedBytes != null)
                {
                    result.Artifacts.Add(new ImportArtifact
                    {
                        Name = name,
                        Type = nameof(CollisionMeshData),
                        Data = new CollisionMeshData { CookedBytes = cookedBytes },
                        Hidden = true
                    });
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning("StaticMeshImporter",
                    $"Failed to cook collision mesh for '{name}': {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Cook a PhysX TriangleMesh from the referenced mesh geometry,
        /// using collision-flagged meshparts (or lowest LOD / all as fallback).
        /// </summary>
        private static byte[] CookCollisionMesh(string yaml, string name)
        {
            // Extract Mesh GUID from YAML
            var meshGuid = ExtractGuid(yaml, "Mesh");
            if (meshGuid == null)
            {
                Debug.Log($"[StaticMeshImporter] No Mesh GUID found for '{name}', skipping collision cook");
                return null;
            }

            // Read the cached MeshData — must specifically request MeshData type,
            // otherwise ResolveCachePathByGuid returns AssetDefinitionData (YAML, not binary)
            var cachePath = AssetDatabase.ResolveCachePathByGuid(meshGuid, typeof(MeshData));
            if (cachePath == null || !File.Exists(cachePath))
            {
                Debug.Log($"[StaticMeshImporter] MeshData cache not found for GUID '{meshGuid}', skipping collision cook");
                return null;
            }

            MeshData meshData;
            var packer = new MeshPacker();
            using (var stream = File.OpenRead(cachePath))
                meshData = packer.Read(stream);

            Debug.Log($"[StaticMeshImporter] '{name}' meshGuid={meshGuid} cachePath={cachePath} " +
                $"verts={meshData.Positions?.Length ?? 0} idx={meshData.Indices?.Length ?? 0}" +
                (meshData.Positions?.Length > 0 ? $" v[0]={meshData.Positions[0]}" : ""));

            if (meshData.Positions == null || meshData.Indices == null || meshData.Positions.Length == 0)
            {
                Debug.Log($"[StaticMeshImporter] No geometry in cached MeshData for '{name}'");
                return null;
            }

            // Select collision indices
            var collisionIndices = GetCollisionIndices(yaml, meshData);
            if (collisionIndices == null || collisionIndices.Length == 0)
            {
                Debug.Log($"[StaticMeshImporter] No collision indices for '{name}'");
                return null;
            }

            // Cook PhysX TriangleMesh
            var cooking = PhysicsWorld.Physics.CreateCooking();
            var desc = new TriangleMeshDesc()
            {
                Flags = (MeshFlag)0,
                Triangles = collisionIndices,
                Points = meshData.Positions
            };

            var stream2 = new MemoryStream();
            cooking.CookTriangleMesh(desc, stream2);

            Debug.Log($"[StaticMeshImporter] Cooked collision mesh for '{name}': " +
                      $"{meshData.Positions.Length} verts, {collisionIndices.Length / 3} tris, {stream2.Length} bytes");

            return stream2.ToArray();
        }

        /// <summary>
        /// Get collision indices — all mesh triangles.
        /// PhysX builds an internal BVH, no need for part/LOD selection.
        /// </summary>
        private static int[] GetCollisionIndices(string yaml, MeshData meshData)
        {
            return Array.ConvertAll(meshData.Indices, i => (int)i);
        }

        private static string ExtractGuid(string yaml, string key)
        {
            var match = Regex.Match(yaml, $@"{key}:\s*([0-9a-fA-F]{{32}})", RegexOptions.Multiline);
            return match.Success ? match.Groups[1].Value : null;
        }
    }
}
