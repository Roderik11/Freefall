using System;
using System.IO;
using Freefall.Graphics;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports model files as StaticMesh with auto-created MeshElements.
    /// Searches for textures in the same directory or textures subfolder.
    /// </summary>
    [AssetReader(".fbx", ".dae")]
    public class StaticMeshImporter : AssetReader<StaticMesh>
    {
        public override StaticMesh Import(string filepath)
        {
            var mesh = Engine.Assets.Load<Mesh>(filepath);
            var staticMesh = new StaticMesh { Mesh = mesh, LODGroup = LODGroups.LargeProps };
            
            // Try to find diffuse texture
            var dir = Path.GetDirectoryName(filepath);
            var baseName = Path.GetFileNameWithoutExtension(filepath);
            var parentDir = new DirectoryInfo(dir).Name; // e.g. "Paladin" or "Oak_Trees"
            
            // Common texture naming patterns - expanded to cover real naming conventions
            var searchPatterns = new[]
            {
                // Based on mesh name
                Path.Combine(dir, "textures", $"{baseName}_diffuse.png"),
                Path.Combine(dir, "textures", $"{baseName}_Dif.png"),
                Path.Combine(dir, "Textures", $"{baseName}_diffuse.png"),
                Path.Combine(dir, "Textures", $"{baseName}_Dif.png"),
                // Based on parent directory name (e.g. "Paladin" -> "Paladin_diffuse.png")
                Path.Combine(dir, "textures", $"{parentDir}_diffuse.png"),
                Path.Combine(dir, "textures", $"{parentDir}_Dif.png"),
                Path.Combine(dir, "Textures", $"{parentDir}_diffuse.png"),
                Path.Combine(dir, "Textures", $"{parentDir}_Dif.png"),
                // Oak trees have "Oak_Dif.png" in Textures folder
                Path.Combine(dir, "Textures", "Oak_Dif.png"),
            };

            Texture diffuseTexture = null;
            foreach (var pattern in searchPatterns)
            {
                if (File.Exists(pattern))
                {
                    diffuseTexture = Engine.Assets.Load<Texture>(pattern);
                    Debug.Log("StaticMeshImporter", $"Found texture: {pattern}");
                    break;
                }
            }

            if (diffuseTexture == null)
            {
                Debug.Log("StaticMeshImporter", $"No texture found for {baseName}, using white");
            }

            // Create material — use InternalAssets defaults, per-mesh only if texture found
            Material material;
            if (diffuseTexture != null)
            {
                material = new Material(InternalAssets.DefaultEffect);
                material.SetTexture("AlbedoTex", diffuseTexture);
                material.SetTexture("NormalTex", InternalAssets.FlatNormal);
            }
            else
            {
                material = InternalAssets.DefaultMaterial;
            }

            // Classify mesh parts into base or LOD levels
            // _LOD_0 / _LOD_00 = base mesh (highest quality)
            // _LOD_01+ = lower quality LOD levels
            for (int i = 0; i < mesh.MeshParts.Count; i++)
            {
                var part = mesh.MeshParts[i];
                var name = part.Name;

                int index = name.IndexOf("_LOD_");
                if (index < 0)
                {
                    // Base mesh part
                    staticMesh.MeshParts.Add(new MeshElement
                    {
                        Mesh = mesh,
                        Material = material,
                        MeshPartIndex = i
                    });
                }
                else
                {
                    try
                    {
                        int endIndex = index + 6 < name.Length ? 2 : 1;
                        int lvl = Convert.ToInt32(name.Substring(index + 5, endIndex)) - 1;

                        if (lvl < 0)
                        {
                            // _LOD_0 / _LOD_00 → treat as base mesh part
                            staticMesh.MeshParts.Add(new MeshElement
                            {
                                Mesh = mesh,
                                Material = material,
                                MeshPartIndex = i
                            });
                        }
                        else
                        {
                            while (staticMesh.LODs.Count < lvl + 1)
                                staticMesh.LODs.Add(new StaticMeshLOD { Mesh = mesh });

                            staticMesh.LODs[lvl].MeshParts.Add(new MeshElement
                            {
                                Mesh = mesh,
                                Material = material,
                                MeshPartIndex = i
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.Log($"[StaticMeshImporter] Failed to parse LOD level from {name}: {ex.Message}");
                    }
                }
            }

            if (staticMesh.LODs.Count > 0)
            {
                Debug.Log($"[StaticMeshImporter] Loaded {baseName} with {staticMesh.LODs.Count} LOD levels");
            }

            return staticMesh;
        }
    }
}
