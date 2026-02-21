using System;
using System.IO;
using Freefall.Graphics;
using static Freefall.Assets.InternalAssets;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports a mesh file (.fbx, .dae, .obj) into a fully configured StaticMesh:
    /// GPU buffers, textures (by convention), material, LODs, and pre-cooked physics.
    /// Thread-safe — designed to be called from background threads via Assets.LoadAsync.
    /// </summary>
    public class StaticMeshImporter
    {
        public StaticMesh Load(string filepath)
        {
            var meshName = Path.GetFileNameWithoutExtension(filepath);

            // ── 1. CPU: Parse FBX → MeshData ──
            var importer = new MeshImporter();
            var meshData = importer.ParseRaw(filepath);

            // ── 2. GPU: Create mesh buffers + SRVs (thread-safe) ──
            var mesh = Mesh.CreateAsync(Engine.Device, meshData);
            mesh.Name = meshName;
            mesh.RegisterMeshParts();

            // ── 3. Discover + load textures by convention ──
            var material = DiscoverMaterial(filepath);

            // ── 4. Build StaticMesh with LODs ──
            var staticMesh = new StaticMesh { Name = meshName, Mesh = mesh, LODGroup = LODGroups.LargeProps };
            BuildMeshParts(staticMesh, mesh, material);

            // ── 5. Cook physics mesh ──
            staticMesh.CookPhysicsMesh();

            return staticMesh;
        }

        /// <summary>
        /// Discovers textures in {mesh_dir}/Textures/ using suffix convention (_Dif, _Nor, _Spec, _Aoc)
        /// and builds a Material. Returns DefaultMaterial if no textures found.
        /// </summary>
        private static Material DiscoverMaterial(string meshPath)
        {
            var texturePath = Path.GetDirectoryName(meshPath) + @"\Textures\";
            if (!Directory.Exists(texturePath))
                return DefaultMaterial;

            CpuTextureData albedo = null, normal = null, spec = null, aoc = null;

            foreach (var file in Directory.EnumerateFiles(texturePath))
            {
                if (file.EndsWith(".psd", StringComparison.OrdinalIgnoreCase)) continue;
                var name = Path.GetFileNameWithoutExtension(file);

                try
                {
                    var loadPath = TextureLibrary.ResolvePackedDDS(file) ?? file;

                    if (name.EndsWith("_Dif") || name.EndsWith("_diffuse") || name.EndsWith("_Diffuse"))
                        albedo = Texture.ParseFromFile(Engine.Device, loadPath);
                    else if (name.EndsWith("_Nor") || name.EndsWith("_normal") || name.EndsWith("_Normal"))
                        normal = Texture.ParseFromFile(Engine.Device, loadPath);
                    else if (name.EndsWith("_Spec"))
                        spec = Texture.ParseFromFile(Engine.Device, loadPath);
                    else if (name.EndsWith("_Aoc"))
                        aoc = Texture.ParseFromFile(Engine.Device, loadPath);
                }
                catch { /* skip bad textures */ }
            }

            if (albedo == null && normal == null && spec == null && aoc == null)
                return DefaultMaterial;

            // Create GPU textures + enqueue uploads via StreamingManager
            Texture albedoTex = albedo != null ? Texture.CreateAsync(Engine.Device, albedo) : null;
            Texture normalTex = normal != null ? Texture.CreateAsync(Engine.Device, normal) : null;
            Texture specTex   = spec   != null ? Texture.CreateAsync(Engine.Device, spec)   : null;
            Texture aocTex    = aoc    != null ? Texture.CreateAsync(Engine.Device, aoc)    : null;

            var material = new Material(DefaultEffect);
            material.SetTexture("AlbedoTex", albedoTex ?? White);
            material.SetTexture("NormalTex", normalTex ?? FlatNormal);
            if (specTex != null) material.SetTexture("Roughness", specTex);
            if (aocTex  != null) material.SetTexture("AO", aocTex);

            return material;
        }

        /// <summary>
        /// Splits mesh parts into base MeshParts and LOD levels based on _LOD_N suffix.
        /// </summary>
        private static void BuildMeshParts(StaticMesh staticMesh, Mesh mesh, Material material)
        {
            for (int p = 0; p < mesh.MeshParts.Count; p++)
            {
                var part = mesh.MeshParts[p];
                int lodIndex = part.Name.IndexOf("_LOD_");

                if (lodIndex < 0)
                {
                    // Base mesh part (no LOD suffix)
                    staticMesh.MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                }
                else
                {
                    // Parse LOD level from suffix
                    int endIndex = lodIndex + 6 < part.Name.Length ? 2 : 1;
                    int lvl = Convert.ToInt32(part.Name.Substring(lodIndex + 5, endIndex)) - 1;

                    if (lvl < 0)
                    {
                        // _LOD_0 → base mesh
                        staticMesh.MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                    }
                    else
                    {
                        while (staticMesh.LODs.Count < lvl + 1)
                            staticMesh.LODs.Add(new StaticMeshLOD { Mesh = mesh });
                        staticMesh.LODs[lvl].MeshParts.Add(new MeshElement { Mesh = mesh, Material = material, MeshPartIndex = p });
                    }
                }
            }
        }
    }
}
