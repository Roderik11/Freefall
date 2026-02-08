using System.IO;
using Freefall.Graphics;
using Vortice.DXGI;

namespace Freefall.Assets
{
    /// <summary>
    /// Engine-internal default textures and materials.
    /// Loads real DDS fallback textures so every MaterialData slot
    /// points at something valid. Mirrors Spark.InternalAssets.
    /// </summary>
    public static class InternalAssets
    {
        private static readonly string ResourcesPath = @"D:\Projects\2026\Freefall\Resources";

        // --- Utility textures (procedural) ---
        public static Texture White { get; private set; }
        public static Texture FlatNormal { get; private set; }

        // --- Default textures (loaded from DDS) ---
        public static Texture DefaultDiffuse { get; private set; }
        public static Texture DefaultNormal { get; private set; }
        public static Texture DefaultSpecular { get; private set; }

        // --- Default effect + material ---
        public static Effect DefaultEffect { get; private set; }
        public static Material DefaultMaterial { get; private set; }

        public static void Initialize(GraphicsDevice device)
        {
            // --- Procedural utility textures (4x4, R8G8B8A8_UNorm) ---

            // Solid white
            byte[] white = new byte[4 * 4 * 4];
            for (int i = 0; i < white.Length; i++) white[i] = 255;
            White = Texture.CreateFromData(device, 4, 4, white, Format.R8G8B8A8_UNorm);

            // Flat normal map (tangent-space up: 128,128,255)
            byte[] normal = new byte[4 * 4 * 4];
            for (int i = 0; i < normal.Length; i += 4)
            {
                normal[i]     = 128;
                normal[i + 1] = 128;
                normal[i + 2] = 255;
                normal[i + 3] = 255;
            }
            FlatNormal = Texture.CreateFromData(device, 4, 4, normal, Format.R8G8B8A8_UNorm);

            // --- Default textures from DDS ---
            DefaultDiffuse  = new Texture(device, Path.Combine(ResourcesPath, "default_albedo_map.dds"));
            DefaultNormal   = new Texture(device, Path.Combine(ResourcesPath, "default_normal_map.dds"));
            DefaultSpecular = new Texture(device, Path.Combine(ResourcesPath, "default_specular_map.dds"));

            // --- Default effect + material ---
            DefaultEffect = new Effect("gbuffer");
            DefaultMaterial = new Material(DefaultEffect);
            DefaultMaterial.SetTexture("AlbedoTex", DefaultDiffuse);
            DefaultMaterial.SetTexture("NormalTex", DefaultNormal);
        }
    }
}
