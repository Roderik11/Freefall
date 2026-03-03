using System.IO;
using Freefall.Graphics;
using Vortice.DXGI;

namespace Freefall.Assets
{
    /// <summary>
    /// Engine-internal default textures and materials.
    /// Loads real DDS fallback textures so every MaterialData slot
    /// points at something valid. Mirrors Spark.InternalAssets.
    ///
    /// All internal assets have deterministic stable GUIDs so .asset files
    /// can reference them (e.g. a Material .asset referencing DefaultEffect).
    /// </summary>
    public static class InternalAssets
    {
        private static readonly string ResourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources");

        // ── Stable GUIDs (hardcoded, never change) ──
        public static class Guids
        {
            public const string White           = "00000000000000000000000000000001";
            public const string FlatNormal      = "00000000000000000000000000000002";
            public const string DefaultDiffuse  = "00000000000000000000000000000003";
            public const string DefaultNormal   = "00000000000000000000000000000004";
            public const string DefaultSpecular = "00000000000000000000000000000005";
            public const string Black           = "00000000000000000000000000000006";

            public const string WhiteArray      = "00000000000000000000000000000007";
            public const string FlatNormalArray = "00000000000000000000000000000008";
            public const string BlackArray      = "00000000000000000000000000000009";
            public const string DefaultEffect   = "00000000000000000000000000000010";
            public const string TerrainEffect   = "00000000000000000000000000000011";
            public const string DecoratorEffect  = "00000000000000000000000000000012";
            public const string FoliageEffect    = "00000000000000000000000000000013";
            public const string TrunkEffect      = "00000000000000000000000000000014";
            public const string DefaultMaterial = "00000000000000000000000000000020";
        }

        // --- Utility textures (procedural) ---
        public static Texture Black { get; private set; }
        public static Texture White { get; private set; }
        public static Texture FlatNormal { get; private set; }

        public static Texture BlackArray { get; private set; }
        public static Texture WhiteArray { get; private set; }
        public static Texture FlatNormalArray { get; private set; }

        // --- Default textures (loaded from DDS) ---
        public static Texture DefaultDiffuse { get; private set; }
        public static Texture DefaultNormal { get; private set; }
        public static Texture DefaultSpecular { get; private set; }

        // --- Default effect + material ---
        public static Effect DefaultEffect { get; private set; }
        public static Effect TerrainEffect { get; private set; }
        public static Effect DecoratorEffect { get; private set; }
        public static Effect FoliageEffect { get; private set; }
        public static Effect TrunkEffect { get; private set; }
        public static Material DefaultMaterial { get; private set; }
        public static Material DecoratorMaterial { get; private set; }

        public static void Initialize(GraphicsDevice device)
        {
            // --- Procedural utility textures (4x4, R8G8B8A8_UNorm) ---

            // Solid white
            byte[] white = new byte[4 * 4 * 4];
            for (int i = 0; i < white.Length; i++) white[i] = 255;
            White = Texture.CreateFromData(device, 4, 4, white, Format.R8G8B8A8_UNorm);
            White.Name = "White";

            // Solid black
            byte[] black = new byte[4 * 4 * 4];
            for (int i = 0; i < black.Length; i += 4)
            {
                black[i] = 0; black[i + 1] = 0; black[i + 2] = 0; black[i + 3] = 255;
            }
            Black = Texture.CreateFromData(device, 4, 4, black, Format.R8G8B8A8_UNorm);
            Black.Name = "Black";

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
            FlatNormal.Name = "FlatNormal";

            // Arrays
            WhiteArray = Texture.CreateTexture2DArray(device, new[] { White });
            WhiteArray.Name = "WhiteArray";
            BlackArray = Texture.CreateTexture2DArray(device, new[] { Black });
            BlackArray.Name = "BlackArray";
            FlatNormalArray = Texture.CreateTexture2DArray(device, new[] { FlatNormal });
            FlatNormalArray.Name = "FlatNormalArray";

            // --- Default textures from DDS ---
            DefaultDiffuse  = new Texture(device, Path.Combine(ResourcesPath, "default_albedo_map.dds"));
            DefaultDiffuse.Name = "DefaultDiffuse";
            DefaultNormal   = new Texture(device, Path.Combine(ResourcesPath, "default_normal_map.dds"));
            DefaultNormal.Name = "DefaultNormal";
            DefaultSpecular = new Texture(device, Path.Combine(ResourcesPath, "default_specular_map.dds"));
            DefaultSpecular.Name = "DefaultSpecular";

            // --- Default effect + material ---
            DefaultEffect = new Effect("gbuffer");
            TerrainEffect = new Effect("gputerrain");
            DecoratorEffect = new Effect("grass");
            FoliageEffect = new Effect("gbuffer_foliage");
            TrunkEffect = new Effect("gbuffer_trunk");
            DefaultMaterial = new Material(DefaultEffect);
            DefaultMaterial.Name = "DefaultMaterial";
            DefaultMaterial.SetTexture("AlbedoTex", DefaultDiffuse);
            DefaultMaterial.SetTexture("NormalTex", DefaultNormal);
            DecoratorMaterial = new Material(DecoratorEffect);
            DecoratorMaterial.Name = "DecoratorMaterial";
        }

        /// <summary>
        /// Register all internal assets with stable GUIDs into the AssetManager.
        /// Call after Initialize() and after AssetManager is created.
        /// Makes internal assets discoverable via LoadByGuid&lt;T&gt;().
        /// </summary>
        public static void Register(AssetManager manager)
        {
            manager.RegisterAsset(Guids.White, White);
            manager.RegisterAsset(Guids.Black, Black);
            manager.RegisterAsset(Guids.FlatNormal, FlatNormal);
            manager.RegisterAsset(Guids.WhiteArray, WhiteArray);
            manager.RegisterAsset(Guids.BlackArray, BlackArray);
            manager.RegisterAsset(Guids.FlatNormalArray, FlatNormalArray);

            manager.RegisterAsset(Guids.DefaultDiffuse, DefaultDiffuse);
            manager.RegisterAsset(Guids.DefaultNormal, DefaultNormal);
            manager.RegisterAsset(Guids.DefaultSpecular, DefaultSpecular);
            manager.RegisterAsset(Guids.DefaultEffect, DefaultEffect);
            manager.RegisterAsset(Guids.TerrainEffect, TerrainEffect);
            manager.RegisterAsset(Guids.DecoratorEffect, DecoratorEffect);
            manager.RegisterAsset(Guids.FoliageEffect, FoliageEffect);
            manager.RegisterAsset(Guids.TrunkEffect, TrunkEffect);
            manager.RegisterAsset(Guids.DefaultMaterial, DefaultMaterial);
        }
    }
}
