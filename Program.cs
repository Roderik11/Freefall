using System;
using System.Numerics;
using System.Collections.Generic;
using Freefall;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Assets;
using Freefall.Scripts;
using Vortice.Mathematics;

namespace Freefall
{
    class Program
    {
        [STAThread]
        public static void Main()
        {
            Engine.Initialize("Freefall (Deferred)", 1280, 720);
            
            // Add Test Objects
            CreateTestScene(); 

            Engine.Run();
        }
        
        static void CreateTestScene()
        {
             Debug.Log("CreateTestScene Started");

             // Initialize GPU frustum culler
             CommandBuffer.InitializeCuller(Engine.Device);

             // Create Directional Light (Sun)
             var lightEntity = new Entity("Sun");
             lightEntity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(0, MathHelper.PiOver4, 0);
             var light = lightEntity.AddComponent<DirectionalLight>();
             light.Color = new Color3(1, 1, 1);
             light.Intensity = 1.0f;


             // Skybox
             var skyboxEntity = new Entity("Skybox");
             var skybox = skyboxEntity.AddComponent<SkyboxRenderer>();
             skybox.SunLight = light;
             skybox.TimeOfDay = 15f; // Afternoon
             skybox.AnimateTimeOfDay = false;

             // Player spawn position (beach spawn from GameTestScene)
             Vector3 playerSpawn = new Vector3(1167, 0, 830);
             // castle spawn from GameTestScene
             playerSpawn = new Vector3(896, 164, 920);

             // === TERRAIN — toggle here ===
             IHeightProvider heightProvider = SpawnGPUTerrain();
             //IHeightProvider heightProvider = SpawnTerrain();
             
             // Ensure all terrain textures are uploaded before first render
             StreamingManager.Instance?.Flush();

             // ===== PLAYER SETUP =====
             // Load Paladin mesh (target mesh for animations)
             var paladinMesh = Engine.Assets.Load<Mesh>(@"D:\Projects\2024\ProjectXYZ\Resources\Characters\Paladin\paladin_j_nordstrom.dae");
             // Set root rotation to face correct direction (model is backwards)
             paladinMesh.RootRotation = Matrix4x4.CreateRotationY(MathF.PI);
             Debug.Log($"[Player] Loaded Paladin mesh with {paladinMesh.MeshParts.Count} parts, {paladinMesh.Bones?.Length ?? 0} bones");

             // Load Knight mesh (source mesh for animations - animations are made for Knight)
             var knightMesh = Engine.Assets.Load<Mesh>(@"D:\Projects\2024\ProjectXYZ\Resources\Characters\Knight\knight_d_pelegrini.dae");
             Debug.Log($"[Player] Loaded Knight mesh with {knightMesh.MeshParts.Count} parts, {knightMesh.Bones?.Length ?? 0} bones");

             // Retarget animations from Knight skeleton to Paladin skeleton
             paladinMesh.BindPoseDifference(knightMesh);

             // Set player spawn height from terrain
             playerSpawn.Y = heightProvider.GetHeight(playerSpawn) + 0.1f;

             // Create Player Entity
             var playerEntity = new Entity("Player");
             playerEntity.Transform.Position = playerSpawn;
             playerEntity.Transform.Scale = Vector3.One; // Importer now handles cm to m conversion
             
             // Player mesh - Paladin model with SkinnedMeshRenderer
             var playerRenderer = playerEntity.AddComponent<SkinnedMeshRenderer>();
             playerRenderer.Mesh = paladinMesh;
             playerRenderer.Enabled = true;  // DEBUG: Disabled to isolate tree flickering

             // Load player texture
             var paladinTexture = Engine.Assets.Load<Texture>(@"D:\Projects\2024\ProjectXYZ\Resources\Characters\Paladin\textures\Paladin_diffuse.png");
             var paladinMat = new Material(new Effect("gbuffer_skinned")); // Use skinned shader for animated characters
             paladinMat.SetTexture("Albedo", paladinTexture);
             playerRenderer.Materials.Add(paladinMat);
             
             // Setup animations (creates Animator component with full state machine)
             AnimationSetup.CreateBlendTreeAnimations(playerEntity, paladinMesh);
             
             // Player components
             playerEntity.AddComponent<Player>();
             var characterController = playerEntity.AddComponent<CharacterController>();
             characterController.Terrain = heightProvider;
             characterController.Height = 1.8f;

             //SpawnLights(playerSpawn, heightProvider);
             //SpawnTrees(heightProvider, playerSpawn, 16f, 5);
             //SpawnCharacters(10, heightProvider, playerSpawn, paladinMesh, paladinTexture, paladinMat);
             
             // ===== CAMERA SETUP =====
             var cameraEntity = new Entity("Camera");
             var camera = cameraEntity.AddComponent<Camera>();
             camera.FarPlane = 2048 * 8; // Match Apex
             
             // Third person camera following player
             var thirdPersonCamera = cameraEntity.AddComponent<ThirdPersonCamera>();
             thirdPersonCamera.Target = playerEntity;
             thirdPersonCamera.Terrain = heightProvider;
             thirdPersonCamera.Offset = Vector3.UnitY * 1.85f;

             Camera.Main = camera;
             
             Input.IsMouseLocked = true;
             
             Debug.Log($"[Player] Created at {playerSpawn}");

             // ===== SCENE LOADING =====
             LoadScene(); // Disabled — per-resource copy path causes TDR during parallel loading. Re-enable after fixing.
         }

         private static IHeightProvider SpawnTerrain()
         {
             string assetsPath = @"D:\Projects\2024\ProjectXYZ\Resources\";
             
             var terrainLayers = new List<Terrain.TextureLayer>
             {
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_01/Sand_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_01/Sand_01_Nor.png"),
                     Tiling = new Vector2(8, 8),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(7, 7),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_Dirt/Sand_Dirt.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_Dirt/Sand_Dirt_Nor.png"),
                     Tiling = new Vector2(20, 30),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sandstone/Sandstone_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sandstone/Sandstone_Nor.png"),
                     Tiling = new Vector2(10, 10),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Gras_01/Gras_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Gras_01/Gras_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_02/Sand_02_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_02/Sand_02_Nor.png"),
                     Tiling = new Vector2(10, 10),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Leafs/Leafs_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Leafs/Leafs_01_Nor.png"),
                     Tiling = new Vector2(3, 3),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(20, 15),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_02_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Nor.png"),
                     Tiling = new Vector2(15, 15),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Raw_Dirt/Raw_Dirt_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Raw_Dirt/Raw_Dirt_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Moos/Moss_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Moos/Moss_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(200, 200),
                 },
             };

             var heightmap = new Texture(Engine.Device, assetsPath + "terrain.dds");
             var heightField = Texture.ReadHeightField(assetsPath + "terrain.dds");
             var terrainMaterial = new Material(new Effect("terrain"));

             var terrainEntity = new Entity("Terrain");
             var terrain = terrainEntity.AddComponent<Terrain>();
             terrain.Material = terrainMaterial;
             terrain.TerrainSize = new Vector2(1700, 1700);
             terrain.Heightmap = heightmap;
             terrain.HeightField = heightField;
             terrain.MaxHeight = 600;
             terrain.Layers = terrainLayers;
             
             // Set terrain position to align with scene coordinates
             terrainEntity.Transform.Position = new Vector3(842.0983f - 842.0983f, 85.85109f - 39.8f, 841.4021f - 839.8109f);
             
             Debug.Log("[Terrain] Entity created");
             return terrain;
         }

         private static IHeightProvider SpawnGPUTerrain()
         {
             string assetsPath = @"D:\Projects\2024\ProjectXYZ\Resources\";
             
             var terrainLayers = new List<Terrain.TextureLayer>
             {
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_01/Sand_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_01/Sand_01_Nor.png"),
                     Tiling = new Vector2(8, 8),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(7, 7),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_Dirt/Sand_Dirt.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_Dirt/Sand_Dirt_Nor.png"),
                     Tiling = new Vector2(20, 30),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sandstone/Sandstone_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sandstone/Sandstone_Nor.png"),
                     Tiling = new Vector2(10, 10),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Gras_01/Gras_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Gras_01/Gras_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_02/Sand_02_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Sand_02/Sand_02_Nor.png"),
                     Tiling = new Vector2(10, 10),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Leafs/Leafs_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Leafs/Leafs_01_Nor.png"),
                     Tiling = new Vector2(3, 3),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(20, 15),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_02_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Dirt/Dirt_01_Nor.png"),
                     Tiling = new Vector2(15, 15),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Raw_Dirt/Raw_Dirt_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Raw_Dirt/Raw_Dirt_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Moos/Moss_01_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/Moos/Moss_01_Nor.png"),
                     Tiling = new Vector2(4, 4),
                 },
                 new Terrain.TextureLayer
                 {
                     Diffuse = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/RockWall_Dif.png"),
                     Normals = new Texture(Engine.Device, assetsPath + "Terrain/Terrain Textures/RockWall_01/Rock_01_Nor.png"),
                     Tiling = new Vector2(200, 200),
                 },
             };

             var heightmap = new Texture(Engine.Device, assetsPath + "terrain.dds");
             var heightField = Texture.ReadHeightField(assetsPath + "terrain.dds");
             var terrainMaterial = new Material(new Effect("gputerrain"));

             var terrainEntity = new Entity("GPUTerrain");
             var gpuTerrain = terrainEntity.AddComponent<GPUTerrain>();
             gpuTerrain.Material = terrainMaterial;
             gpuTerrain.TerrainSize = new Vector2(1700, 1700);
             gpuTerrain.Heightmap = heightmap;
             gpuTerrain.HeightField = heightField;
             gpuTerrain.MaxHeight = 600;
             gpuTerrain.Layers = terrainLayers;
             gpuTerrain.DetailBalance = 2.0f;
             gpuTerrain.MaxDepth = 7;

             // Load splatmaps (same paths as Terrain.cs)
             var splatPaths = new[] {
                 "Resources/Terrain/Terrain_splatmap_0.dds",
                 "Resources/Terrain/Terrain_splatmap_1.dds",
                 "Resources/Terrain/Terrain_splatmap_2.dds",
                 "Resources/Terrain/Terrain_splatmap_3.dds"
             };
             for (int i = 0; i < splatPaths.Length && i < gpuTerrain.ControlMaps.Length; i++)
             {
                 if (System.IO.File.Exists(splatPaths[i]))
                     gpuTerrain.ControlMaps[i] = new Texture(Engine.Device, splatPaths[i]);
                 else
                     gpuTerrain.ControlMaps[i] = Texture.CreateFromData(Engine.Device, 1, 1, new byte[] {0,0,0,0});
             }
             
             terrainEntity.Transform.Position = new Vector3(842.0983f - 842.0983f, 85.85109f - 39.8f, 841.4021f - 839.8109f);
             
             Debug.Log("[GPUTerrain] Entity created");
             return gpuTerrain;
         }


         private static void SpawnCharacters(int count, IHeightProvider terrain, Vector3 center, Mesh mesh, Texture texture, Material material = null)
         {
            // ===== SKINNED MESHES TEST =====
             // Spawn characters around the player (Render only, no Controller)
             // Create Shared Material for Minions to enable Batching
             var minionMaterial = new Material(new Effect("gbuffer_skinned")); // Use skinned shader for animated characters
             minionMaterial.SetTexture("Albedo", texture);

             Random npcRandom = new Random(42);
             int minionCount = count; 
             for (int i = 0; i < minionCount; i++)
             {
                 var npc = new Entity($"NPC_{i}");
                 // Random scatter around player spawn
                 float offsetX = (float)(npcRandom.NextDouble() * 200 - 100);
                 float offsetZ = (float)(npcRandom.NextDouble() * 200 - 100);
                 var npcPos = center + new Vector3(offsetX, 0, offsetZ);
                 npcPos.Y = terrain.GetHeight(npcPos);
                 npc.Transform.Position = npcPos;
                 npc.Transform.Scale = Vector3.One;
                 npc.Transform.Rotation = Quaternion.CreateFromYawPitchRoll((float)(npcRandom.NextDouble() * MathF.PI * 2), 0, 0);

                 var npcRenderer = npc.AddComponent<SkinnedMeshRenderer>();
                 npcRenderer.Mesh = mesh;
                 npcRenderer.Materials.Add(material ?? minionMaterial);

                 AnimationSetup.CreateBlendTreeAnimations(npc, mesh);
             }
             Debug.Log($"[NPCs] Spawned {minionCount} minions");
         }

         private static void SpawnLights(Vector3 center, IHeightProvider terrain)
         {
             // Test Point Lights — placed around player at terrain height + 5m
             float lightHeight = 5f;
             var redLightEntity = new Entity("RedLight");
             var redPos = center + new Vector3(-10, 0, 0);
             redPos.Y = terrain.GetHeight(redPos) + lightHeight;
             redLightEntity.Transform.Position = redPos;
             var redLight = redLightEntity.AddComponent<PointLight>();
             redLight.Color = new Color3(1, 0, 0);
             redLight.Intensity = 10.0f;
             redLight.Range = 30;

             var greenLightEntity = new Entity("GreenLight");
             var greenPos = center + new Vector3(0, 0, 10);
             greenPos.Y = terrain.GetHeight(greenPos) + lightHeight;
             greenLightEntity.Transform.Position = greenPos;
             var greenLight = greenLightEntity.AddComponent<PointLight>();
             greenLight.Color = new Color3(0, 1, 0);
             greenLight.Intensity = 10.0f;
             greenLight.Range = 30;

             var blueLightEntity = new Entity("BlueLight");
             var bluePos = center + new Vector3(10, 0, 0);
             bluePos.Y = terrain.GetHeight(bluePos) + lightHeight;
             blueLightEntity.Transform.Position = bluePos;
             var blueLight = blueLightEntity.AddComponent<PointLight>();
             blueLight.Color = new Color3(0, 0, 1);
             blueLight.Intensity = 10.0f;
             blueLight.Range = 30;
        }

         private static void SpawnTrees(IHeightProvider terrain, Vector3 center, float minSpacing, int variationCount)
         {
            // ===== TREES (5 variations  100 instances = 500 total) =====
             // Test multi-draw batching: different meshes, same shader/PSO
             string[] oakVariations = {
                 @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_01.fbx",
                 @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_02.fbx",
                 @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_03.fbx",
                 @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_04.fbx",
                 @"D:\Projects\2024\ProjectXYZ\Resources\Tree Prototypes\Oak\Oak_Trees\Oak_05.fbx"
             };
             
             StaticMesh[] oakMeshes = new StaticMesh[oakVariations.Length];
             for (int v = 0; v < oakVariations.Length; v++)
             {
                 oakMeshes[v] = Engine.Assets.Load<StaticMesh>(oakVariations[v]);
                 Debug.Log($"[Tree] Loaded Oak_{v+1:D2}.fbx with {oakMeshes[v].Mesh.MeshParts.Count} parts, PosBufferIndex: {oakMeshes[v].Mesh.PosBufferIndex}");
             }

             // Poisson disk sampling from regular grid to avoid tree overlap
             // Uses grid cells with jitter to ensure minimum spacing between trees
             Random treeRandom = new Random(54321);
             float areaSize = 1000f;  // 1000x1000 area
             float halfArea = areaSize / 2f;
             int gridSize = (int)(areaSize / minSpacing);  // 100x100 grid cells
             
             // Generate grid positions with jitter (simple Poisson approximation)
             var treePositions = new List<(float x, float z, int variation)>();
             for (int gx = 0; gx < gridSize; gx++)
             {
                 for (int gz = 0; gz < gridSize; gz++)
                 {
                     // Skip some cells randomly for natural look (50% density)
                     if (treeRandom.NextDouble() > 0.5)
                         continue;
                         
                     // Base position at grid cell center
                     float baseX = gx * minSpacing - halfArea + minSpacing / 2f;
                     float baseZ = gz * minSpacing - halfArea + minSpacing / 2f;
                     
                     // Add jitter within cell (keeps minimum spacing guarantee)
                     // 0.4 multiplier = ±2m jitter, guaranteeing minimum 6m between trees
                     float jitterX = (float)(treeRandom.NextDouble() - 0.5) * minSpacing * 0.4f;
                     float jitterZ = (float)(treeRandom.NextDouble() - 0.5) * minSpacing * 0.4f;
                     
                     float x = center.X + baseX + jitterX;
                     float z = center.Z + baseZ + jitterZ;
                     
                     int variation = treeRandom.Next(Math.Min(variationCount, oakMeshes.Length));
                     treePositions.Add((x, z, variation));
                 }
             }
             
             // Create tree entities from sampled positions
             int totalTrees = 0;
             foreach (var (x, z, variation) in treePositions)
             {
                 var treeEntity = new Entity($"Tree_{variation}_{totalTrees}");
                 
                 // Get terrain height at this position
                 float y = terrain.GetHeight(new Vector3(x, 0, z));
                 
                 treeEntity.Transform.Position = new Vector3(x, y, z);
                 treeEntity.Transform.Scale = Vector3.One * (0.8f + (float)treeRandom.NextDouble() * 0.4f);
                 treeEntity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll((float)(treeRandom.NextDouble() * MathF.PI * 2), 0, 0);
                 
                 var renderer = treeEntity.AddComponent<StaticMeshRenderer>();
                 renderer.StaticMesh = oakMeshes[variation];
                 totalTrees++;
             }
             Debug.Log($"[Trees] Placed {totalTrees} trees using Poisson sampling ({oakMeshes.Length} variations, {minSpacing}m spacing)");
         }

         private static void LoadScene()
         {
             var loader = new Freefall.Loaders.SceneLoader(@"D:\Projects\2024\ProjectXYZ\Resources");
             
             // Fire-and-forget async load — entire pipeline runs on background thread
             // (D3D12 free-threaded + lock-protected AssetManager/BindlessIndex/EntityManager)
             // Entities appear at next frame boundary via EntityManager.FlushPending()
             var progress = new Progress<string>(msg => Debug.Log($"[SceneLoader] {msg}"));
             Task.Run(() => loader.LoadAsync(@"D:\Dump\ExportedScene.json", progress))
                 .ContinueWith(t =>
                 {
                     if (t.IsFaulted)
                         Debug.Log($"[SceneLoader] FATAL: {t.Exception?.Flatten().InnerException}");
                     else
                         Debug.Log("[SceneLoader] Async load completed successfully.");
                 });
             
             Debug.Log("[SceneLoader] Async scene load started!");
         }
     }
}
