using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Editor.Tools;
using Freefall.Graphics;
using Freefall.Reflection;
using Squid;
using System.Reflection;
using static Freefall.Editor.MenuBuilder;
using Reflector = Freefall.Reflection.Reflector;

namespace Freefall.Editor
{
    public class EditorDesktop : Desktop
    {
        public DockGroup? Explorer;
        public DockGroup? Inspector;
        public DockGroup? Assets;
        public DockGroup? Settings;
        public DockGroup? Preferences;
        public DockGroup? Terrain;
        public DockGroup? Scene;
        public ViewportControl? SceneViewport;
        public DockGroup? DebugConsole;
        public DockGroup? Stats;
        public DockGroup? Graph;

        private DropDownButton componentsItem;

        public AssetBrowserControl AssetBrowser;
        public InspectorControl InspectorPanel;

        public DockRegion? DockArea;

        public Frame? statusBar;

        private ToastFrame toastFrame;

        /// <summary>Path of the scene last opened/saved (UI or command server); target of "Save Scene".</summary>
        public string? CurrentScenePath { get; set; }

        public EditorDesktop()
        {
            AssetCreator.Initialize();
            EditorSkin.Apply(this);

            // Hook up Windows clipboard for Squid GUI copy/paste
            Gui.OnSetClipboard = (text) => System.Windows.Forms.Clipboard.SetText(text);
            Gui.OnGetClipboard = () => System.Windows.Forms.Clipboard.ContainsText() ? System.Windows.Forms.Clipboard.GetText() : string.Empty;

            ModalColor = ColorInt.ARGB(.5f, 0f, 0f, 0f);

            // --- Menu Bar ---
            Frame menuBar = new Frame();
            menuBar.Style = "window";
            menuBar.Size = new Point(100, 30);
            menuBar.Dock = DockStyle.Top;
            Controls.Add(menuBar);

            DropDownButton btn = CreateMenuItem(menuBar, "File");
            AddMenuItem(btn, "New Scene", (s, a) => { });
            AddMenuItem(btn, "Open Scene", (s, a) =>
            {
                if (!Engine.IsEditor)
                    return;

                var dlg = new System.Windows.Forms.OpenFileDialog
                {
                    Filter = "Scene files (*.scene)|*.scene|All files (*.*)|*.*",
                    InitialDirectory = Engine.Project?.AssetsDirectory ?? "",
                    Title = "Open Scene"
                };
                if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    try
                    {
                        EntityManager.ClearScene();
                        var serializer = new Freefall.Serialization.EntitySerializer();
                        var entities = serializer.Load(dlg.FileName);

                        CurrentScenePath = dlg.FileName;

                        Toast.Show($"[Editor] Loaded scene: {entities.Count} entities");
                        Debug.Log($"[Editor] Loaded scene: {entities.Count} entities");
                        MessageDispatcher.Send(Msg.RefreshExplorer);
                    }
                    catch (Exception ex)
                    {
                        Toast.Show("Failed to load scene");
                        Debug.Log("Failed to load scene");
                    }
                }
            });
            AddSeparator(btn);
            AddMenuItem(btn, "Save Scene", (s, a) =>
            {
                if (!Engine.IsEditor)
                    return;

                if (!File.Exists(CurrentScenePath))
                {
                    Toast.Show("No scene to save. Use Save Scene As...", 2.0f);
                    return;
                }

                try
                {
                    var serializer = new Freefall.Serialization.EntitySerializer();
                    var entities = EntityManager.Entities.ToArray();
                    serializer.Save(CurrentScenePath, entities);
                    ProjectThumbnails.Capture();
                    Toast.Show("Saved scene: " + CurrentScenePath);
                }
                catch (Exception ex)
                {
                    Toast.Show("Failed to save scene");
                    Debug.Log("Failed to save scene");
                }
                return;
            });

            AddMenuItem(btn, "Save Scene as...", (s, a) => 
            {
                if (!Engine.IsEditor)
                    return;

                var dlg = new System.Windows.Forms.SaveFileDialog
                {
                    Filter = "Scene files (*.scene)|*.scene|All files (*.*)|*.*",
                    InitialDirectory = Engine.Project?.AssetsDirectory ?? "",
                    Title = "Save Scene"
                };

                if (dlg.ShowDialog() != System.Windows.Forms.DialogResult.OK)
                    return;
                

                try
                {
                    var serializer = new Freefall.Serialization.EntitySerializer();
                    var entities = EntityManager.Entities.ToArray();
                    serializer.Save(dlg.FileName, entities);
                    CurrentScenePath = dlg.FileName;
                    ProjectThumbnails.Capture();
                    Toast.Show("Saved scene: " + CurrentScenePath);
                }
                catch (Exception ex)
                {
                    Toast.Show("Failed to save scene");
                    Debug.Log("Failed to save scene");
                }

            });
            AddSeparator(btn);
            AddMenuItem(btn, "Exit", (s, a) => { System.Windows.Forms.Application.Exit(); });

            btn = CreateMenuItem(menuBar, "Edit");
            AddMenuItem(btn, "Undo", null);
            AddMenuItem(btn, "Redo", null);
            AddSeparator(btn);
            AddMenuItem(btn, "Cut", null);
            AddMenuItem(btn, "Copy", null);
            AddMenuItem(btn, "Paste", null);
            AddMenuItem(btn, "Duplicate", null);
            AddSeparator(btn);
            AddMenuItem(btn, "Delete", null);
            AddSeparator(btn);
            AddMenuItem(btn, "Preferences", (s, a) =>
            {
                if (Settings == null) return;
                foreach (var tab in Settings.TabPages)
                {
                    if (tab is Squid.TabPage tp && tp.Button.Text == "Preferences")
                    {
                        Settings.SelectedTab = tp;
                        break;
                    }
                }
            });

            btn = CreateMenuItem(menuBar, "Entity");
            AddMenuItem(btn, "Create Empty", (s, a) => CreateEmptyEntity());
            AddSeparator(btn);

            AddMenuItem(btn, "Sky", (s,a) => CreateSky());
            AddMenuItem(btn, "Terrain", (s,a) => CreateTerrain());
            AddMenuItem(btn, "Ocean", (s,a) => CreateOcean());
            AddMenuItem(btn, "Particle System", (s,a) => CreateParticleSystem());
           
            var sub = AddMenuItem(btn, "Spline...", null);
            AddMenuItem(sub, "Spline", (s, a) => CreateSpline());
            AddMenuItem(sub, "Road", (s, a) => CreateRoad());
            AddMenuItem(sub, "Wall", (s, a) => CreateWall());
            AddMenuItem(sub, "Floor", (s, a) => CreateFloor());

            sub = AddMenuItem(btn, "3D Primitive...", null);
            AddMenuItem(sub, "Cube", (s, a) => CreateCube());
            AddMenuItem(sub, "Sphere", (s, a) => CreateSphere());
            AddMenuItem(sub, "Capsule", null);
            AddMenuItem(sub, "Cylinder", null);
            AddMenuItem(sub, "Plane", null);
            AddMenuItem(sub, "Quad", null);

            sub = AddMenuItem(btn, "Light...", null);
            AddMenuItem(sub, "Point Light", (s, a) => CreatePointLight());
            AddMenuItem(sub, "Spot Light", (s, a) => CreateSpotLight());
            AddMenuItem(sub, "Area Light", (s, a) => CreateAreaLight());
            AddMenuItem(sub, "Directional Light", (s, a) => CreateDirectionalLight());

            sub = AddMenuItem(btn, "Audio...", null);
            AddMenuItem(sub, "Audio Source", (s, a) => CreateAudioSource());
            AddMenuItem(sub, "Reverb Zone", (s, a) => CreateReverbZone());

            AddSeparator(btn);
            AddMenuItem(btn, "Create Prefab from Selection", (s, a) => CreatePrefabFromSelection());


            componentsItem = CreateMenuItem(menuBar, "Component");
            CreateComponentItems();
                
            btn = CreateMenuItem(menuBar, "Window");
            AddMenuItem(btn, "Scene", (s, a) => { });
            AddMenuItem(btn, "Explorer", (s, a) => { });
            AddMenuItem(btn, "Inspector", (s, a) => { });
            AddMenuItem(btn, "Project", (s, a) => { });
            AddMenuItem(btn, "Settings", (s, a) => { });
            AddMenuItem(btn, "Console", (s, a) => { });

            btn = CreateMenuItem(menuBar, "Tools");
            AddMenuItem(btn, "Save Selected Asset...", (s, a) => SaveSelectedAsset());
            AddMenuItem(btn, "Save Changed Assets", (s, a) => SaveChangedAssets());
            AddSeparator(btn);
            AddMenuItem(btn, "Snap to Ground", (s, a) => SnapToGround());
            AddMenuItem(btn, "Convert to Skinned Mesh", (s, a) => ConvertMeshToSkinnedRenderer());

            AddSeparator(btn);
            AddMenuItem(btn, "Import Unity Pack...", (s, a) => ImportUnityPack());
            AddMenuItem(btn, "Load Unity Scene", (s, a) => LoadUnityScene());
            AddMenuItem(btn, "Noise Editor", null);

            // --- Status Bar ---
            statusBar = new Frame
            {
                Style = "window",
                Size = new Point(20, 26),
                Dock = DockStyle.Bottom,
                Parent = this
            };

            var fpslabel = new Label
            {
                Style = "statusBarLabel",
                Size = new Point(80, 20),
                Dock = DockStyle.Right,
                Margin = new Margin(1, 0, 0, 0),
            };

            var fill = new Frame { Style = "frame", Dock = DockStyle.Fill };

            statusBar.Controls.Add(fpslabel);
            statusBar.Controls.Add(fill);
            fpslabel.Update += (s) => { fpslabel.Text = $"FPS: {Freefall.Base.Time.FPS}"; };

            // --- Toolstrip ---
            var toolstrip = new Frame
            {
                Style = "frame",
                Dock = DockStyle.Top,
                Size = new Point(32, 32),
                Margin = new Margin(0, 0, 0, 0)
            };
            Controls.Add(toolstrip);

            var playModeFrame = new Frame
            {
                Size = new Point(32 * 2 + 1, 32),
                Dock = DockStyle.CenterX,
                Margin = new Margin(0, 0, 0, 0)
            };
            toolstrip.Controls.Add(playModeFrame);
            
            var playButton = new Button
            {
                Style = "iconplay",
                Size = new Point(32, 32),
                Margin = new Margin(0, 0, 1, 0),
                Dock = DockStyle.Left
            };

            var stopButton = new Button
            {
                Style = "iconstop",
                Size = new Point(32, 32),
                Margin = new Margin(0, 0, 0, 0),
                Dock = DockStyle.Left
            };

            playButton.MouseClick += (s, a) =>
            {
                // if scene has no main camera -> throw a Toast and return
                var validCamera = ComponentCache<Camera>.All.FirstOrDefault(x => x != EditorCamera.Camera);
                if (validCamera   == null)
                {
                    Toast.Show("Scene must have a main camera to enter play mode.", 3.0f);
                    return;
                }

                // if scene has not been saved -> throw a Toast and return
                if (!File.Exists(CurrentScenePath))
                {
                    Toast.Show("Scene must be saved before entering play mode.", 3.0f);
                    return;
                }

                // save scene
                var serializer = new Freefall.Serialization.EntitySerializer();
                var entities = EntityManager.Entities.ToArray();
                serializer.Save(CurrentScenePath, entities);

                // clear scene
                EntityManager.ClearScene();

                // destroy EditorCamera
                EditorCamera.DestroyCamera();

                // enter playmode
                Engine.SetPlaymode(true);

                // load scene
                serializer = new Freefall.Serialization.EntitySerializer();
                var result = serializer.Load(CurrentScenePath);

                foreach (var camera in ComponentCache<Camera>.All)
                    camera.Target = RenderView.All[1];

                playButton.Enabled = false;
                stopButton.Enabled = true;
                MessageDispatcher.Send(Msg.RefreshExplorer);
            };


            stopButton.MouseClick += (s, a) =>
            {
                // clear scene
                EntityManager.ClearScene();

                // exit playmode (Engine.IsEditor = true)
                Engine.SetPlaymode(false);

                // load scene
                var serializer = new Freefall.Serialization.EntitySerializer();
                var result = serializer.Load(CurrentScenePath);

                // recreate EditorCamera
                EditorCamera.CreateCamera();

                foreach (var camera in ComponentCache<Camera>.All)
                    camera.Target = RenderView.All[1];

                playButton.Enabled = true;
                stopButton.Enabled = false;
                MessageDispatcher.Send(Msg.RefreshExplorer);
            };

            stopButton.Enabled = false;

            playModeFrame.Controls.Add(playButton);
            playModeFrame.Controls.Add(stopButton);

            // --- Dock Area with placeholder panels ---
            DockArea = new DockRegion { Dock = DockStyle.Fill };

            // Viewport with render-to-texture
            var sceneViewport = new ViewportControl();
            SceneViewport = sceneViewport;
            Scene = DockArea.DockContent("Scene", sceneViewport);
            var graphEditor = new GraphEditor();
            Graph = DockArea.DockContent("Graph", graphEditor);
            CustomAssetEditorRegistry.Register(graphEditor);

            var animEditor = new AnimationEditor();
            DockArea.DockContent(Graph!, "Animation", animEditor, DockStyle.Fill);
            CustomAssetEditorRegistry.Register(animEditor);

            // Explorer (right side)
            var explorerControl = new ExplorerControl();
            Explorer = DockArea.DockContent(Scene!, "Explorer", explorerControl, DockStyle.Right);
            Settings = DockArea.DockContent(Explorer!, "Settings", new SettingsControl(), DockStyle.Fill);
            Preferences = DockArea.DockContent(Settings!, "Preferences", new PreferencesControl(), DockStyle.Fill);

            // Inspector (below explorer)
            InspectorPanel = new InspectorControl();
            Inspector = DockArea.DockContent(Scene!, "Inspector", InspectorPanel, DockStyle.Right);
            Stats = DockArea.DockContent(Inspector!, "Stats", new StatsControl(), DockStyle.Fill);

            // Assets panel (below scene)
            AssetBrowser = new AssetBrowserControl();
            AssetBrowser.Size = new Point(200, 200);
            Assets = DockArea.DockContent(Scene!, "Assets", AssetBrowser, DockStyle.Left);
            Terrain = DockArea.DockContent(Assets!, "Terrain", new TerrainPanel(), DockStyle.Fill);
            var watabou = DockArea.DockContent(Assets!, "Watabou", new WatabouImporterPanel(), DockStyle.Fill);

            // Debug placeholder (tab next to assets)
            DebugConsole = DockArea.DockContent(Preferences!, "Debug", new ConsoleControl(), DockStyle.Fill);

            Controls.Add(DockArea);

            // Wire up asset browser refresh via MessageDispatcher
            // ScanAndSync must run on the main thread to trigger actual reimport
            MessageDispatcher.AddListener(Msg.RefreshAssets, (msg) =>
            {
                AssetDatabase.Refresh();
                AssetDatabase.ImportAll();
                AssetDatabase.GenerateMissingThumbnails(rendererFactory: () => new ThumbnailRenderer());
                AssetBrowser.RefreshCurrentFolder();
            });

            // Bridge engine PCG events to editor UI refresh
            MessageDispatcher.AddListener(EngineMsg.PCGExecuted, (msg) =>
            {
                MessageDispatcher.Send(Msg.RefreshExplorer);
            });

            toastFrame = new ToastFrame();
            toastFrame.Size = new Point(200, 200);
            //toastFrame.Position = Size - toastFrame.Size - new Point(30, 40);
            //toastFrame.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            toastFrame.Dock = DockStyle.Center;
            Elements.Add(toastFrame);

            // Agent screenshot heads-up (drawn over everything, see Mcp/ScreenshotScheduler)
            Elements.Add(new ScreenshotCueOverlay());

            MessageDispatcher.AddListener(Msg.OpenAssetInEditor, OpenAssetInEditor);
            MessageDispatcher.AddListener(Msg.ScriptsReloaded, (m) => CreateComponentItems());
        }

        private void ConvertMeshToSkinnedRenderer()
        {
            foreach(var entity in Selector.Selection)
                ConvertMeshToSkinnedRenderer(entity);
        }

        private void ConvertMeshToSkinnedRenderer(Entity entity)
        {
            var rend = entity.GetComponent<MeshRenderer>();
            if (rend != null)
            {
                var mesh = rend.Mesh;
                if (mesh != null)
                {
                    entity.RemoveComponent<MeshRenderer>();

                    var skinned = entity.AddComponent<SkinnedMeshRenderer>();
                    skinned.Mesh = mesh;

                    foreach (var mat in rend.Materials)
                        skinned.Materials.Add(mat.Material);
                }
            }

            foreach (Transform child in entity.Transform)
                ConvertMeshToSkinnedRenderer(child.Entity);
        }

        private void CreateSphere()
        {
            var entity = new Entity("New Sphere");
            var rend = entity.AddComponent<MeshRenderer>();
            rend.Mesh = InternalAssets.SphereMesh;
            rend.Material = InternalAssets.DefaultMaterial;

            if (Selector.SelectedEntity != null)
                entity.Transform.Parent = Selector.SelectedEntity.Transform;
            MessageDispatcher.Send(Msg.RefreshExplorer);

        }

        private void CreateCube()
        {
            var entity = new Entity("New Cube");
            var rend = entity.AddComponent<MeshRenderer>();
            rend.Mesh = InternalAssets.CubeMesh;
            rend.Material = InternalAssets.DefaultMaterial;
            if (Selector.SelectedEntity != null)
                entity.Transform.Parent = Selector.SelectedEntity.Transform;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }


        private void CreateEmptyEntity()
        {
            var entity = new Entity("New Entity");
            if (Selector.SelectedEntity != null)
                entity.Transform.Parent = Selector.SelectedEntity.Transform;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateAudioSource()
        {
            var entity = new Entity("New Audio Source");
            entity.AddComponent<AudioSource>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateReverbZone()
        {
            var entity = new Entity("New Reverb Zone");
            //entity.AddComponent<ReverbZone>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateAreaLight()
        {
            var entity = new Entity("New Area Light");
            //var light = entity.AddComponent<AreaLight>();
            //light.Intensity = 1;
            //light.Range = 10;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateSpotLight()
        {
            var entity = new Entity("New Spot Light");
            //var light = entity.AddComponent<SpotLight>();
            //light.Intensity = 1;
            //light.Range = 10;
            //light.SpotAngle = 45;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreatePointLight()
        {
            var entity = new Entity("New Point Light");
            var light = entity.AddComponent<PointLight>();
            light.Intensity = 1;
            light.Range = 10;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateDirectionalLight()
        {
            var entity = new Entity("New Directional Light");
            var light = entity.AddComponent<DirectionalLight>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateOcean()
        {
            var entity = new Entity("New Ocean");
            entity.AddComponent<OceanRenderer>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateParticleSystem()
        {
            var entity = new Entity("New Particle System");
            var emitter = entity.AddComponent<ParticleEmitter>();
            emitter.ParticleTexture = InternalAssets.White;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateSky()
        {
            var entity = new Entity("New Sky");
            entity.AddComponent<SkyboxRenderer>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateTerrain()
        {
            var entity = new Entity("New Terrain");
            entity.AddComponent<TerrainRenderer>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateSpline()
        {
            var entity = new Entity("New Spline");
            entity.AddComponent<Spline>();
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateRoad()
        {
            var entity = new Entity("New Road");
            entity.AddComponent<Spline>();
            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 0, Material = InternalAssets.DefaultMaterial });
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 1, Material = InternalAssets.DefaultMaterial });
            var mesh = entity.AddComponent<RuntimeMesh>();
            mesh.EnableCurbs = true;
            mesh.Width = 4;
            mesh.Height = .1f;
            mesh.CurbHeight = .12f;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateWall()
        {
            var entity = new Entity("New Wall");
            entity.AddComponent<Spline>();
            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 0, Material = InternalAssets.DefaultMaterial });
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 1, Material = InternalAssets.DefaultMaterial });
            var mesh = entity.AddComponent<RuntimeMesh>();
            mesh.EnableCurbs = false;
            mesh.Width = .2f;
            mesh.Height = 2f;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        private void CreateFloor()
        {
            var entity = new Entity("New Floor");
            var spline = entity.AddComponent<Spline>();
            spline.Closed = true;
            var renderer = entity.AddComponent<MeshRenderer>();
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 0, Material = InternalAssets.DefaultMaterial });
            renderer.Materials.Add(new MaterialOverride { MaterialSlot = 1, Material = InternalAssets.DefaultMaterial });
            var mesh = entity.AddComponent<RuntimeMesh>();
            mesh.Height = .1f;
            MessageDispatcher.Send(Msg.RefreshExplorer);
        }

        void CreateComponentItems()
        {
            componentsItem.Dropdown.Controls.Clear();

            var types = Reflector.GetTypes<Component>();
            types.Sort((a, b) => a.Name.CompareTo(b.Name));

            foreach (Type type in types)
            {
                if (type.IsAbstract) continue;
                AddMenuItem(componentsItem, type.Name, (s, a) => CreateComponent(type));
            }
        }

        void OpenAssetInEditor(Message message)
        {
            (string guid, Type assetType) = ((string, Type))message.Data;

            var editor = CustomAssetEditorRegistry.Find(assetType);
            if (editor == null)
            {
                Debug.Log($"[AssetBrowser] No editor found for: {assetType.Name}");
                return;
            }

            try
            {
                var asset = Engine.Assets.LoadByGuid(guid, assetType);
                if (asset == null)
                {
                    Debug.Log($"[AssetBrowser] Failed to load asset: {guid}");
                    return;
                }

                editor.OpenAsset(asset);

                var control = editor as Control;
                var page = FindTabPage(control);

                if(page!= null && page.Tag is DockGroup dockGroup)
                {
                    dockGroup.SelectedTab = page;
                }
            }
            catch (Exception ex)
            {
                Debug.Log($"[AssetBrowser] Failed to open asset in editor: {ex.Message}");
            }
        }

        Squid.TabPage FindTabPage(Control control)
        {
            Control current = control;
            while (current != null)
            {
                if (current is Squid.TabPage tp)
                    return tp;
                
                current = current.Parent;
            }
            return null;
        }

        void SnapToGround()
        {
            var terrain = Commands.TerrainHeightCommand.FindTerrainRenderer();
            if (terrain == null) return;

            foreach (var entity in Selector.Selection)
            {
                var renderer = entity.GetComponent<MeshRenderer>();
                if (renderer == null || renderer.Mesh == null) continue;
                var pos = entity.Transform.Position;
                pos.Y = terrain.GetHeight(new System.Numerics.Vector3(pos.X, 0, pos.Z));
                entity.Transform.Position = pos;
            }
        }

        public void CreateComponent(Type type)
        {
            MethodInfo info1 = typeof(EditorDesktop).GetMethod("CreateComponent", new Type[] { });
            MethodInfo info2 = info1.MakeGenericMethod(type);
            info2.Invoke(this, null);
        }

        public void CreateComponent<T>() where T: Freefall.Base.Component, new()
        {
            foreach(var entity in Selector.Selection)
            {
                if(entity.GetComponent<T>() != null)
                    continue;
                entity.AddComponent<T>();
            }

            MessageDispatcher.Send(Msg.RefreshInspector); 
        }

        private static void SaveSelectedAsset()
        {
            var selected = Selector.SelectedObject;
            Asset selectedAsset = null;

            if (selected is Asset asset)
            {
                selectedAsset = asset;
            }
            else if (selected is Freefall.Base.Entity entity)
            {
                // Scan entity components for asset fields
                foreach (var comp in entity.Components)
                {
                    var mapping = Freefall.Reflection.Reflector.GetMapping(comp.GetType());
                    foreach (var field in mapping)
                    {
                        if (!typeof(Asset).IsAssignableFrom(field.Type)) continue;
                        var value = field.GetValue(comp) as Asset;
                        if (value != null) { selectedAsset = value; break; }
                    }
                    if (selectedAsset != null) break;
                }
            }

            if (selectedAsset == null)
            {
                Toast.Show("Nothing selected.", 2.0f);
                return;
            }

            // Resolve source path from asset GUID or prompt for location
            string savePath = null;
            if (!string.IsNullOrEmpty(selectedAsset.Guid))
                savePath = AssetDatabase.GuidToPath(selectedAsset.Guid);

            if (!string.IsNullOrEmpty(savePath) && Engine.Project != null)
                savePath = System.IO.Path.Combine(Engine.Project.AssetsDirectory, savePath);

            if (string.IsNullOrEmpty(savePath) || !System.IO.File.Exists(savePath))
            {
                var dlg = new System.Windows.Forms.SaveFileDialog
                {
                    Filter = "Asset files (*.*)|*.*",
                    InitialDirectory = Engine.Project?.AssetsDirectory ?? "",
                    FileName = selectedAsset.Name ?? "asset",
                    Title = "Save Asset"
                };
                if (dlg.ShowDialog() != System.Windows.Forms.DialogResult.OK) return;
                savePath = dlg.FileName;
            }

            try
            {
                Engine.Assets.SaveAsset(selectedAsset, savePath);

                // Reimport source → cache so the packed binary matches
                // the YAML we just wrote. Without this, EvictByGuid would
                // cause the next load to read stale cache data.
                if (!string.IsNullOrEmpty(selectedAsset.Guid))
                {
                    var relativePath = AssetDatabase.GuidToPath(selectedAsset.Guid);
                    if (relativePath != null)
                        AssetDatabase.ImportAssetByPath(relativePath);
                }

                Debug.Log($"[Editor] Asset saved: {savePath}");
                Toast.Show($"Saved to:\n{savePath}", 2.0f);
            }
            catch (Exception ex)
            {
                Debug.Log($"[Editor] Failed to save asset: {savePath}");
            }
        }

        private static void SaveChangedAssets()
        {
            if (!AssetCreator.HasChangedAssets)
            {
                Toast.Show("No changed assets.", 2.0f);
                return;
            }

            int count = AssetCreator.SaveChangedAssets();
            Toast.Show($"Saved {count} asset(s).", 2.0f);
        }

        // ── Prefab ──

        private static void CreatePrefabFromSelection()
        {
            if (Selector.SelectedObject is not Entity entity)
                return;

            // create new prefab asset
            //var prefab = Prefab.Create(entity);

            //try
            //{
            //    Engine.Assets.SaveAsset(prefab, savePath);

            //    // Register in AssetDatabase
            //    AssetDatabase.Refresh();
            //    string relativePath = AssetCreator.GetRelativePath(savePath);
            //    var guid = AssetDatabase.PathToGuid(relativePath);

            //    if (guid != null)
            //        AssetDatabase.ImportAssetByPath(relativePath);

            //    Debug.Log($"[AssetCreator] Created {prefab.Name}: {savePath}");
            //}
            //catch (Exception ex)
            //{
            //    Debug.Log($"[AssetCreator] Failed to create {prefab.Name}: {ex.Message}");
            //}

            // save to folder that is currently selected in asset browser,
            // or default to project assets folder
        }


        // ── Unity Import ──

        private void ImportUnityPack()
        {
            var window = new UnityImporterWindow();
            window.Show(this);
        }

        private void LoadUnityScene()
        {
            using (System.Windows.Forms.OpenFileDialog dlg = new System.Windows.Forms.OpenFileDialog())
            {
                dlg.Filter = "Scene Json|*.json";
                if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    UnityImporter.LoadScene(dlg.FileName);
                    Debug.Log($"[LoadUnityScene] Loaded scene from {dlg.FileName}");
                }
            }
        }


    }
}
