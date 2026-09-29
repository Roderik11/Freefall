using System;
using System.Threading;
using System.Globalization;
using System.Runtime.InteropServices;
using Vortice.WinForms;
using System.Windows.Forms;
using Freefall;
using Freefall.Base;
using Freefall.Assets;
using Freefall.Graphics;
using Freefall.Components;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    static class Program
    {
        [DllImport("DwmApi")]
        static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, int[] attrValue, int attrSize);

        static SquidRenderer squidRenderer = null!;
        static RenderView mainView = null!;
        static RenderForm form = null!;
        static EditorCommandServer commandServer = null!;

        // Active desktop — either LandingDesktop or EditorUI
        static Squid.Desktop activeDesktop = null!;
        static EditorUI editorUI = null!;
        static LandingDesktop landingDesktop = null!;

        // Exposed for command server access
        internal static RenderForm Form => form;
        internal static EditorUI EditorUI => editorUI;
        internal static LandingDesktop Landing => landingDesktop;
        internal static EditorCommandServer CommandServer => commandServer;
        internal static bool IsProjectOpen => editorUI != null;

        /// <summary>
        /// Routes WM mouse-button messages to Input.ProcessMessage
        /// </summary>
        class InputFilter : System.Windows.Forms.IMessageFilter
        {
            public bool PreFilterMessage(ref System.Windows.Forms.Message m)
            {
                // WM_INPUT = 0x00FF — raw mouse movement for camera delta
                if (m.Msg == 0x00FF)
                    Input.ProcessRawInput(m.LParam);

                Input.ProcessMessage((uint)m.Msg, (UIntPtr)(long)m.WParam, m.LParam);
                return false; // don't swallow — let WinForms process normally
            }
        }

        [STAThread]
        static void Main()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            var color = System.Drawing.ColorTranslator.FromHtml("#222222");

            form = new RenderForm
            {
                BackColor = color,
                Size = new System.Drawing.Size(1680, 1020),
                StartPosition = FormStartPosition.CenterScreen,
                Text = "Freefall Editor"
            };

            form.HandleCreated += (s, e) =>
            {
                // Enable dark title bar
                if (DwmSetWindowAttribute(form.Handle, 19, [1], 4) != 0)
                    DwmSetWindowAttribute(form.Handle, 20, [1], 4);
            };

            form.Show();

            // Initialize engine WITHOUT a project (landing page first)
            Engine.Initialize(form.Handle, form.ClientSize.Width, form.ClientSize.Height);

            // Register editor assembly so Reflector can discover PropertyControl types, GUIInspectors, etc.
            Freefall.Reflection.Reflector.RegisterAssemblies(System.Reflection.Assembly.GetAssembly(typeof(Program)));

            // The main swapchain view was auto-registered in RenderView.All by Engine.Initialize.
            mainView = RenderView.All[0];
            mainView.OnRender = RenderGui;

            // Ensure the GUI view renders LAST (after scene viewports) — Apex pattern
            RenderView.All.Remove(mainView);
            RenderView.All.Add(mainView);

            // Forward resize events to the main view
            form.SizeChanged += (s, e) =>
            {
                if (form.WindowState == FormWindowState.Minimized) return;
                mainView.Resize(form.ClientSize.Width, form.ClientSize.Height);
            };

            // Initialize timing
            Freefall.Base.Time.Initialize();

            // --- Squid GUI setup ---
            squidRenderer = new SquidRenderer(Engine.Device);
            Squid.Gui.Renderer = squidRenderer;

            // Inject thumbnail textures into Squid when they're lazy-loaded
            AssetDatabase.OnThumbnailLoaded = (name, tex) => squidRenderer.InsertTexture(name, tex);

            // Register the Gray fallback texture so GetThumbnail's fallback renders correctly
            squidRenderer.InsertTexture(InternalAssets.Gray.Name, InternalAssets.Gray);

            // Route WM messages to Input for mouse button tracking
            Application.AddMessageFilter(new InputFilter());

            // --- Landing page ---
            RecentProjects.Load();
            EditorPreferences.Load();
            landingDesktop = new LandingDesktop();
            landingDesktop.OnProjectSelected += OpenProject;
            activeDesktop = landingDesktop;

            // Start the AI agent command server (before project open so agent can trigger it)
            commandServer = new EditorCommandServer();
            commandServer.Start();

            RenderLoop.Run(form, Engine.Tick, true);

            commandServer?.Dispose();
            squidRenderer.Dispose();
            EditorPreferences.Save();
            Engine.Shutdown();
        }

        /// <summary>
        /// Called when a project is selected from the landing page.
        /// Opens the project, runs async import with progress dialog,
        /// then swaps to the editor desktop on success.
        /// </summary>
        internal static async void OpenProject(string path)
        {
            var landing = activeDesktop as LandingDesktop;
            if (landing == null) return;

            try
            {
                // Sync: open project, scan meta files
                Engine.OpenProject(path);
                RecentProjects.Add(Engine.Project.Name, path);

                // Show import progress dialog
                landing.ShowImportDialog(Engine.Project.Name);

                // Use direct callback instead of Progress<T> to avoid SynchronizationContext marshaling issues
                // UpdateImportStatus is already thread-safe (volatile + OnUpdate polling)
                await AssetDatabase.ImportAllAsync(
                    status => landing.UpdateImportStatus(status));

                // Marshal back to UI thread for editor initialization
                form.Invoke(() =>
                {
                    landing.HideImportDialog();

                    //Freefall.Editor.Tools.ManifestImporter.GenerateAssets(Engine.Project.AssetsDirectory);
                    //Freefall.Editor.Tools.ManifestImporter.SaveScene(Engine.Project.AssetsDirectory);

                    // SaveScene may create new .asset files (terrain, material) that weren't
                    // present during the initial import. Re-scan to pick them up, then import.
                    AssetDatabase.Refresh();
                    AssetDatabase.ImportAll();

                    // Generate thumbnails for any assets that don't have one yet
                    AssetDatabase.GenerateMissingThumbnails(
                        status => landing.UpdateImportStatus(status),
                        () => new ThumbnailRenderer());

                    ScriptCompiler.Initialize(Engine.Project.AssetsDirectory);

                    // Swap from landing page to editor desktop
                    editorUI = new EditorUI(form);
                    EditorPreferences.RegisterStyles(editorUI);
                    editorUI.AssetBrowser.Initialize();
                    EditorPreferences.DiscoverAllTypes();
                    activeDesktop = editorUI;

                    CommandBuffer.InitializeCuller(Engine.Device);

                    // Create default editor scene (sun, skybox, camera)
                    var defaultScene = new EditorDefaultScene();
                    EditorCamera.Camera.Target = RenderView.All[1];

                    MessageDispatcher.Send(Msg.RefreshExplorer);

                    // Notify listeners that the project is ready
                    MessageDispatcher.Send(Msg.ProjectOpened, new
                    {
                        path = Engine.Project.RootDirectory,
                        name = Engine.Project.Name,
                        entityCount = Base.EntityManager.Entities.Count
                    });
                });
            }
            catch (Exception ex)
            {
                Debug.LogWarning("Editor", $"Failed to open project: {ex.Message}");

                form.Invoke(() =>
                {
                    landing.ShowErrorDialog($"Failed to open project:\n{ex.Message}");
                    landing.OnErrorDismissed += () => form.Close();
                });
            }
        }

        /// <summary>
        /// Apex OnRender callback: render viewports + GUI onto the main swapchain.
        /// Called by Engine.Tick for views with OnRender set.
        /// </summary>
        static void RenderGui(RenderView view)
        {
            view.Prepare();

            var cmd = view.CommandList;

            // Squid layout pass
            squidRenderer.SetContext(cmd.Native, view.Width, view.Height);

            // Process any AI agent commands queued by the HTTP server
            commandServer?.ProcessCommands();

            // Agent screenshots: cue → settle → capture, one step per frame (never blocks here)
            Mcp.ScreenshotScheduler.Tick();

            // Update active desktop (landing page or editor)
            if (activeDesktop is EditorUI editor)
            {
                editor.Update();


                // find all active camera
                var cameras = ComponentCache<Camera>.All;
                
                foreach(var camera in cameras)
                {
                    if(camera != Camera.Main) continue;
                    camera.Render(cmd.Native);
                    CommandBuffer.Clear();
                }

                // Render headless views (preview viewports etc.) on the primary command list.
                foreach (var headless in RenderView.All)
                {
                    if (headless == view) continue;
                    if (!headless.Enabled) continue;
                    if (headless.HasSwapChain) continue;
                    if (headless.OnRender == null) continue;

                    headless.OnRender(headless);
                }
            }
            else if (activeDesktop is LandingDesktop landing)
            {
                // Landing page: feed mouse/time to Squid and update
                Squid.Gui.TimeElapsed = Freefall.Base.Time.DeltaMilliseconds;

                var cursorPos = System.Windows.Forms.Cursor.Position;
                var clientPos = form.PointToClient(new System.Drawing.Point(cursorPos.X, cursorPos.Y));
                Squid.Gui.SetMouse(clientPos.X, clientPos.Y, -Input.MouseWheelDelta);

                Squid.Gui.SetButtons(
                    (System.Windows.Forms.Control.MouseButtons & MouseButtons.Left) != 0,
                    (System.Windows.Forms.Control.MouseButtons & MouseButtons.Right) != 0,
                    (System.Windows.Forms.Control.MouseButtons & MouseButtons.Middle) != 0
                );

                landing.Size = new Squid.Point(form.ClientSize.Width, form.ClientSize.Height);
                landing.Update();
            }

            // GUI rendering onto swapchain backbuffer
            cmd.SetRenderTargets(view.BackBufferTarget, view.DepthBufferTarget);
            cmd.SetViewport(new Viewport(0, 0, view.Width, view.Height, 0.0f, 1.0f));
            cmd.SetScissorRect(new RectI(0, 0, view.Width, view.Height));
            cmd.ClearRenderTargetView(view.BackBufferTarget, new Color4(0, 0, 0, 1));
            activeDesktop.Draw();

            view.Present();
        }
    }
}
