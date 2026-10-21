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

        static EditorSquidRenderer squidRenderer = null!;
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

        static System.Drawing.Icon? LoadAppIcon()
        {
            using var stream = typeof(Program).Assembly.GetManifestResourceStream("Freefall.Editor.icon.ico");
            return stream != null ? new System.Drawing.Icon(stream) : null;
        }

        [STAThread]
        static void Main()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            // The landing page's fonts rasterize on workers while the engine starts (not in the first frame)
            LandingArt.PrefetchFonts();

            var color = System.Drawing.ColorTranslator.FromHtml("#222222");

            form = new RenderForm
            {
                BackColor = color,
                Size = new System.Drawing.Size(1680, 1020),
                StartPosition = FormStartPosition.CenterScreen,
                Text = "Freefall Editor",
                Icon = LoadAppIcon()
            };

            form.HandleCreated += (s, e) =>
            {
                // Enable dark title bar
                if (DwmSetWindowAttribute(form.Handle, 19, [1], 4) != 0)
                    DwmSetWindowAttribute(form.Handle, 20, [1], 4);
            };

            // Landing-page snapshot of the project being closed (no-op while on the landing page)
            form.FormClosing += (s, e) => ProjectThumbnails.Capture();

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
            squidRenderer = new EditorSquidRenderer(Engine.Device);
            Squid.Gui.Renderer = squidRenderer;

            // One control's bug must not close the editor: Squid skips a control whose own draw or update throws
            // and reports it here. Each distinct failure is logged once (it would otherwise repeat every frame).
            // A slider laid out 2 px wide for one frame threw from Math.Clamp and took the editor down (2026-10-15).
            var reportedControlErrors = new System.Collections.Generic.HashSet<string>();
            Squid.Gui.ControlException = (control, ex) =>
            {
                string key = control.GetType().FullName + "|" + ex.GetType().FullName + "|" + ex.Message;
                if (reportedControlErrors.Add(key))
                    Debug.LogError("[GUI]", $"{control.GetType().Name} skipped: {ex}");
            };

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

            // Start the AI agent command server (before project open so agent can trigger it). Kestrel and the MCP
            // tool scan take ~0.25 s: on a worker, the first frame does not wait for them (requests are queued for
            // the main thread anyway).
            commandServer = new EditorCommandServer();
            System.Threading.Tasks.Task.Run(commandServer.Start);

            // Re-apply the icon now that startup is done: the taskbar asks the window for its icon while
            // Engine.Initialize is still blocking this thread, gives up, and keeps a generic one until told again.
            form.Icon = LoadAppIcon();

            RenderLoop.Run(form, Engine.Tick, true);

            commandServer?.Dispose();
            squidRenderer.Dispose();
            EditorPreferences.Save();
            Engine.Shutdown();
        }

        /// <summary>
        /// Called when a project is selected from the landing page (or MCP project_open): starts opening it in the
        /// background with the progress popup (ProjectOpening). Main thread. Does nothing while an open is running.
        /// </summary>
        internal static void OpenProject(string path)
        {
            ProjectOpening.Start(path, activeDesktop as LandingDesktop);
        }

        /// <summary>
        /// Last step of ProjectOpening, main thread: swap the landing page for the editor desktop (built in an earlier
        /// slice) and create the default scene.
        /// </summary>
        internal static void ShowEditor(EditorUI ui)
        {
            (activeDesktop as LandingDesktop)?.HideImportDialog();

            editorUI = ui;
            activeDesktop = ui;

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

            using (_commandsMarker.Auto())
            {
                // Process any AI agent commands queued by the HTTP server
                commandServer?.ProcessCommands();

                // Agent screenshots: cue → settle → capture, one step per frame (never blocks here)
                Mcp.ScreenshotScheduler.Tick();
            }

            // Update active desktop (landing page or editor)
            if (activeDesktop is EditorUI editor)
            {
                using (_editorUiMarker.Auto())
                {
                    editor.Update();

                    // The game's UI over the scene viewport in play mode (after the editor's: it takes
                    // the mouse only where the editor left it to the picture)
                    GameUIHost.Update(editor);
                }

                // find all active camera
                var cameras = ComponentCache<Camera>.All;

                using (_sceneViewMarker.Auto())
                {
                    PixMarker.Begin(cmd.Native, "Scene view");
                    foreach(var camera in cameras)
                    {
                        if(camera != Camera.Main) continue;
                        camera.Render(cmd.Native);
                        CommandBuffer.Clear();
                    }
                    PixMarker.End(cmd.Native);
                }

                // Render headless views (preview viewports etc.) on the primary command list.
                using (_previewsMarker.Auto())
                {
                    PixMarker.Begin(cmd.Native, "Previews");
                    foreach (var headless in RenderView.All)
                    {
                        if (headless == view) continue;
                        if (!headless.Enabled) continue;
                        if (headless.HasSwapChain) continue;
                        if (headless.OnRender == null) continue;

                        headless.OnRender(headless);
                    }
                    PixMarker.End(cmd.Native);
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
            using (_editorUiDrawMarker.Auto())
            {
                PixMarker.Begin(cmd.Native, "Editor UI");
                activeDesktop.Draw();
                PixMarker.End(cmd.Native);
            }

            view.Present();

            if (!_firstFrameLogged)
            {
                // Measured 2026-10-17: 11.1 s before the shader disk cache, PVD check and font prefetch, 2.1 s after
                _firstFrameLogged = true;
                var shaders = Shader.CacheStats;
                var sinceStart = DateTime.Now - System.Diagnostics.Process.GetCurrentProcess().StartTime;
                Debug.LogAlways($"[Startup] first frame {sinceStart.TotalSeconds:F1} s after process start; shaders: " +
                                $"{shaders.DiskHits} from the disk cache, {shaders.Misses} compiled ({shaders.CompileMs / 1000:F1} s)");
            }
        }

        private static bool _firstFrameLogged;

        private static readonly ProfilerMarker _commandsMarker = ProfilerMarker.Get("Editor commands");
        private static readonly ProfilerMarker _editorUiMarker = ProfilerMarker.Get("Editor UI");
        private static readonly ProfilerMarker _sceneViewMarker = ProfilerMarker.Get("Scene view");
        private static readonly ProfilerMarker _previewsMarker = ProfilerMarker.Get("Previews");
        private static readonly ProfilerMarker _editorUiDrawMarker = ProfilerMarker.Get("Editor UI draw");
    }
}
