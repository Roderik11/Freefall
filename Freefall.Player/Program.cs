
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Freefall;
using Freefall.Assets;
using Freefall.Assets.Packs;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.UI;
using Vortice.Mathematics;
using Vortice.WinForms;

namespace Freefall.Player
{
    /// <summary>
    /// A published Freefall game. Next to it (or in the folder <c>--game</c> names): Data/ with game.manifest and the
    /// packs, and FreefallScripts.dll (the project's scripts). It opens a window (borderless fullscreen; Alt+Enter
    /// toggles a window), starts the engine in game mode on the packs, loads the scripts, starts loading the build's
    /// first scene (SceneManager.LoadAsync, from the first frame on) and draws the game's UI (GameUI, Graphics.Overlay)
    /// over the picture. Writes nothing into its own folder, which an installer may have made read-only (Program Files):
    /// what it writes goes to %LOCALAPPDATA%\&lt;Product&gt; (<see cref="UserFolder"/>): player.log (the run before is
    /// player-prev.log) and the shader cache.
    /// Development switches: <c>--game &lt;folder&gt;</c>, <c>--windowed [WxH]</c>, <c>--screenshot &lt;seconds&gt; &lt;file.png&gt;</c>
    /// (the window that long after the first scene is ready; a negative number: that long after the window opened, while it
    /// may still load), <c>--quit &lt;seconds&gt;</c> (after the first scene is ready), <c>--scene &lt;name&gt;</c> (start in that
    /// scene of the build's instead of its first).
    /// </summary>
    static class Program
    {
        [DllImport("user32.dll")]
        static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
        const uint PW_CLIENTONLY = 1, PW_RENDERFULLCONTENT = 2;

        static RenderForm form = null!;
        static RenderView view = null!;
        static SquidRenderer ui = null!;
        static StreamWriter? log;
        static readonly SquidKeyboard keyboard = new();

        static string gameFolder = AppContext.BaseDirectory;
        static bool windowed;
        static System.Drawing.Size windowSize = new(1600, 900);
        static double screenshotAfter = double.NaN, quitAfter = -1;
        static string? screenshotFile;
        static string? startSceneName;  // --scene: start in this scene of the game's instead of its first (tests)
        static readonly System.Diagnostics.Stopwatch sinceStart = new();    // from the start scene being ready
        static readonly System.Diagnostics.Stopwatch sinceWindow = new();   // from the first frame

        [STAThread]
        static int Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            ParseArgs(args);
            UserFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), ProductName());
            StartLog();

            try
            {
                Run();
                return 0;
            }
            catch (Exception ex)
            {
                Log($"[Player] Stopped by an error: {ex}");
                MessageBox.Show(ex.Message, "The game stopped", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
            finally
            {
                log?.Dispose();
            }
        }

        /// <summary>
        /// Where the game writes: %LOCALAPPDATA%\&lt;Product&gt;. Its own folder may be read-only (installed for all users
        /// under Program Files), and a write there threw UnauthorizedAccessException at the first line of the log.
        /// </summary>
        static string UserFolder = "";

        // No shader compiled yet for this game on this computer: its first start (or one after the cache was cleared)
        static bool FirstStart()
        {
            var cache = Path.Combine(UserFolder, "ShaderCache");
            try { return !Directory.Exists(cache) || !Directory.EnumerateFiles(cache, "*.cso").Any(); }
            catch (Exception) { return false; }
        }

        // What the window says while the engine starts compiling its shaders: plain WinForms text on the black window,
        // painted before Engine.Initialize blocks the thread, and taken away before the first frame
        static System.Windows.Forms.Label ShowPreparingNotice()
        {
            var notice = new System.Windows.Forms.Label
            {
                Text = "Preparing shaders for your graphics card\n\nThe first start takes a little longer. After that the game starts at once.",
                Dock = DockStyle.Fill,
                TextAlign = System.Drawing.ContentAlignment.MiddleCenter,
                ForeColor = System.Drawing.Color.FromArgb(204, 189, 158),   // the title's worn bone, no clean white
                BackColor = System.Drawing.Color.Black,
                Font = new System.Drawing.Font("Segoe UI", 15f),
            };
            form.Controls.Add(notice);
            form.Refresh();
            Log("[Player] First start: compiling the shaders");
            return notice;
        }

        // The product's name from the manifest, before anything else is read (the log goes into its folder); "Freefall"
        // when the manifest cannot be read, which Run then reports
        static string ProductName()
        {
            try
            {
                var product = PackManifest.Load(Path.Combine(gameFolder, "Data", PackManifest.FileName)).Product;
                return string.IsNullOrWhiteSpace(product) ? "Freefall" : product;
            }
            catch (Exception) { return "Freefall"; }
        }

        static void Run()
        {
            var dataFolder = Path.Combine(gameFolder, "Data");
            var manifest = PackManifest.Load(Path.Combine(dataFolder, PackManifest.FileName));
            var product = string.IsNullOrWhiteSpace(manifest.Product) ? "Freefall" : manifest.Product;
            if (manifest.Scenes.Count == 0)
                throw new InvalidDataException("The game has no scenes (its manifest lists none)");

            // Its own shader cache, not the editor's, and the engine's files from their pack: before the engine compiles its
            // first shader (Engine.Initialize)
            ShaderCache.SetRoot(Path.Combine(UserFolder, "ShaderCache"));
            if (!string.IsNullOrEmpty(manifest.EnginePack))
                EngineFiles.Mount(PackSet.Open(new[] { Path.Combine(dataFolder, manifest.EnginePack) }));

            Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
            form = new RenderForm { Text = product, BackColor = System.Drawing.Color.Black, StartPosition = FormStartPosition.CenterScreen };
            if (windowed) form.ClientSize = windowSize;
            else Fullscreen(true);
            form.Show();

            // The first start compiles every shader the engine starts with (~100, 11 s) before the window shows anything: it
            // says so meanwhile, in plain text, since nothing of the engine can draw yet
            var notice = FirstStart() ? ShowPreparingNotice() : null;

            Engine.Initialize(form.Handle, form.ClientSize.Width, form.ClientSize.Height, editor: false);
            view = RenderView.All[0];
            view.ClearColor = new Color4(0f, 0f, 0f, 1f);   // before the first scene, and while a load hides the world
            form.SizeChanged += (s, e) =>
            {
                if (form.WindowState == FormWindowState.Minimized) return;
                view.Resize(form.ClientSize.Width, form.ClientSize.Height);
            };
            Time.Initialize();

            ui = new SquidRenderer(Engine.Device);
            Squid.Gui.Renderer = ui;
            GameUI.ShowFps = true;   // until the game has options of its own for it
            var reported = new HashSet<string>();
            Squid.Gui.ControlException = (control, ex) =>
            {
                if (reported.Add(control.GetType().FullName + "|" + ex.Message))
                    Freefall.Debug.LogError("[GUI]", $"{control.GetType().Name} skipped: {ex}");
            };

            Engine.QuitRequested += () => form.Close();   // a menu's Quit
            Application.AddMessageFilter(new InputFilter());
            keyboard.Attach(form);
            form.KeyDown += (s, e) =>
            {
                if (e.Alt && e.KeyCode == System.Windows.Forms.Keys.Enter) { Fullscreen(form.FormBorderStyle != FormBorderStyle.None); e.Handled = true; }
            };

            Engine.OpenGame(dataFolder, manifest);
            LoadScripts();
            CommandBuffer.PrepareCuller(Engine.Device);
            CommandBuffer.InitializeCuller(Engine.Device);
            if (notice != null)
            {
                // From the first frame on the game says it itself (the loading screen, while shaders compile)
                form.Controls.Remove(notice);
                notice.Dispose();
                var stats = Shader.CacheStats;
                Log($"[Player] First start: {stats.Misses} shaders compiled before the first frame ({stats.CompileMs / 1000:0.0} s of DXC)");
            }

            // Every scene's cameras draw into the window (the start scene, and any game code loads later)
            SceneManager.Loaded += (scene, entities) =>
            {
                foreach (var camera in ComponentCache<Camera>.All)
                    camera.Target = view;
                Log($"[Player] Scene {scene?.Name}: {entities.Count} entities");
            };

            // The window shows from the first frame on while the start scene loads in slices: the game's UI draws over a
            // black picture (a loading screen of the game's, once one of its scenes has put it up), the half-made world
            // is not drawn. --screenshot and --quit count from the moment the start scene is ready.
            SceneManager.HideWorldWhileLoading = true;
            SceneAsset? start = null;
            if (startSceneName != null)
            {
                start = manifest.Scenes.Select(guid => Engine.Assets.LoadByGuid<SceneAsset>(guid))
                    .FirstOrDefault(scene => string.Equals(scene?.Name, startSceneName, StringComparison.OrdinalIgnoreCase));
                if (start == null) Log($"[Player] --scene {startSceneName}: the game has no such scene, starting with its first");
            }
            start ??= Engine.Assets.LoadByGuid<SceneAsset>(manifest.Scenes[0])
                ?? throw new InvalidDataException($"The start scene {manifest.Scenes[0]} is not in the packs");
            var load = SceneManager.LoadAsync(start);
            load.Completed += done =>
            {
                Log($"[Player] {done.Name} ready {sinceWindow.Elapsed.TotalSeconds:0.0} s after the window opened");
                sinceStart.Start();
            };
            view.OnAfterRender = DrawUI;
            sinceWindow.Start();

            RenderLoop.Run(form, Tick, true);

            Log($"[Player] Closed after {sinceStart.Elapsed.TotalSeconds:0.0} s; {Artifacts.OpenedFromPacks} artifacts read from the packs, " +
                $"{EngineFiles.ReadFromPack} engine files from {manifest.EnginePack ?? "the disk"}");
            // The GPU may still draw the last frames with the UI's buffers and pipeline: done with them first
            view.WaitForGpu();
            ui.Dispose();
            Engine.Shutdown();
        }

        /// <summary>The project's scripts (FreefallScripts.dll in the game folder), registered like the editor registers them; plugin DLLs resolve from the same folder.</summary>
        static void LoadScripts()
        {
            AppDomain.CurrentDomain.AssemblyResolve += (s, e) =>
            {
                var path = Path.Combine(gameFolder, new AssemblyName(e.Name).Name + ".dll");
                return File.Exists(path) ? Assembly.LoadFrom(path) : null;
            };

            var scripts = Path.Combine(gameFolder, "FreefallScripts.dll");
            if (!File.Exists(scripts))
            {
                Log("[Player] No FreefallScripts.dll: the game runs without its scripts");
                return;
            }
            var assembly = Assembly.LoadFrom(scripts);
            Freefall.Reflection.Reflector.RegisterAssemblies(assembly);
            Log($"[Player] Scripts: {assembly.GetName().Name}, {assembly.GetTypes().Length} types");
        }

        static void Tick()
        {
            Engine.Tick();

            double t = sinceStart.Elapsed.TotalSeconds;
            if (screenshotFile != null && !double.IsNaN(screenshotAfter)
                && (screenshotAfter >= 0 ? sinceStart.IsRunning && t >= screenshotAfter : sinceWindow.Elapsed.TotalSeconds >= -screenshotAfter))
            {
                screenshotAfter = double.NaN;
                Screenshot(screenshotFile);
            }
            if (quitAfter >= 0 && sinceStart.IsRunning && t >= quitAfter)
            {
                quitAfter = -1;
                form.Close();
            }
        }

        /// <summary>After the picture is in the back buffer: the game's UI (its desktop draws the marks game code asked for, Graphics.Overlay).</summary>
        static void DrawUI(RenderView target)
        {
            var cmd = target.CommandList;
            var size = new Squid.Point(target.Width, target.Height);

            // Squid's input: the window's mouse unless the game holds it (locked: nothing of the UI follows it)
            Squid.Gui.TimeElapsed = Time.DeltaMilliseconds;
            var mouse = form.PointToClient(Cursor.Position);
            if (Input.IsMouseLocked) Squid.Gui.SetMouse(-100000, -100000, 0);
            else Squid.Gui.SetMouse(mouse.X, mouse.Y, -Input.MouseWheelDelta);
            Squid.Gui.SetButtons((Control.MouseButtons & MouseButtons.Left) != 0, (Control.MouseButtons & MouseButtons.Right) != 0,
                                 (Control.MouseButtons & MouseButtons.Middle) != 0);
            keyboard.Feed();

            GameUI.Update();
            var desktop = GameUI.Desktop;
            desktop.Position = new Squid.Point(0, 0);
            desktop.Size = size;
            desktop.Update();

            cmd.SetRenderTargets(target.BackBufferTarget, target.DepthBufferTarget);
            cmd.SetViewport(new Viewport(0, 0, target.Width, target.Height, 0.0f, 1.0f));
            cmd.SetScissorRect(new RectI(0, 0, target.Width, target.Height));
            ui.SetContext(cmd.Native, target.Width, target.Height);

            // One desktop, one batch: the game UI's desktop draws the overlay's marks and the frame rate under its controls
            GameUI.Desktop.Draw();
        }

        /// <summary>Borderless over the whole screen the window is on, or back to a window.</summary>
        static void Fullscreen(bool on)
        {
            if (on)
            {
                form.FormBorderStyle = FormBorderStyle.None;
                form.Bounds = Screen.FromControl(form).Bounds;
            }
            else
            {
                form.FormBorderStyle = FormBorderStyle.Sizable;
                form.ClientSize = windowSize;
                var screen = Screen.FromControl(form).WorkingArea;
                form.Location = new System.Drawing.Point(screen.X + (screen.Width - form.Width) / 2, screen.Y + (screen.Height - form.Height) / 2);
            }
        }

        /// <summary>The window as it is on screen, game UI included (as the editor's screenshots), to a PNG.</summary>
        static void Screenshot(string file)
        {
            var client = form.ClientRectangle;
            using var bitmap = new Bitmap(client.Width, client.Height, PixelFormat.Format32bppArgb);
            using (var g = System.Drawing.Graphics.FromImage(bitmap))
            {
                var hdc = g.GetHdc();
                PrintWindow(form.Handle, hdc, PW_CLIENTONLY | PW_RENDERFULLCONTENT);
                g.ReleaseHdc(hdc);
            }
            var path = Path.GetFullPath(file);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            bitmap.Save(path, ImageFormat.Png);
            Log($"[Player] Screenshot {client.Width}x{client.Height} {sinceWindow.Elapsed.TotalSeconds:0.0} s after the window opened" +
                (sinceStart.IsRunning ? $", {sinceStart.Elapsed.TotalSeconds:0.0} s after the start scene was ready" : ", still loading") + $": {path}");
        }

        static void ParseArgs(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i].ToLowerInvariant())
                {
                    case "--game" when i + 1 < args.Length:
                        gameFolder = Path.GetFullPath(args[++i]);
                        break;
                    case "--windowed":
                        windowed = true;
                        if (i + 1 < args.Length && args[i + 1].Split('x') is [var w, var h] && int.TryParse(w, out int width) && int.TryParse(h, out int height))
                        {
                            windowSize = new System.Drawing.Size(width, height);
                            i++;
                        }
                        break;
                    case "--screenshot" when i + 2 < args.Length:
                        screenshotAfter = double.Parse(args[++i], CultureInfo.InvariantCulture);
                        screenshotFile = args[++i];
                        break;
                    case "--quit" when i + 1 < args.Length:
                        quitAfter = double.Parse(args[++i], CultureInfo.InvariantCulture);
                        break;
                    case "--scene" when i + 1 < args.Length:
                        startSceneName = args[++i];
                        break;
                }
            }
        }

        /// <summary>
        /// player.log in <see cref="UserFolder"/>: what the engine reports (warnings, errors, LogAlways) and the player's own
        /// lines. The last run's log is kept as player-prev.log. No log at all (the game runs on) when none can be written.
        /// </summary>
        static void StartLog()
        {
            try
            {
                Directory.CreateDirectory(UserFolder);
                var path = Path.Combine(UserFolder, "player.log");
                if (File.Exists(path)) File.Copy(path, Path.Combine(UserFolder, "player-prev.log"), overwrite: true);
                log = new StreamWriter(path, append: false) { AutoFlush = true };
                Freefall.Debug.OnLog += (message, _) => Log(message);
                AppDomain.CurrentDomain.UnhandledException += (s, e) => Log($"[Player] Unhandled: {e.ExceptionObject}");
                Log($"[Player] {DateTime.Now:yyyy-MM-dd HH:mm:ss}, game folder {gameFolder}, log {path}");
            }
            catch (Exception) { log = null; }   // read-only, full, or denied: an IOException is not the only one
        }

        static readonly object logLock = new();

        static void Log(string line)
        {
            lock (logLock)
                log?.WriteLine(line);
        }

        /// <summary>WM messages to the engine's Input: raw mouse movement, buttons, keys.</summary>
        sealed class InputFilter : IMessageFilter
        {
            public bool PreFilterMessage(ref System.Windows.Forms.Message m)
            {
                if (m.Msg == 0x00FF)   // WM_INPUT
                    Input.ProcessRawInput(m.LParam);
                Input.ProcessMessage((uint)m.Msg, (UIntPtr)(long)m.WParam, m.LParam);
                return false;
            }
        }
    }
}
