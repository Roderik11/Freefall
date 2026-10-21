
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
    /// toggles a window), starts the engine in game mode on the packs, loads the scripts, starts the build's first scene
    /// and draws the game's UI (GameUI, Graphics.Overlay) over the picture. Writes player.log next to the game.
    /// Development switches: <c>--game &lt;folder&gt;</c>, <c>--windowed [WxH]</c>, <c>--screenshot &lt;seconds&gt; &lt;file.png&gt;</c>
    /// (the window that long after the scene loaded), <c>--quit &lt;seconds&gt;</c>.
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
        static double screenshotAfter = -1, quitAfter = -1;
        static string? screenshotFile;
        static readonly System.Diagnostics.Stopwatch sinceStart = new();

        [STAThread]
        static int Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            ParseArgs(args);
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

        static void Run()
        {
            var dataFolder = Path.Combine(gameFolder, "Data");
            var manifest = PackManifest.Load(Path.Combine(dataFolder, PackManifest.FileName));
            var product = string.IsNullOrWhiteSpace(manifest.Product) ? "Freefall" : manifest.Product;
            if (manifest.Scenes.Count == 0)
                throw new InvalidDataException("The game has no scenes (its manifest lists none)");

            // Its own shader cache, not the editor's; before the engine compiles its first shader
            ShaderCache.SetRoot(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), product, "ShaderCache"));

            Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
            form = new RenderForm { Text = product, BackColor = System.Drawing.Color.Black, StartPosition = FormStartPosition.CenterScreen };
            if (windowed) form.ClientSize = windowSize;
            else Fullscreen(true);
            form.Show();

            Engine.Initialize(form.Handle, form.ClientSize.Width, form.ClientSize.Height, editor: false);
            view = RenderView.All[0];
            form.SizeChanged += (s, e) =>
            {
                if (form.WindowState == FormWindowState.Minimized) return;
                view.Resize(form.ClientSize.Width, form.ClientSize.Height);
            };
            Time.Initialize();

            ui = new SquidRenderer(Engine.Device);
            Squid.Gui.Renderer = ui;
            var reported = new HashSet<string>();
            Squid.Gui.ControlException = (control, ex) =>
            {
                if (reported.Add(control.GetType().FullName + "|" + ex.Message))
                    Freefall.Debug.LogError("[GUI]", $"{control.GetType().Name} skipped: {ex}");
            };

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

            // Every scene's cameras draw into the window (the start scene, and any game code loads later)
            SceneManager.Loaded += (scene, entities) =>
            {
                foreach (var camera in ComponentCache<Camera>.All)
                    camera.Target = view;
                Log($"[Player] Scene {scene?.Name}: {entities.Count} entities");
            };

            var start = Engine.Assets.LoadByGuid<SceneAsset>(manifest.Scenes[0])
                ?? throw new InvalidDataException($"The start scene {manifest.Scenes[0]} is not in the packs");
            SceneManager.LoadNow(start);
            view.OnAfterRender = DrawUI;
            sinceStart.Start();

            RenderLoop.Run(form, Tick, true);

            Log($"[Player] Closed after {sinceStart.Elapsed.TotalSeconds:0.0} s; {Artifacts.OpenedFromPacks} artifacts read from the packs");
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
            if (screenshotFile != null && screenshotAfter >= 0 && t >= screenshotAfter)
            {
                screenshotAfter = -1;
                Screenshot(screenshotFile);
            }
            if (quitAfter >= 0 && t >= quitAfter)
            {
                quitAfter = -1;
                form.Close();
            }
        }

        /// <summary>After the picture is in the back buffer: the marks game code asked for (Graphics.Overlay), then the game's UI.</summary>
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

            ui.StartBatch();
            Overlay.Draw(ui, new Squid.Point(0, 0), size);
            ui.EndBatch(true);
            if (GameUI.Running) GameUI.Desktop.Draw();
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
            Log($"[Player] Screenshot {client.Width}x{client.Height} at {sinceStart.Elapsed.TotalSeconds:0.0} s: {path}");
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
                }
            }
        }

        /// <summary>player.log next to the game: what the engine reports (warnings, errors, LogAlways) and the player's own lines.</summary>
        static void StartLog()
        {
            try
            {
                log = new StreamWriter(Path.Combine(gameFolder, "player.log"), append: false) { AutoFlush = true };
                Freefall.Debug.OnLog += (message, _) => Log(message);
                AppDomain.CurrentDomain.UnhandledException += (s, e) => Log($"[Player] Unhandled: {e.ExceptionObject}");
                Log($"[Player] {DateTime.Now:yyyy-MM-dd HH:mm:ss}, game folder {gameFolder}");
            }
            catch (IOException) { log = null; }
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
