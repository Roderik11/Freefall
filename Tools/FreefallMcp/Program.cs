// Freefall MCP bridge (stdio).
//
// The editor hosts MCP over HTTP at localhost:21721/mcp, but an HTTP MCP server that isn't up when the
// client starts is simply marked failed. This bridge is launched by the client instead, so it is always
// connected:
//   - tools/list  → the editor's live tools, or the last list cached on disk while the editor is closed
//   - tools/call  → forwarded to the editor; a clear "not running, call editor_launch" error while it is down
//   - editor_launch (bridge tool) → starts the editor detached, waits until it answers, optionally opens a project
//   - a background poll sends notifications/tools/list_changed when the editor's tool set changes
//
// Args: --url <editor mcp url>   (default http://127.0.0.1:21721/mcp)
//       --editor <Freefall.Editor.exe> (default: <repo>/Freefall.Editor/bin/Release/net10.0-windows/Freefall.Editor.exe)

using System.Diagnostics;
using System.Management;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

var url = new Uri(ArgValue(args, "--url") ?? "http://127.0.0.1:21721/mcp");
var editorExe = ArgValue(args, "--editor") ?? DefaultEditorPath();

var bridge = new EditorBridge(url, editorExe);
bridge.LoadCache();

var builder = Host.CreateApplicationBuilder();
// stdout carries the protocol — every log line must go to stderr.
builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);
builder.Logging.SetMinimumLevel(LogLevel.Warning);

builder.Services
    .AddMcpServer(o =>
    {
        o.ServerInfo = new() { Name = "freefall-editor-bridge", Version = "1.0.0" };
        o.ServerInstructions = bridge.Instructions;
        o.Capabilities = new() { Tools = new() { ListChanged = true } };
    })
    .WithStdioServerTransport()
    .WithListToolsHandler(async (ctx, ct) =>
    {
        bridge.AttachServer(ctx.Server);
        await bridge.RefreshAsync(notify: false, ct);
        return new ListToolsResult { Tools = bridge.AllTools() };
    })
    .WithCallToolHandler(async (ctx, ct) =>
    {
        bridge.AttachServer(ctx.Server);
        var p = ctx.Params!;
        return p.Name == EditorBridge.LaunchToolName
            ? await bridge.LaunchAsync(p.Arguments, ct)
            : await bridge.CallAsync(p.Name, p.Arguments, ct);
    });

var app = builder.Build();
_ = bridge.PollLoopAsync(app.Services.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping);
await app.RunAsync();

static string? ArgValue(string[] args, string name)
{
    var i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

static string DefaultEditorPath()
{
    // The bridge runs from an installed copy (see FreefallMcp.csproj), so the repo root is baked in at build time.
    var root = typeof(EditorBridge).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
        .Cast<System.Reflection.AssemblyMetadataAttribute>()
        .FirstOrDefault(a => a.Key == "RepoRoot")?.Value ?? Directory.GetCurrentDirectory();
    return Path.Combine(root, "Freefall.Editor", "bin", "Release", "net10.0-windows", "Freefall.Editor.exe");
}

sealed class EditorBridge(Uri mcpUrl, string editorExe)
{
    public const string LaunchToolName = "editor_launch";

    private static readonly JsonSerializerOptions Json = McpJsonUtilities.DefaultOptions;
    private static readonly string CachePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Freefall", "mcp-bridge-cache.json");

    private readonly SemaphoreSlim _clientLock = new(1, 1);
    private readonly SemaphoreSlim _refreshLock = new(1, 1); // poll loop and editor_launch both refresh
    private readonly HttpClient _ping = new() { Timeout = TimeSpan.FromSeconds(2) };
    private McpClient? _client;
    private McpServer? _server;
    private IList<Tool> _editorTools = [];
    private string _toolsHash = "";
    private volatile bool _editorUp;

    public string Instructions { get; private set; } = DefaultInstructions;

    private const string DefaultInstructions =
        "Controls the running Freefall editor. If a call reports the editor is not running, call editor_launch.";

    private const string BridgeNote =
        "\n\nBridge: the tool list is cached while the editor is closed; calls then fail until editor_launch (or the user) starts it.";

    // --- tool list ---

    public IList<Tool> AllTools() => [LaunchTool, .. _editorTools];

    private static readonly Tool LaunchTool = new()
    {
        Name = LaunchToolName,
        Title = "Launch the editor",
        Description = "Start the Freefall editor if it isn't running and wait until it responds (shader compilation can take a " +
                      "minute). Optionally open a project afterwards. Use when other tools report that the editor is not running.",
        InputSchema = JsonSerializer.Deserialize<JsonElement>("""
            {
              "type": "object",
              "properties": {
                "project": { "type": "string", "description": "Project root directory to open once the editor is up (optional)" },
                "timeoutSeconds": { "type": "integer", "default": 180, "description": "Max seconds to wait for the editor to respond" }
              }
            }
            """),
        Annotations = new() { ReadOnlyHint = false, DestructiveHint = false, IdempotentHint = true, OpenWorldHint = false },
    };

    public void AttachServer(McpServer server) => _server ??= server;

    // --- editor connection ---

    private async Task<McpClient?> GetClientAsync(CancellationToken ct)
    {
        if (_client != null) return _client;
        await _clientLock.WaitAsync(ct);
        try
        {
            if (_client != null) return _client;
            var transport = new HttpClientTransport(
                new HttpClientTransportOptions
                {
                    Endpoint = mcpUrl,
                    TransportMode = HttpTransportMode.StreamableHttp,
                    ConnectionTimeout = TimeSpan.FromSeconds(3),
                },
                // Long-running editor tools (project_open, import_unity_pack) outlive HttpClient's 100 s default.
                new HttpClient { Timeout = Timeout.InfiniteTimeSpan },
                loggerFactory: null,
                ownsHttpClient: true);
            _client = await McpClient.CreateAsync(transport, cancellationToken: ct);
            if (!string.IsNullOrEmpty(_client.ServerInstructions))
                Instructions = _client.ServerInstructions + BridgeNote;
            return _client;
        }
        catch (Exception) when (!ct.IsCancellationRequested)
        {
            return null;
        }
        finally
        {
            _clientLock.Release();
        }
    }

    private async Task DropClientAsync()
    {
        var c = Interlocked.Exchange(ref _client, null);
        if (c != null) try { await c.DisposeAsync(); } catch { }
    }

    private enum EditorState { Up, Down, Busy }

    /// <summary>
    /// /api/ping runs on the editor main thread, so a timeout (connection accepted, no answer) means the editor
    /// is running but its main thread is stuck — different advice than "not running".
    /// </summary>
    private async Task<EditorState> ProbeAsync(CancellationToken ct)
    {
        try
        {
            var ping = new Uri(mcpUrl, "/api/ping");
            using var r = await _ping.GetAsync(ping, ct);
            return r.IsSuccessStatusCode ? EditorState.Up : EditorState.Busy;
        }
        catch (TaskCanceledException) when (!ct.IsCancellationRequested)
        {
            return EditorState.Busy;
        }
        catch (Exception) when (!ct.IsCancellationRequested)
        {
            return EditorState.Down;
        }
    }

    private async Task<bool> PingAsync(CancellationToken ct) => await ProbeAsync(ct) == EditorState.Up;

    /// <summary>Fetch the live tool list; on change update the cache and (optionally) notify the client.</summary>
    public async Task<bool> RefreshAsync(bool notify, CancellationToken ct)
    {
        if (!await PingAsync(ct))
        {
            _editorUp = false;
            return false;
        }

        var client = await GetClientAsync(ct);
        if (client == null) return false;

        await _refreshLock.WaitAsync(ct);
        try
        {
            var tools = (await client.ListToolsAsync(cancellationToken: ct)).Select(t => t.ProtocolTool).ToList();
            _editorUp = true;
            var hash = Hash(tools);
            if (hash == _toolsHash) return true;

            _editorTools = tools;
            _toolsHash = hash;
            SaveCache();
            if (notify && _server != null)
                await _server.SendNotificationAsync(NotificationMethods.ToolListChangedNotification, ct);
            return true;
        }
        catch (Exception) when (!ct.IsCancellationRequested)
        {
            _editorUp = false;
            await DropClientAsync();
            return false;
        }
        finally
        {
            _refreshLock.Release();
        }
    }

    public async Task PollLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await RefreshAsync(notify: true, ct);
                await Task.Delay(_editorUp ? TimeSpan.FromSeconds(15) : TimeSpan.FromSeconds(3), ct);
            }
            catch (OperationCanceledException) { break; }
            catch { /* keep polling */ }
        }
    }

    // --- calls ---

    public async Task<CallToolResult> CallAsync(string name, IDictionary<string, JsonElement>? arguments, CancellationToken ct)
    {
        var state = await ProbeAsync(ct);
        if (state == EditorState.Busy)
            return Error("The editor is running but not responding: its main thread is busy or hung (long operation, " +
                         "modal dialog, or a freeze). Wait and retry, or check the editor window.");
        var client = state == EditorState.Up ? await GetClientAsync(ct) : null;
        if (client == null)
            return NotRunning();

        try
        {
            return await client.CallToolAsync(new CallToolRequestParams { Name = name, Arguments = arguments }, ct);
        }
        catch (McpException ex)
        {
            return Error($"Editor rejected the call: {ex.Message}");
        }
        catch (Exception ex) when (!ct.IsCancellationRequested)
        {
            await DropClientAsync();
            // The request may or may not have executed — don't retry blindly.
            return await PingAsync(ct)
                ? Error($"Call failed: {ex.Message}")
                : Error("The editor stopped responding during the call (closed or crashed?). " +
                        "Check with editor_status after editor_launch before retrying — the action may have been applied.");
        }
    }

    public async Task<CallToolResult> LaunchAsync(IDictionary<string, JsonElement>? arguments, CancellationToken ct)
    {
        string? project = arguments != null && arguments.TryGetValue("project", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString() : null;
        int timeout = arguments != null && arguments.TryGetValue("timeoutSeconds", out var t) && t.TryGetInt32(out var ts) ? ts : 180;

        var log = new StringBuilder();
        var state = await ProbeAsync(ct);
        if (state == EditorState.Busy)
        {
            return Error("An editor instance is running but not responding (main thread busy or hung). " +
                         "Not starting a second instance — wait, or ask the user to close the frozen one.");
        }
        if (state == EditorState.Up)
        {
            log.AppendLine("Editor is already running.");
        }
        else
        {
            if (!File.Exists(editorExe))
                return Error($"Editor executable not found: {editorExe}. Build Freefall.Editor (Release) first.");

            StartDetached(editorExe);
            log.AppendLine($"Started {editorExe}.");

            var deadline = DateTime.UtcNow.AddSeconds(timeout);
            while (!await PingAsync(ct))
            {
                if (DateTime.UtcNow > deadline)
                    return Error(log + $"Editor did not respond within {timeout}s. It may still be compiling shaders — retry editor_launch, or check the editor window.");
                await Task.Delay(1000, ct);
            }
            log.AppendLine("Editor is responding.");
        }

        await RefreshAsync(notify: true, ct);

        if (project != null)
        {
            var open = await CallAsync("project_open", new Dictionary<string, JsonElement>
            {
                ["path"] = JsonSerializer.SerializeToElement(project),
            }, ct);
            var text = open.Content.OfType<TextContentBlock>().FirstOrDefault()?.Text ?? "";
            if (open.IsError == true)
                return Error(log + "Opening the project failed: " + text);
            log.AppendLine("Project: " + text);
        }

        return new CallToolResult { Content = [new TextContentBlock { Text = log.ToString().TrimEnd() }] };
    }

    /// <summary>
    /// Start the editor outside our process tree (via WMI), so closing the MCP client — which kills the
    /// bridge and possibly its children — never takes the editor and its unsaved work down with it.
    /// </summary>
    private static void StartDetached(string exe)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(exe))!;
        try
        {
            using var processClass = new ManagementClass("Win32_Process");
            using var startup = new ManagementClass("Win32_ProcessStartup").CreateInstance();
            startup["ShowWindow"] = (ushort)1; // SW_SHOWNORMAL
            using var inParams = processClass.GetMethodParameters("Create");
            inParams["CommandLine"] = $"\"{Path.GetFullPath(exe)}\"";
            inParams["CurrentDirectory"] = dir;
            inParams["ProcessStartupInformation"] = startup;
            using var result = processClass.InvokeMethod("Create", inParams, null);
            if (Convert.ToUInt32(result["ReturnValue"]) == 0) return;
        }
        catch { /* fall back below */ }

        Process.Start(new ProcessStartInfo(exe) { WorkingDirectory = dir, UseShellExecute = true });
    }

    // --- cache ---

    public void LoadCache()
    {
        try
        {
            if (!File.Exists(CachePath)) return;
            var node = JsonNode.Parse(File.ReadAllText(CachePath))!;
            _editorTools = node["tools"]!.Deserialize<List<Tool>>(Json) ?? [];
            _toolsHash = Hash(_editorTools);
            var instructions = node["instructions"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(instructions)) Instructions = instructions;
        }
        catch { _editorTools = []; }
    }

    private void SaveCache()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(CachePath)!);
            var node = new JsonObject
            {
                ["instructions"] = Instructions,
                ["tools"] = JsonSerializer.SerializeToNode(_editorTools, Json),
            };
            File.WriteAllText(CachePath, node.ToJsonString());
        }
        catch { /* cache is best effort */ }
    }

    private static string Hash(IList<Tool> tools)
        => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(tools, Json))));

    // --- results ---

    private CallToolResult NotRunning() => Error(
        $"The Freefall editor is not running (nothing answers at {mcpUrl}). " +
        "Call editor_launch to start it, or ask the user to start it.");

    private static CallToolResult Error(string message) => new()
    {
        IsError = true,
        Content = [new TextContentBlock { Text = message }],
    };
}
