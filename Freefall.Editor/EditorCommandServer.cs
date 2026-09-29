using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Freefall.Base;
using Freefall.Editor.Commands;
using Freefall.Editor.Mcp;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Freefall.Editor
{
    /// <summary>
    /// AI-agent command server for the editor. Hosts Kestrel on localhost:21721 with:
    ///   /mcp        — Model Context Protocol endpoint (Streamable HTTP, stateless). Tools live in Mcp/.
    ///   /api/...    — plain REST pass-through to the same EditorCommand routes, for curl/scripts.
    ///
    /// Requests arrive on thread-pool threads; all editor/engine work is marshalled onto the main
    /// thread via RunOnMainThread, drained once per frame by ProcessCommands().
    ///
    /// Commands are auto-discovered via [CommandRoute] attributes on EditorCommand subclasses.
    /// </summary>
    public class EditorCommandServer : IDisposable
    {
        public static EditorCommandServer Instance { get; private set; }

        public const int Port = 21721;
        private static readonly TimeSpan MainThreadTimeout = TimeSpan.FromSeconds(60);

        private readonly ConcurrentQueue<Action> _mainThreadQueue = new();
        private readonly List<RouteEntry> _routes = new();
        private WebApplication _app;

        public EditorCommandServer()
        {
            Instance = this;
            DiscoverCommands();
        }

        /// <summary>
        /// Scan the editor assembly for all [CommandRoute] classes and register them.
        /// </summary>
        private void DiscoverCommands()
        {
            var assembly = Assembly.GetExecutingAssembly();
            var commandTypes = assembly.GetTypes()
                .Where(t => t.IsSubclassOf(typeof(EditorCommand)) && !t.IsAbstract);

            foreach (var type in commandTypes)
            {
                var attr = type.GetCustomAttribute<CommandRouteAttribute>();
                if (attr == null) continue;

                var instance = (EditorCommand)Activator.CreateInstance(type);
                var entry = new RouteEntry(attr.Method, attr.Pattern, instance);
                _routes.Add(entry);
            }

            // Sort routes: static routes before parameterized ones for correct matching
            _routes.Sort((a, b) => a.HasParams.CompareTo(b.HasParams));

            Debug.Log($"[CommandServer] Discovered {_routes.Count} commands");
        }

        public void Start()
        {
            try
            {
                var builder = WebApplication.CreateBuilder(new WebApplicationOptions
                {
                    ContentRootPath = AppContext.BaseDirectory
                });

                builder.Logging.ClearProviders();
                builder.WebHost.ConfigureKestrel(o => o.ListenLocalhost(Port));

                builder.Services
                    .AddMcpServer(o =>
                    {
                        o.ServerInfo = new() { Name = "freefall-editor", Version = "1.0.0" };
                        o.ServerInstructions = McpInstructions.Text;
                    })
                    .WithHttpTransport(o => o.Stateless = true)
                    .WithToolsFromAssembly(typeof(EditorCommandServer).Assembly, McpJson.Options);

                _app = builder.Build();

                // Localhost binding alone doesn't stop a browser page from POSTing here
                // (DNS rebinding / cross-site requests). Reject any foreign Origin.
                _app.Use(async (ctx, next) =>
                {
                    var origin = ctx.Request.Headers.Origin.ToString();
                    if (origin.Length > 0 && !IsLocalOrigin(origin))
                    {
                        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
                        return;
                    }
                    await next();
                });

                _app.MapMcp("/mcp");
                _app.Map("/api/{**path}", HandleRestAsync);

                _app.StartAsync().GetAwaiter().GetResult();
                Debug.Log($"[CommandServer] Listening on http://localhost:{Port}/ (MCP at /mcp, REST at /api)");
            }
            catch (Exception ex)
            {
                Debug.LogWarning("CommandServer", $"Failed to start: {ex.Message}");
                _app = null;
            }
        }

        public void Stop()
        {
            if (_app == null) return;
            try { _app.StopAsync().Wait(TimeSpan.FromSeconds(2)); } catch { }
            _app = null;
            Debug.Log("[CommandServer] Stopped.");
        }

        private static bool IsLocalOrigin(string origin)
        {
            return Uri.TryCreate(origin, UriKind.Absolute, out var uri)
                && (uri.IsLoopback || uri.Host == "localhost");
        }

        // --- Main-thread marshalling ---

        /// <summary>
        /// Called on the main thread each tick (from Program.RenderGui).
        /// Runs everything queued by RunOnMainThread.
        /// </summary>
        public void ProcessCommands()
        {
            while (_mainThreadQueue.TryDequeue(out var action))
                action();
        }

        /// <summary>
        /// Run a function on the editor main thread and await its result. Safe to call from any thread.
        /// </summary>
        public Task<T> RunOnMainThread<T>(Func<T> func)
        {
            var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            _mainThreadQueue.Enqueue(() =>
            {
                if (tcs.Task.IsCompleted) return; // timed out already
                try { tcs.TrySetResult(func()); }
                catch (Exception ex) { tcs.TrySetException(ex); }
            });
            return tcs.Task.WaitAsync(MainThreadTimeout);
        }

        /// <summary>
        /// Dispatch a route on the main thread (IAsyncEditorCommands run their own async path instead).
        /// Exceptions become 500 results.
        /// </summary>
        public async Task<CommandResult> DispatchAsync(string method, string pathAndQuery, string? body)
        {
            try
            {
                // Route matching touches no engine state, so it is safe off the main thread.
                var command = MatchRoute(method, pathAndQuery, body, out var context);
                if (command == null)
                    return NoRoute(method, pathAndQuery);
                if (command is IAsyncEditorCommand asyncCommand)
                    return await asyncCommand.ExecuteAsync(context);
                return await RunOnMainThread(() => command.Execute(context));
            }
            catch (TimeoutException)
            {
                return CommandResult.Error(504, "Command timed out waiting for the editor main thread");
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, ex.Message);
            }
        }

        // --- REST pass-through ---

        private async Task HandleRestAsync(HttpContext ctx)
        {
            string body = null;
            if (ctx.Request.ContentLength > 0 || ctx.Request.Headers.TransferEncoding.Count > 0)
            {
                using var reader = new StreamReader(ctx.Request.Body, Encoding.UTF8);
                body = await reader.ReadToEndAsync();
            }

            var pathAndQuery = ctx.Request.Path.Value + ctx.Request.QueryString.Value;
            var result = await DispatchAsync(ctx.Request.Method, pathAndQuery, body);

            ctx.Response.StatusCode = result.StatusCode;
            ctx.Response.ContentType = result.ContentType;

            if (result.BinaryBody != null)
                await ctx.Response.Body.WriteAsync(result.BinaryBody);
            else if (result.Body != null)
                await ctx.Response.WriteAsync(result.Body, Encoding.UTF8);
        }

        // --- Route matching ---

        /// <summary>Synchronous dispatch; main thread only.</summary>
        internal CommandResult DispatchCommand(string method, string pathAndQuery, string body)
        {
            var command = MatchRoute(method, pathAndQuery, body, out var context);
            return command != null ? command.Execute(context) : NoRoute(method, pathAndQuery);
        }

        private EditorCommand? MatchRoute(string method, string pathAndQuery, string body, out CommandContext context)
        {
            // Strip query string for route matching, but preserve full path for commands
            var routePath = RoutePath(pathAndQuery);

            foreach (var route in _routes)
            {
                if (route.TryMatch(method, routePath, out var parameters))
                {
                    context = new CommandContext
                    {
                        Path = pathAndQuery,
                        Body = body,
                        Params = parameters
                    };
                    return route.Command;
                }
            }

            context = null;
            return null;
        }

        private static string RoutePath(string pathAndQuery)
        {
            var qIdx = pathAndQuery.IndexOf('?');
            return qIdx >= 0 ? pathAndQuery.Substring(0, qIdx) : pathAndQuery;
        }

        private static CommandResult NoRoute(string method, string pathAndQuery)
            => CommandResult.NotFound($"No command registered for {method} {RoutePath(pathAndQuery)}");

        // --- Inner types ---

        /// <summary>
        /// A registered route with its pattern, compiled regex, and command instance.
        /// </summary>
        private class RouteEntry
        {
            public string Method { get; }
            public string Pattern { get; }
            public EditorCommand Command { get; }
            public bool HasParams { get; }

            private readonly Regex _regex;
            private readonly string[] _paramNames;

            public RouteEntry(string method, string pattern, EditorCommand command)
            {
                Method = method;
                Pattern = pattern;
                Command = command;

                // Extract parameter names and build regex
                // "/api/entity/{id}/transform" → regex: ^/api/entity/(?<id>[^/]+)/transform$
                var paramList = new List<string>();
                var regexPattern = "^" + Regex.Replace(pattern, @"\{(\w+)\}", m =>
                {
                    paramList.Add(m.Groups[1].Value);
                    return $"(?<{m.Groups[1].Value}>[^/]+)";
                }) + "$";

                _regex = new Regex(regexPattern, RegexOptions.Compiled);
                _paramNames = paramList.ToArray();
                HasParams = _paramNames.Length > 0;
            }

            public bool TryMatch(string method, string path, out Dictionary<string, string> parameters)
            {
                parameters = null;

                if (!string.Equals(Method, method, StringComparison.OrdinalIgnoreCase))
                    return false;

                var match = _regex.Match(path);
                if (!match.Success)
                    return false;

                parameters = new Dictionary<string, string>();
                foreach (var name in _paramNames)
                    parameters[name] = match.Groups[name].Value;

                return true;
            }
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
