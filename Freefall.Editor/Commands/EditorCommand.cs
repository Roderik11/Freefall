using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Marks a class as a command handler and defines its HTTP route.
    /// Supports path parameters via {paramName} syntax, e.g. "/api/scene/entity/{id}".
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public class CommandRouteAttribute : Attribute
    {
        public string Method { get; }
        public string Pattern { get; }

        public CommandRouteAttribute(string method, string pattern)
        {
            Method = method.ToUpperInvariant();
            Pattern = pattern;
        }
    }

    /// <summary>
    /// Context passed to command handlers. Contains the parsed request
    /// and extracted path parameters from the route pattern.
    /// </summary>
    public class CommandContext
    {
        /// <summary>Original request path.</summary>
        public string Path { get; init; }

        /// <summary>Raw request body (null if none).</summary>
        public string Body { get; init; }

        /// <summary>Path parameters extracted from route pattern, e.g. {id} → "12345".</summary>
        public Dictionary<string, string> Params { get; init; } = new();

        /// <summary>Parse the body as JSON.</summary>
        public JsonDocument ParseBody() => JsonDocument.Parse(Body ?? "{}");

        /// <summary>Get a path parameter as ulong.</summary>
        public ulong GetULong(string name)
        {
            if (Params.TryGetValue(name, out var val) && ulong.TryParse(val, out var result))
                return result;
            throw new ArgumentException($"Missing or invalid path parameter: {name}");
        }

        /// <summary>Get a path parameter as int.</summary>
        public int GetInt(string name)
        {
            if (Params.TryGetValue(name, out var val) && int.TryParse(val, out var result))
                return result;
            throw new ArgumentException($"Missing or invalid path parameter: {name}");
        }

        /// <summary>Get a path parameter as string.</summary>
        public string GetString(string name)
        {
            if (Params.TryGetValue(name, out var val))
                return val;
            throw new ArgumentException($"Missing path parameter: {name}");
        }
    }

    /// <summary>
    /// Result returned by command handlers.
    /// </summary>
    public class CommandResult
    {
        public int StatusCode { get; set; } = 200;
        public string ContentType { get; set; } = "application/json";
        public string Body { get; set; }
        public byte[] BinaryBody { get; set; }

        public static CommandResult Json(object obj)
        {
            return new CommandResult
            {
                StatusCode = 200,
                ContentType = "application/json",
                Body = JsonSerializer.Serialize(obj, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                })
            };
        }

        public static CommandResult Error(int code, string message)
        {
            return new CommandResult
            {
                StatusCode = code,
                ContentType = "application/json",
                Body = JsonSerializer.Serialize(new { error = message })
            };
        }

        public static CommandResult NotFound(string message) => Error(404, message);
        public static CommandResult BadRequest(string message) => Error(400, message);
    }

    /// <summary>
    /// Base class for all editor commands. Subclass this and add a [CommandRoute]
    /// attribute to register a new endpoint. Commands are auto-discovered via reflection.
    /// </summary>
    public abstract class EditorCommand
    {
        public abstract CommandResult Execute(CommandContext context);
    }

    /// <summary>
    /// A command that must span several frames (e.g. screenshot waits for a cue to play). When dispatched
    /// through EditorCommandServer.DispatchAsync, ExecuteAsync runs on a thread-pool thread instead of the
    /// main thread — marshal engine work yourself. Execute remains the synchronous main-thread fallback.
    /// </summary>
    public interface IAsyncEditorCommand
    {
        Task<CommandResult> ExecuteAsync(CommandContext context);
    }
}
