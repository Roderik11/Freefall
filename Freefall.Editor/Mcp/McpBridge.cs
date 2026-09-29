using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;

namespace Freefall.Editor.Mcp
{
    public static class McpJson
    {
        /// <summary>Used for tool argument schemas/binding and for building route bodies.</summary>
        public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
        {
            // The SDK marks these options read-only, which requires an explicit resolver.
            // Its own resolver first (protocol types), reflection for our argument records.
            TypeInfoResolver = System.Text.Json.Serialization.Metadata.JsonTypeInfoResolver.Combine(
                ModelContextProtocol.McpJsonUtilities.DefaultOptions.TypeInfoResolver,
                new System.Text.Json.Serialization.Metadata.DefaultJsonTypeInfoResolver()),
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
        };
    }

    /// <summary>
    /// Forwards MCP tool calls to the existing EditorCommand routes on the main thread and turns
    /// the CommandResult into a tool result. Non-2xx results become isError tool results so the
    /// model sees the message instead of a protocol failure.
    /// </summary>
    public static class McpBridge
    {
        public static Task<CallToolResult> Get(string path, params (string key, object? value)[] query)
            => Call("GET", path + QueryString(query), null);

        public static Task<CallToolResult> Post(string path, object? body = null)
            => Call("POST", path, body);

        public static async Task<CallToolResult> Call(string method, string path, object? body)
        {
            var server = EditorCommandServer.Instance
                ?? throw new InvalidOperationException("Editor command server is not running.");

            string? json = body switch
            {
                null => null,
                string s => s,
                _ => JsonSerializer.Serialize(body, McpJson.Options),
            };

            var result = await server.DispatchAsync(method, path, json);
            return ToToolResult(result);
        }

        public static CallToolResult ToToolResult(Commands.CommandResult result)
        {
            bool ok = result.StatusCode is >= 200 and < 300;
            string text = result.Body ?? (result.BinaryBody != null ? $"<{result.BinaryBody.Length} bytes {result.ContentType}>" : "");
            if (!ok) text = $"HTTP {result.StatusCode}: {text}";

            return new CallToolResult
            {
                IsError = !ok,
                Content = [new TextContentBlock { Text = text }],
            };
        }

        public static CallToolResult Error(string message) => new()
        {
            IsError = true,
            Content = [new TextContentBlock { Text = message }],
        };

        public static string Esc(string s) => Uri.EscapeDataString(s);

        private static string QueryString((string key, object? value)[] query)
        {
            var parts = query
                .Where(q => q.value != null && !(q.value is string s && s.Length == 0))
                .Select(q => $"{Uri.EscapeDataString(q.key)}={Uri.EscapeDataString(Format(q.value!))}")
                .ToList();
            return parts.Count == 0 ? "" : "?" + string.Join("&", parts);
        }

        private static string Format(object v) => v switch
        {
            bool b => b ? "true" : "false",
            IFormattable f => f.ToString(null, System.Globalization.CultureInfo.InvariantCulture),
            _ => v.ToString() ?? "",
        };
    }
}
