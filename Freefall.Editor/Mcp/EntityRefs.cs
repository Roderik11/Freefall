using System.Collections.Generic;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;

namespace Freefall.Editor.Mcp
{
    /// <summary>
    /// Tools address entities by runtime id (changes on every scene load) or by persistent UID (saved in the scene).
    /// UIDs travel as strings: they exceed 2^53, which JSON clients parsing numbers as doubles would round.
    /// </summary>
    internal static class EntityRefs
    {
        public const string UidHelp = "Persistent entity UID (string, from any entity result) — survives scene reloads; use instead of 'id'";

        /// <summary>The current runtime id for (id, uid); uid wins when both are given. Error result when unresolvable.</summary>
        public static async Task<(int? id, CallToolResult? error)> Resolve(int? id, string? uid, bool required = true)
        {
            if (!string.IsNullOrWhiteSpace(uid))
            {
                if (!ulong.TryParse(uid.Trim(), out var u))
                    return (null, McpBridge.Error($"'{uid}' is not a UID (a decimal string)."));
                var found = await EditorCommandServer.Instance.RunOnMainThread(() => Commands.CommandHelpers.FindEntityByUid(u)?.Id);
                return found == null
                    ? (null, McpBridge.Error($"No entity with UID {uid} in the open scene."))
                    : (found, null);
            }
            if (id == null && required) return (null, McpBridge.Error("Pass 'id' or 'uid'."));
            return (id, null);
        }

        /// <summary>Runtime ids for a mixed list of ids and UIDs.</summary>
        public static async Task<(List<int>? ids, CallToolResult? error)> ResolveMany(int[]? ids, string[]? uids)
        {
            var result = new List<int>(ids ?? []);
            foreach (var uid in uids ?? [])
            {
                var (id, error) = await Resolve(null, uid);
                if (error != null) return (null, error);
                result.Add(id!.Value);
            }
            return (result, null);
        }
    }
}
