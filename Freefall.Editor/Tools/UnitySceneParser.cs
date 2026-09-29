using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text.RegularExpressions;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Parses a Unity .unity scene file (YAML) and extracts PrefabInstance data:
    /// prefab GUID, display name, and world-space transform.
    /// </summary>
    public class UnitySceneParser
    {
        /// <summary>
        /// A prefab instance extracted from the Unity scene.
        /// </summary>
        public class PrefabInstance
        {
            public string PrefabGuid;       // Unity GUID from m_SourcePrefab
            public string DisplayName;      // from m_Name property modification
            public Vector3 Position;        // world-space
            public Quaternion Rotation;     // world-space
            public Vector3 Scale;           // world-space (lossy)
        }

        /// <summary>
        /// Internal transform record for parent chain resolution.
        /// </summary>
        private class TransformRecord
        {
            public long FileID;
            public long ParentFileID;
            public Vector3 LocalPosition;
            public Quaternion LocalRotation = Quaternion.Identity;
            public Vector3 LocalScale = Vector3.One;
        }

        // Regex patterns
        private static readonly Regex DocumentStartRegex = new(@"^--- !u!(\d+) &(\d+)", RegexOptions.Compiled);
        private static readonly Regex GuidRegex = new(@"guid:\s*([0-9a-f]{32})", RegexOptions.Compiled);
        private static readonly Regex FileIdRegex = new(@"fileID:\s*(\d+)", RegexOptions.Compiled);
        private static readonly Regex FloatValueRegex = new(@"value:\s*(-?[\d.eE+-]+)", RegexOptions.Compiled);
        private static readonly Regex InlineFloatRegex = new(@"{\s*x:\s*(-?[\d.eE+-]+),\s*y:\s*(-?[\d.eE+-]+),\s*z:\s*(-?[\d.eE+-]+)(?:,\s*w:\s*(-?[\d.eE+-]+))?\s*}", RegexOptions.Compiled);

        /// <summary>
        /// Parse a Unity .unity scene file and return all PrefabInstance data
        /// with world-space transforms.
        /// </summary>
        public List<PrefabInstance> Parse(string scenePath)
        {
            Debug.Log($"[UnitySceneParser] Parsing {scenePath}...");
            var lines = File.ReadAllLines(scenePath);
            Debug.Log($"[UnitySceneParser] {lines.Length} lines read");

            // Pass 1: Extract all Transform records (for parent chain resolution)
            // and all PrefabInstance records
            var transforms = new Dictionary<long, TransformRecord>();
            var prefabInstances = new List<RawPrefabData>();

            int i = 0;
            while (i < lines.Length)
            {
                var docMatch = DocumentStartRegex.Match(lines[i]);
                if (!docMatch.Success) { i++; continue; }

                int typeId = int.Parse(docMatch.Groups[1].Value);
                long fileId = long.Parse(docMatch.Groups[2].Value);

                // Collect all lines until next document separator
                int blockStart = i;
                i++;
                while (i < lines.Length && !lines[i].StartsWith("--- "))
                    i++;
                int blockEnd = i;

                switch (typeId)
                {
                    case 4: // Transform
                        var tr = ParseTransformBlock(lines, blockStart, blockEnd, fileId);
                        if (tr != null)
                            transforms[fileId] = tr;
                        break;
                    case 1001: // PrefabInstance
                        var pi = ParsePrefabInstanceBlock(lines, blockStart, blockEnd, fileId);
                        if (pi != null)
                            prefabInstances.Add(pi);
                        break;
                }
            }

            Debug.Log($"[UnitySceneParser] Found {transforms.Count} transforms, {prefabInstances.Count} prefab instances");

            // Pass 2: Resolve world transforms
            var result = new List<PrefabInstance>();
            var worldTransformCache = new Dictionary<long, (Vector3 pos, Quaternion rot, Vector3 scale)>();

            foreach (var raw in prefabInstances)
            {
                var worldPos = raw.LocalPosition;
                var worldRot = raw.LocalRotation;
                var worldScale = raw.LocalScale;

                // Walk parent chain
                if (raw.TransformParentFileID != 0 && transforms.ContainsKey(raw.TransformParentFileID))
                {
                    var (parentPos, parentRot, parentScale) = GetWorldTransform(
                        raw.TransformParentFileID, transforms, worldTransformCache);

                    // child world = parent world * child local
                    worldPos = parentPos + Vector3.Transform(raw.LocalPosition * parentScale, parentRot);
                    worldRot = parentRot * raw.LocalRotation;
                    worldScale = parentScale * raw.LocalScale;
                }

                result.Add(new PrefabInstance
                {
                    PrefabGuid = raw.PrefabGuid,
                    DisplayName = raw.DisplayName,
                    Position = worldPos,
                    Rotation = worldRot,
                    Scale = worldScale
                });
            }

            Debug.Log($"[UnitySceneParser] Resolved {result.Count} prefab instances with world transforms");
            return result;
        }

        /// <summary>
        /// Recursively compute world transform from parent chain.
        /// </summary>
        private (Vector3 pos, Quaternion rot, Vector3 scale) GetWorldTransform(
            long fileId,
            Dictionary<long, TransformRecord> transforms,
            Dictionary<long, (Vector3 pos, Quaternion rot, Vector3 scale)> cache)
        {
            if (cache.TryGetValue(fileId, out var cached))
                return cached;

            if (!transforms.TryGetValue(fileId, out var tr))
            {
                var identity = (Vector3.Zero, Quaternion.Identity, Vector3.One);
                cache[fileId] = identity;
                return identity;
            }

            if (tr.ParentFileID == 0 || !transforms.ContainsKey(tr.ParentFileID))
            {
                var local = (tr.LocalPosition, tr.LocalRotation, tr.LocalScale);
                cache[fileId] = local;
                return local;
            }

            var (parentPos, parentRot, parentScale) = GetWorldTransform(tr.ParentFileID, transforms, cache);
            var worldPos = parentPos + Vector3.Transform(tr.LocalPosition * parentScale, parentRot);
            var worldRot = parentRot * tr.LocalRotation;
            var worldScale = parentScale * tr.LocalScale;

            var result = (worldPos, worldRot, worldScale);
            cache[fileId] = result;
            return result;
        }

        // ─── Raw data structures for intermediate parse ─────────────────

        private class RawPrefabData
        {
            public string PrefabGuid;
            public string DisplayName;
            public long TransformParentFileID;
            public Vector3 LocalPosition;
            public Quaternion LocalRotation = Quaternion.Identity;
            public Vector3 LocalScale = Vector3.One;
        }

        // ─── Block parsers ──────────────────────────────────────────────

        /// <summary>
        /// Parse a Transform block (typeId=4) for a non-prefab GameObject.
        /// </summary>
        private TransformRecord ParseTransformBlock(string[] lines, int start, int end, long fileId)
        {
            var tr = new TransformRecord { FileID = fileId };

            for (int i = start; i < end; i++)
            {
                var line = lines[i].TrimStart();

                if (line.StartsWith("m_LocalPosition:"))
                {
                    var m = InlineFloatRegex.Match(line);
                    if (m.Success)
                        tr.LocalPosition = ParseVector3(m);
                }
                else if (line.StartsWith("m_LocalRotation:"))
                {
                    var m = InlineFloatRegex.Match(line);
                    if (m.Success)
                        tr.LocalRotation = ParseQuaternion(m);
                }
                else if (line.StartsWith("m_LocalScale:"))
                {
                    var m = InlineFloatRegex.Match(line);
                    if (m.Success)
                        tr.LocalScale = ParseVector3(m);
                }
                else if (line.StartsWith("m_Father:"))
                {
                    var m = FileIdRegex.Match(line);
                    if (m.Success)
                        tr.ParentFileID = long.Parse(m.Groups[1].Value);
                }
            }

            return tr;
        }

        /// <summary>
        /// Parse a PrefabInstance block (typeId=1001).
        /// Extracts prefab GUID, transform parent, and property modifications.
        /// </summary>
        private RawPrefabData ParsePrefabInstanceBlock(string[] lines, int start, int end, long fileId)
        {
            var data = new RawPrefabData();
            string currentPropertyPath = null;

            for (int i = start; i < end; i++)
            {
                var line = lines[i].TrimStart();

                // m_SourcePrefab: {fileID: 100100000, guid: <32hex>, type: 3}
                if (line.StartsWith("m_SourcePrefab:"))
                {
                    var m = GuidRegex.Match(line);
                    if (m.Success)
                        data.PrefabGuid = m.Groups[1].Value;
                }
                // m_TransformParent: {fileID: <id>}
                else if (line.StartsWith("m_TransformParent:"))
                {
                    var m = FileIdRegex.Match(line);
                    if (m.Success)
                        data.TransformParentFileID = long.Parse(m.Groups[1].Value);
                }
                // Property modification: propertyPath
                else if (line.StartsWith("propertyPath:"))
                {
                    currentPropertyPath = line.Substring("propertyPath:".Length).Trim();
                }
                // Property modification: value
                else if (line.StartsWith("value:") && currentPropertyPath != null)
                {
                    var valMatch = FloatValueRegex.Match(line);

                    switch (currentPropertyPath)
                    {
                        case "m_LocalPosition.x":
                            if (valMatch.Success) data.LocalPosition.X = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_LocalPosition.y":
                            if (valMatch.Success) data.LocalPosition.Y = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_LocalPosition.z":
                            if (valMatch.Success) data.LocalPosition.Z = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_LocalRotation.x":
                            if (valMatch.Success) data.LocalRotation = new Quaternion(
                                ParseFloat(valMatch.Groups[1].Value),
                                data.LocalRotation.Y, data.LocalRotation.Z, data.LocalRotation.W);
                            break;
                        case "m_LocalRotation.y":
                            if (valMatch.Success) data.LocalRotation = new Quaternion(
                                data.LocalRotation.X, ParseFloat(valMatch.Groups[1].Value),
                                data.LocalRotation.Z, data.LocalRotation.W);
                            break;
                        case "m_LocalRotation.z":
                            if (valMatch.Success) data.LocalRotation = new Quaternion(
                                data.LocalRotation.X, data.LocalRotation.Y,
                                ParseFloat(valMatch.Groups[1].Value), data.LocalRotation.W);
                            break;
                        case "m_LocalRotation.w":
                            if (valMatch.Success) data.LocalRotation = new Quaternion(
                                data.LocalRotation.X, data.LocalRotation.Y,
                                data.LocalRotation.Z, ParseFloat(valMatch.Groups[1].Value));
                            break;
                        case "m_LocalScale.x":
                            if (valMatch.Success) data.LocalScale.X = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_LocalScale.y":
                            if (valMatch.Success) data.LocalScale.Y = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_LocalScale.z":
                            if (valMatch.Success) data.LocalScale.Z = ParseFloat(valMatch.Groups[1].Value);
                            break;
                        case "m_Name":
                            data.DisplayName = line.Substring("value:".Length).Trim();
                            break;
                    }
                    currentPropertyPath = null;
                }
            }

            // Skip if no prefab GUID found
            if (string.IsNullOrEmpty(data.PrefabGuid))
                return null;

            return data;
        }

        // ─── Helpers ────────────────────────────────────────────────────

        private static float ParseFloat(string s)
        {
            float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float v);
            return v;
        }

        private static Vector3 ParseVector3(Match m)
        {
            return new Vector3(
                ParseFloat(m.Groups[1].Value),
                ParseFloat(m.Groups[2].Value),
                ParseFloat(m.Groups[3].Value));
        }

        private static Quaternion ParseQuaternion(Match m)
        {
            return new Quaternion(
                ParseFloat(m.Groups[1].Value),
                ParseFloat(m.Groups[2].Value),
                ParseFloat(m.Groups[3].Value),
                ParseFloat(m.Groups[4].Value));
        }
    }
}
