using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Freefall.Editor
{
    public class RecentProjectEntry
    {
        public string Name { get; set; }
        public string Path { get; set; }
        public DateTime LastOpened { get; set; }

        // Recorded when the project is closed or a scene is saved; empty until then.
        public string Scene { get; set; }
        public int EntityCount { get; set; }
    }

    /// <summary>
    /// Manages the list of recently opened projects.
    /// Persists to %APPDATA%/Freefall/recent_projects.json.
    /// </summary>
    public static class RecentProjects
    {
        private const int MaxEntries = 20;

        private static readonly string AppDataDir =
            System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Freefall");
        private static readonly string FilePath =
            System.IO.Path.Combine(AppDataDir, "recent_projects.json");

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            WriteIndented = true,
            PropertyNameCaseInsensitive = true
        };

        public static List<RecentProjectEntry> Entries { get; private set; } = new();

        public static void Load()
        {
            if (!File.Exists(FilePath))
            {
                Entries = new List<RecentProjectEntry>();
                return;
            }

            try
            {
                var json = File.ReadAllText(FilePath);
                Entries = JsonSerializer.Deserialize<List<RecentProjectEntry>>(json, JsonOptions) ?? new();
            }
            catch
            {
                Entries = new List<RecentProjectEntry>();
            }
        }

        public static void Save()
        {
            try
            {
                Directory.CreateDirectory(AppDataDir);
                var json = JsonSerializer.Serialize(Entries, JsonOptions);
                File.WriteAllText(FilePath, json);
            }
            catch (Exception ex)
            {
                Freefall.Debug.LogWarning("RecentProjects", $"Failed to save: {ex.Message}");
            }
        }

        /// <summary>The project opened in this session, or null on the landing page.</summary>
        public static RecentProjectEntry Current { get; private set; }

        public static void Add(string name, string path)
        {
            var previous = Entries.Find(e => SamePath(e.Path, path));

            // Remove existing entry for same folder, however it was spelled
            Entries.RemoveAll(e => SamePath(e.Path, path));

            // Insert at top
            Current = new RecentProjectEntry
            {
                Name = name,
                Path = path,
                LastOpened = DateTime.UtcNow,
                Scene = previous?.Scene,
                EntityCount = previous?.EntityCount ?? 0,
            };
            Entries.Insert(0, Current);

            // Trim to max
            if (Entries.Count > MaxEntries)
                Entries = Entries.Take(MaxEntries).ToList();

            Save();
        }

        public static void Remove(string path)
        {
            Entries.RemoveAll(e => SamePath(e.Path, path));
            Save();
        }

        /// <summary>One folder however it is spelled ("D:/x" and "D:\x\"), ignoring case.</summary>
        private static bool SamePath(string a, string b) =>
            string.Equals(FullPath(a), FullPath(b), StringComparison.OrdinalIgnoreCase);

        private static string FullPath(string path)
        {
            try { return Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)); }
            catch { return path; }
        }
    }
}
