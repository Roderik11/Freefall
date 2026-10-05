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
            var previous = Entries.Find(e => string.Equals(e.Path, path, StringComparison.OrdinalIgnoreCase));

            // Remove existing entry for same path (case-insensitive)
            Entries.RemoveAll(e => string.Equals(e.Path, path, StringComparison.OrdinalIgnoreCase));

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
            Entries.RemoveAll(e => string.Equals(e.Path, path, StringComparison.OrdinalIgnoreCase));
            Save();
        }
    }
}
