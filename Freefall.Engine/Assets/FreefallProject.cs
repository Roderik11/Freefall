using System;
using System.IO;
using System.Text.Json;

namespace Freefall.Assets
{
    /// <summary>
    /// Represents a Freefall project. A project is a folder containing:
    /// - Assets/  : source files (fbx, dae, png, etc.)
    /// - Library/ : meta files (version-controlled)
    /// - Cache/   : imported binary artifacts (gitignored, rebuildable)
    /// </summary>
    public class FreefallProject
    {
        public string Name { get; set; }
        public string RootDirectory { get; set; }

        public string AssetsDirectory => Path.Combine(RootDirectory, "Assets");
        public string LibraryDirectory => Path.Combine(RootDirectory, "Library");
        public string CacheDirectory => Path.Combine(RootDirectory, "Cache");

        /// <summary>
        /// Opens a project from a .ffproject file or a directory containing one.
        /// Creates Assets/, Library/, Cache/ directories if they don't exist.
        /// </summary>
        public static FreefallProject Open(string path)
        {
            string projectFile;
            string rootDir;

            if (Directory.Exists(path))
            {
                // Path is a directory — look for .ffproject file
                var files = Directory.GetFiles(path, "*.ffproject");
                if (files.Length == 0)
                    throw new FileNotFoundException($"No .ffproject file found in '{path}'");
                projectFile = files[0];
                rootDir = path;
            }
            else if (File.Exists(path) && path.EndsWith(".ffproject", StringComparison.OrdinalIgnoreCase))
            {
                projectFile = path;
                rootDir = Path.GetDirectoryName(path);
            }
            else
            {
                throw new FileNotFoundException($"Project not found: '{path}'");
            }

            var json = File.ReadAllText(projectFile);
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var project = JsonSerializer.Deserialize<FreefallProject>(json, options);
            project.RootDirectory = rootDir;

            // Ensure directories exist
            Directory.CreateDirectory(project.AssetsDirectory);
            Directory.CreateDirectory(project.LibraryDirectory);
            Directory.CreateDirectory(project.CacheDirectory);

            Debug.Log($"[Project] Opened '{project.Name}' at {rootDir}");
            return project;
        }

        /// <summary>
        /// Creates a new project at the specified directory.
        /// </summary>
        public static FreefallProject Create(string directory, string name)
        {
            Directory.CreateDirectory(directory);

            var project = new FreefallProject
            {
                Name = name,
                RootDirectory = directory
            };

            // Create directories
            Directory.CreateDirectory(project.AssetsDirectory);
            Directory.CreateDirectory(project.LibraryDirectory);
            Directory.CreateDirectory(project.CacheDirectory);

            // Write .ffproject file
            var projectFile = Path.Combine(directory, $"{name}.ffproject");
            var json = JsonSerializer.Serialize(new { name }, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(projectFile, json);

            Debug.Log($"[Project] Created '{name}' at {directory}");
            return project;
        }
    }
}
