using System;
using System.Collections.Generic;
using System.Reflection;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Editor
{
    /// <summary>
    /// Discovers and manages [CustomAssetEditor] registrations.
    /// Maps asset types to editor instances for double-click dispatch.
    /// Walks the type hierarchy so PCGGraph finds the NodeGraph editor.
    /// </summary>
    public static class CustomAssetEditorRegistry
    {
        private static readonly Dictionary<Type, ICustomAssetEditor> _editors = new();
        private static bool _initialized;

        /// <summary>
        /// Discover all [CustomAssetEditor] implementations in loaded assemblies.
        /// Called once at startup.
        /// </summary>
        public static void Initialize()
        {
            if (_initialized) return;
            _initialized = true;

            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() };
            foreach (var assembly in assemblies)
            {
                if (assembly == null) continue;
                try
                {
                    foreach (var type in assembly.GetTypes())
                    {
                        var attr = type.GetCustomAttribute<CustomAssetEditorAttribute>();
                        if (attr == null) continue;
                        if (!typeof(ICustomAssetEditor).IsAssignableFrom(type)) continue;

                        // Skip if already registered (first wins)
                        if (_editors.ContainsKey(attr.AssetType)) continue;

                        // Editor must already be instantiated as a UI control in the dock.
                        // We'll register instances later via Register().
                        Debug.Log($"[CustomAssetEditor] Found: {type.Name} → {attr.AssetType.Name}");
                    }
                }
                catch (ReflectionTypeLoadException) { }
            }
        }

        /// <summary>
        /// Register a live editor instance (called from EditorDesktop after dock setup).
        /// </summary>
        public static void Register(ICustomAssetEditor editor)
        {
            var attr = editor.GetType().GetCustomAttribute<CustomAssetEditorAttribute>();
            if (attr == null) return;
            _editors[attr.AssetType] = editor;
            Debug.Log($"[CustomAssetEditor] Registered: {editor.GetType().Name} → {attr.AssetType.Name}");
        }

        /// <summary>
        /// Find an editor for the given asset type, walking up the inheritance chain.
        /// </summary>
        public static ICustomAssetEditor Find(Type assetType)
        {
            var type = assetType;
            while (type != null && type != typeof(object))
            {
                if (_editors.TryGetValue(type, out var editor))
                    return editor;
                type = type.BaseType;
            }
            return null;
        }

        /// <summary>
        /// Try to open an asset in its custom editor.
        /// Returns true if a matching editor was found.
        /// </summary>
        public static bool TryOpen(Asset asset)
        {
            if (asset == null) return false;
            var editor = Find(asset.GetType());
            if (editor == null) return false;
            editor.OpenAsset(asset);
            return true;
        }
    }
}
