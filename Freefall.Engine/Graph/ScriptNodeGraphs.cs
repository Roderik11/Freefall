using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Serialization;

namespace Freefall.Graph
{
    /// <summary>
    /// Graph nodes defined in project scripts across a script hot reload. A loaded graph holding such nodes would keep
    /// the old build loaded and run its old code: before the old build unloads, <see cref="Capture"/> writes each such
    /// graph out as YAML (unsaved edits included) and empties it; once the new build is registered,
    /// <see cref="Restore"/> reads the YAML back into the same instance, so its nodes are made of the new build's
    /// types, everything that references the graph keeps it, and <see cref="EngineMsg.GraphChanged"/> reruns the
    /// PCG components that use it.
    /// </summary>
    public sealed class ScriptNodeGraphs
    {
        private readonly List<(NodeGraph graph, string yaml)> _graphs = new();

        public int Count => _graphs.Count;

        public static ScriptNodeGraphs Capture(IReadOnlyCollection<Assembly> oldAssemblies)
        {
            var captured = new ScriptNodeGraphs();
            if (oldAssemblies.Count == 0 || Engine.Assets == null) return captured;
            foreach (var graph in Engine.Assets.Loaded<NodeGraph>())
            {
                if (!graph.Nodes.Any(n => n != null && oldAssemblies.Contains(n.GetType().Assembly)))
                    continue;
                string yaml;
                try { yaml = NativeImporter.SaveToString(graph); }
                catch (Exception ex)
                {
                    Debug.LogWarning("ScriptReload", $"Graph '{graph.Name}': could not write it out ({ex.Message}); its script nodes are dropped");
                    yaml = null;
                }
                graph.Nodes.Clear();
                graph.Connections.Clear();
                if (graph is IRebuildAfterLoad rebuild) rebuild.RebuildAfterLoad();
                if (yaml != null) captured._graphs.Add((graph, yaml));
            }
            return captured;
        }

        /// <summary>Read each captured graph back with the new build's types. Returns how many came back.</summary>
        public int Restore()
        {
            int restored = 0;
            foreach (var (graph, yaml) in _graphs)
            {
                if (NativeImporter.LoadInto(graph, yaml, Engine.Assets))
                {
                    restored++;
                    MessageDispatcher.Send(EngineMsg.GraphChanged, graph);
                }
                else
                    Debug.LogWarning("ScriptReload", $"Graph '{graph.Name}' could not be read back after the reload");
            }
            _graphs.Clear();
            return restored;
        }
    }
}
