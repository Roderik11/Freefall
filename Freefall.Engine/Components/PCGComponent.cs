using System;
using System.Collections.Generic;
using Freefall.Base;
using Freefall.PCG;

namespace Freefall.Components
{
    /// <summary>
    /// PCG (Procedural Content Generation) component.
    /// References a PCGGraph asset and executes it, spawning entities as children.
    /// 
    /// On Execute: destroys previous output, injects context (Spline, etc.) into source nodes,
    /// then runs the graph. SpawnPrefab nodes parent their output under this entity.
    /// 
    /// Live editing: listens for SplineChanged and GraphChanged messages to auto-regenerate.
    /// </summary>
    public class PCGComponent : Component
    {
        /// <summary>PCGGraph asset to execute.</summary>
        public PCGGraph Graph;

        /// <summary>Auto-execute when the component wakes up.</summary>
        [System.ComponentModel.DefaultValue(false)]
        public bool ExecuteOnAwake = false;

        /// <summary>
        /// Child entity that holds all spawned output.
        /// Destroyed and recreated on each Execute() call.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public Entity OutputEntity { get; private set; }

        /// <summary>
        /// Execute the PCG graph: destroy previous output, inject context, run graph.
        /// </summary>
        public void Execute()
        {
            if (Graph == null || Graph.Nodes.Count == 0)
            {
                Debug.Log("[PCG] No graph to execute.");
                return;
            }

            // 1. Destroy previous output
            DestroyOutput();

            // 2. Create output container
            OutputEntity = new Entity("PCG_Output");
            OutputEntity.Transform.Parent = Transform;

            // 3. Inject context into nodes that need it
            InjectContext();

            // 4. Execute graph
            Graph.Execute();

            int childCount = OutputEntity.Transform.GetChildCount();
            Debug.Log($"[PCG] Executed graph: {Graph.Nodes.Count} nodes, {childCount} children spawned.");

            MessageDispatcher.Send(EngineMsg.PCGExecuted, this);
        }

        /// <summary>
        /// Destroy all previously spawned output entities.
        /// </summary>
        public void DestroyOutput()
        {
            if (OutputEntity == null) return;
            DestroyHierarchy(OutputEntity);
            OutputEntity = null;
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener(EngineMsg.GraphChanged, OnGraphChanged);
            DestroyOutput();
            base.Destroy();
        }

        protected override void Awake()
        {
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener(EngineMsg.GraphChanged, OnGraphChanged);

            if (ExecuteOnAwake && Graph != null)
                Execute();
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is not Spline spline) return;
            if (!IsDescendant(spline.Entity)) return;
            Execute();
        }

        private bool IsDescendant(Entity other)
        {
            var t = other?.Transform;
            while (t != null)
            {
                if (t.Entity == Entity) return true;
                t = t.Parent;
            }
            return false;
        }

        private void OnGraphChanged(Message msg)
        {
            if (msg.Data != Graph) return; // not our graph
            Execute();
        }

        /// <summary>
        /// Inject this entity's components as context for graph nodes.
        /// </summary>
        private void InjectContext()
        {
            Spline spline = FindInDescendants<Spline>(Entity);

            foreach (var node in Graph.Nodes)
            {
                if (node is SplineSampler sampler && spline != null)
                    sampler.Spline = spline;

                if (node is SpawnPrefab spawner)
                    spawner.SpawnParent = OutputEntity;
            }
        }

        private static T FindInDescendants<T>(Entity entity) where T : Component
        {
            foreach (var comp in entity.Components)
            {
                if (comp is T found) return found;
            }

            int childCount = entity.Transform.GetChildCount();
            for (int i = 0; i < childCount; i++)
            {
                var child = entity.Transform.GetChild(i);
                if (child?.Entity == null) continue;
                var result = FindInDescendants<T>(child.Entity);
                if (result != null) return result;
            }

            return null;
        }

        private static void DestroyHierarchy(Entity entity)
        {
            var children = new List<Entity>();
            int childCount = entity.Transform.GetChildCount();
            for (int i = 0; i < childCount; i++)
            {
                var child = entity.Transform.GetChild(i);
                if (child?.Entity != null)
                    children.Add(child.Entity);
            }

            foreach (var child in children)
                DestroyHierarchy(child);

            entity.Destroy();
        }
    }
}
