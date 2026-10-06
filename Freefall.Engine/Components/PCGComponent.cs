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
    /// Live editing: SplineChanged, GraphChanged and TerrainHeightsChanged messages Invalidate() the component;
    /// PCGScheduler then regenerates it once, after the edit (gizmo drag, inspector drag, build) has finished.
    /// </summary>
    [Icon("icon_pcg.png")]
    public class PCGComponent : Component
    {
        /// <summary>PCGGraph asset to execute.</summary>
        public PCGGraph Graph;

        /// <summary>
        /// Auto-execute when the component wakes up. On by default: spawned output is DontSave, so a component that
        /// doesn't run on awake comes back empty after every scene load.
        /// </summary>
        [System.ComponentModel.DefaultValue(true)]
        public bool ExecuteOnAwake = true;

        /// <summary>
        /// Child entity that holds all spawned output.
        /// Destroyed and recreated on each Execute() call.
        /// </summary>
        private Entity? OutputEntity;

        /// <summary>
        /// Meters around the spline that still count as this component's area when deciding whether a terrain
        /// change concerns it (points can be offset from the curve by the graph).
        /// </summary>
        private const float RegionPadding = 10f;

        /// <summary>Diagnostics: total Execute() calls across all PCG components (editor debug stats).</summary>
        public static int ExecuteCount;

        /// <summary>
        /// Execute the PCG graph: destroy previous output, inject context, run graph.
        /// </summary>
        public void Execute()
        {
            PCGScheduler.Cancel(this);

            if (Graph == null || Graph.Nodes.Count == 0)
            {
                Debug.Log("[PCG] No graph to execute.");
                return;
            }

            ExecuteCount++;

            // 1. Destroy previous output
            DestroyOutput();

            // 2. Create output container
            OutputEntity = new Entity("PCG_Output");
            OutputEntity.Transform.Parent = Transform;
            OutputEntity.Flags |= EntityFlags.DontSave;

            // 3. Inject context into nodes that need it
            InjectContext();

            // 4. Execute graph
            Graph.Execute();

            int childCount = OutputEntity.Transform.GetChildCount();
            Debug.Log($"[PCG] Executed graph: {Graph.Nodes.Count} nodes, {childCount} children spawned.");

            MessageDispatcher.Send(EngineMsg.PCGExecuted, this);
        }

        /// <summary>
        /// Request a regeneration. Runs once via PCGScheduler, however many changes arrive before it does.
        /// </summary>
        public void Invalidate() => PCGScheduler.Request(this);

        /// <summary>
        /// Destroy all previously spawned output entities.
        /// </summary>
        public void DestroyOutput()
        {
            OutputEntity?.Destroy();
            OutputEntity = null;
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener(EngineMsg.GraphChanged, OnGraphChanged);
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            PCGScheduler.Cancel(this);
            DestroyOutput();
            base.Destroy();
        }

        protected override void Awake()
        {
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener(EngineMsg.GraphChanged, OnGraphChanged);
            MessageDispatcher.AddListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);

            if (ExecuteOnAwake && Graph != null)
                Execute();
        }

        /// <summary>Inspector / command-server edits (Graph, ExecuteOnAwake) re-run the graph.</summary>
        public override void OnMemberChanged()
        {
            if (Graph != null) Invalidate();
            else
            {
                PCGScheduler.Cancel(this);
                DestroyOutput();
            }
        }

        // TerrainProjection samples the CPU HeightField, which can be stale at load until the GPU bake is read
        // back (see RuntimeMesh). Re-run so projected output lands on the real surface.
        private void OnTerrainHeightsChanged(Message msg)
        {
            if (Graph == null || OutputEntity == null) return;

            // Heights changed somewhere else on the terrain: this component's output is still right
            if (msg.Data is TerrainHeightsChange { All: false } change)
            {
                var spline = Entity.GetComponentInChildren<Spline>();
                if (spline != null && spline.TryGetWorldBoundsXZ(RegionPadding, out var min, out var max)
                    && !change.Overlaps(min, max))
                    return;
            }

            foreach (var node in Graph.Nodes)
                if (node is TerrainProjection) { Invalidate(); return; }
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is not Spline spline) return;
            if (!IsDescendant(spline.Entity)) return;
            Invalidate();
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
            Invalidate();
        }

        /// <summary>
        /// Inject this entity's components as context for graph nodes.
        /// </summary>
        private void InjectContext()
        {
            Spline spline = Entity.GetComponentInChildren<Spline>();

            foreach (var node in Graph.Nodes)
            {
                if (node is SplineSampler sampler && spline != null)
                    sampler.Spline = spline;

                if (node is SpawnPrefab spawner)
                    spawner.SpawnParent = OutputEntity;

                if (node is ExcludeObstacles exclude)
                    exclude.IgnoreRoot = OutputEntity;

                if (node is ExcludeStamps excludeStamps)
                    excludeStamps.IgnoreEntity = Entity;

                if (node is MeshProjection meshProjection)
                    meshProjection.IgnoreRoot = OutputEntity;

                // Points are local to this entity; nodes testing them against world data need the transform.
                if (node is IWorldSpaceNode worldNode)
                    worldNode.WorldMatrix = Transform?.Matrix ?? System.Numerics.Matrix4x4.Identity;
            }
        }
    }
}
