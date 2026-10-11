using System;
using System.Collections.Generic;
using System.Numerics;
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
    ///
    /// The output is a function of the scene alone, whatever ran when: all components run in one order
    /// (CompareOrder), a graph only takes the output of components before it in that order into account (Sees),
    /// and a component that regenerates sends the later ones around it to regenerate as well.
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

        // Output container -> the component it belongs to
        private static readonly Dictionary<Entity, PCGComponent> _generators = new();

        /// <summary>
        /// Meters around the spline that still count as this component's area when deciding whether a change
        /// elsewhere (terrain heights, another component's output) concerns it: points can be offset from the
        /// curve by the graph, and obstacles are tested with a margin.
        /// </summary>
        private const float RegionPadding = 15f;

        /// <summary>Diagnostics: total Execute() calls across all PCG components (editor debug stats).</summary>
        public static int ExecuteCount;

        /// <summary>
        /// Execute the PCG graph now: destroy previous output, inject context, run graph.
        /// While terrain heights are still on their way (scene load) the run is queued instead and happens
        /// as soon as they are there: until then the terrain reads as flat ground at height 0.
        /// </summary>
        public void Execute()
        {
            PCGScheduler.Cancel(this);

            if (Graph == null || Graph.Nodes.Count == 0)
            {
                Debug.Log("[PCG] No graph to execute.");
                return;
            }

            if (PCGScheduler.TerrainHeightsPending)
            {
                Invalidate();
                return;
            }

            ExecuteCount++;

            // 1. Destroy previous output
            DestroyOutput();

            // 2. Create output container
            OutputEntity = new Entity("PCG_Output");
            OutputEntity.Transform.Parent = Transform;
            OutputEntity.Flags |= EntityFlags.DontSave;
            _generators[OutputEntity] = this;

            // 3. Inject context into nodes that need it
            InjectContext();

            // 4. Execute graph
            Graph.Execute();

            int childCount = OutputEntity.Transform.GetChildCount();
            Debug.Log($"[PCG] '{Entity?.Name}' ran '{Graph.Name}' (tick {Engine.TickCount}): {Graph.Nodes.Count} nodes, {childCount} children spawned." +
                      (TraceNodes ? " Points per node: " + DescribeNodeOutputs() : ""));

            // 5. Whoever takes this output into account is out of date now
            PCGScheduler.OutputChanged(this);

            MessageDispatcher.Send(EngineMsg.PCGExecuted, this);
        }

        /// <summary>
        /// Diagnostics: add the number of points each node put out to the execution log line, to see which
        /// node behaves differently between two runs. Set FREEFALL_PCG_TRACE=1 before starting the editor.
        /// </summary>
        public static bool TraceNodes = Environment.GetEnvironmentVariable("FREEFALL_PCG_TRACE") == "1";

        private string DescribeNodeOutputs()
        {
            var text = new System.Text.StringBuilder();
            foreach (var node in Graph.Nodes)
            {
                foreach (var port in node.Outputs)
                {
                    if (port.Field.GetValue(node) is not SamplePointSet points) continue;
                    if (text.Length > 0) text.Append(", ");
                    text.Append(node.ID).Append(':').Append(node.GetType().Name).Append('=').Append(points.Count);
                }
            }
            return text.ToString();
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
            if (OutputEntity == null) return;

            _generators.Remove(OutputEntity);
            OutputEntity.Destroy();
            OutputEntity = null;
        }

        /// <summary>True once the graph has run and its output (possibly empty) is in the scene.</summary>
        [System.ComponentModel.Browsable(false)]
        public bool HasOutput => OutputEntity != null;

        // ── Order and visibility between components ──

        /// <summary>The PCG component an entity was spawned by, or null (authored, or built by something else).</summary>
        public static PCGComponent? GeneratorOf(Entity? entity)
        {
            if (_generators.Count == 0) return null;

            for (var t = entity?.Transform; t != null; t = t.Parent)
                if (t.Entity != null && _generators.TryGetValue(t.Entity, out var generator))
                    return generator;
            return null;
        }

        /// <summary>
        /// The graph tests its points against other content of the scene (ExcludeObstacles, MeshProjection onto
        /// generated geometry), so what it puts out depends on what other components put out.
        /// </summary>
        internal bool ReadsScene
        {
            get
            {
                // Asked for every obstacle a graph looks at: remembered per graph and node count
                if (Graph != _readsSceneGraph || (Graph?.Nodes.Count ?? 0) != _readsSceneNodes)
                {
                    _readsSceneGraph = Graph;
                    _readsSceneNodes = Graph?.Nodes.Count ?? 0;
                    _readsScene = false;
                    if (Graph != null)
                        foreach (var node in Graph.Nodes)
                            if (node is ExcludeObstacles or MeshProjection { IgnoreGenerated: false })
                            {
                                _readsScene = true;
                                break;
                            }
                }
                return _readsScene;
            }
        }

        private PCGGraph? _readsSceneGraph;
        private int _readsSceneNodes = -1;
        private bool _readsScene;

        /// <summary>
        /// The one order all PCG components run in: those that do not read the scene first (a forest), then
        /// those that do (roadside shrubs keeping clear of trees), each group by entity UID — fixed, and the
        /// same in every session. Negative if <paramref name="a"/> runs before <paramref name="b"/>.
        /// </summary>
        public static int CompareOrder(PCGComponent a, PCGComponent b)
        {
            int byKind = a.ReadsScene.CompareTo(b.ReadsScene);
            if (byKind != 0) return byKind;

            ulong ua = a.Entity?.UID ?? 0, ub = b.Entity?.UID ?? 0;
            return ua != ub ? ua.CompareTo(ub) : a.UID.CompareTo(b.UID);
        }

        /// <summary>
        /// Whether this component's graph takes a scene entity into account. Output of PCG components that run
        /// after this one never counts, even while it happens to be in the scene (left over from an earlier
        /// run): otherwise the result would depend on which component last ran when.
        /// </summary>
        public bool Sees(Entity? entity)
        {
            var generator = GeneratorOf(entity);
            return generator == null || (generator != this && CompareOrder(generator, this) < 0);
        }

        /// <summary>World XZ area this component works in. False if it has no spline: then it may be anywhere.</summary>
        internal bool TryGetRegion(out Vector2 min, out Vector2 max)
        {
            var spline = Entity?.GetComponentInChildren<Spline>();
            if (spline != null && spline.TryGetWorldBoundsXZ(RegionPadding, out min, out max))
                return true;

            min = max = default;
            return false;
        }

        // ── Lifecycle ──

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener(EngineMsg.GraphChanged, OnGraphChanged);
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            PCGScheduler.Cancel(this);
            RemoveOutput();
            base.Destroy();
        }

        // The output goes away for good: later components that kept clear of it regenerate
        private void RemoveOutput()
        {
            if (OutputEntity == null) return;

            PCGScheduler.OutputChanged(this);
            DestroyOutput();
        }

        protected override void Awake()
        {
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener(EngineMsg.GraphChanged, OnGraphChanged);
            MessageDispatcher.AddListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);

            // Queued, not run here: components wake in no particular order, and at load before the terrain
            // heights are there. PCGScheduler runs them at the end of the frame, in order.
            if (ExecuteOnAwake && Graph != null)
                Invalidate();
        }

        /// <summary>Inspector / command-server edits (Graph, ExecuteOnAwake) re-run the graph.</summary>
        public override void OnMemberChanged()
        {
            if (Graph != null) Invalidate();
            else
            {
                PCGScheduler.Cancel(this);
                RemoveOutput();
            }
        }

        // TerrainProjection samples the CPU HeightField, which can be stale at load until the GPU bake is read
        // back (see RuntimeMesh). Re-run so projected output lands on the real surface.
        private void OnTerrainHeightsChanged(Message msg)
        {
            if (Graph == null || OutputEntity == null) return;

            // Heights changed somewhere else on the terrain: this component's output is still right
            if (msg.Data is TerrainHeightsChange { All: false } change
                && TryGetRegion(out var min, out var max) && !change.Overlaps(min, max))
                return;

            foreach (var node in Graph.Nodes)
                if (node is TerrainProjection or MeshProjection { IncludeTerrain: true }) { Invalidate(); return; }
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
            _readsSceneNodes = -1;         // a node setting may have changed what the graph reads
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
                {
                    exclude.IgnoreRoot = OutputEntity;
                    exclude.Owner = this;
                }

                if (node is ExcludeStamps excludeStamps)
                    excludeStamps.IgnoreEntity = Entity;

                if (node is MeshProjection meshProjection)
                {
                    meshProjection.IgnoreRoot = OutputEntity;
                    meshProjection.Owner = this;
                }

                // Points are local to this entity; nodes testing them against world data need the transform.
                if (node is IWorldSpaceNode worldNode)
                    worldNode.WorldMatrix = Transform?.Matrix ?? System.Numerics.Matrix4x4.Identity;
            }
        }
    }
}
