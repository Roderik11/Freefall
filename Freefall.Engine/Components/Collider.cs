using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;
using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Components
{
    [Icon("icon_collider.png")]
    public abstract class Collider : Component
    {
        private RigidStatic? _staticActor;
        private Shape? _shape;

        // Where play mode's start goes (a wake-up of thousands of colliders)
        private static readonly ProfilerMarker _findBodyMarker = ProfilerMarker.Get("Collider find body");
        private static readonly ProfilerMarker _actorMarker = ProfilerMarker.Get("Collider actor");
        private static readonly ProfilerMarker _geometryMarker = ProfilerMarker.Get("Collider geometry");
        private static readonly ProfilerMarker _shapeMarker = ProfilerMarker.Get("Collider shape");

        // The entity whose actor the shape is on: this one (its own static actor) or a RigidBody's above it
        private Entity? _shapeHost;
        internal Entity? ShapeHost => _shapeHost;

        public Actor? Actor => _staticActor;

        /// <summary>
        /// Where the shape sits in the entity's frame: turned and scaled with it, so on an entity scaled
        /// by 2 an offset of 1 is 2 m. Read when the shape is made.
        /// </summary>
        public Vector3 Offset;

        /// <summary>
        /// How big the entity is drawn along each of its axes: its scale times its parents'. Every collider
        /// takes it into its geometry and its offset, since an actor's pose carries no scale: a tree that
        /// PCG scaled by 1.5 has a trunk 1.5 times as thick, and its capsule with it.
        /// </summary>
        protected Vector3 WorldScale
        {
            get
            {
                Transform.Matrix.Deconstruct(out var scale, out _, out _);
                return Vector3.Abs(scale);
            }
        }

        private bool _isTrigger;

        /// <summary>
        /// A trigger takes no part in the simulation: bodies fall through it and character
        /// controllers walk through it. Queries still report it (PhysicsWorld.Raycast leaves it out).
        /// Meshes and terrain cannot be triggers.
        /// </summary>
        public bool IsTrigger
        {
            get => _isTrigger;
            set
            {
                _isTrigger = value;
                ApplyTrigger();
            }
        }

        /// <summary>The group every collider is in unless it is put elsewhere (<see cref="Groups"/>).</summary>
        public const uint Solid = 1;

        private uint _groups = Solid;

        /// <summary>
        /// The groups the collider is in, as bits. A query that names groups finds only what is
        /// in one of them: a character's capsule moved with ControllerFilters.FilterData, a scene
        /// query with QueryFilterData (word 0 of either). One that names none finds every
        /// collider, and that is all the rays of PhysicsWorld. A capsule that looks for
        /// <see cref="Solid"/> alone passes through what was taken out of it and still stands on
        /// the ground. Not saved: whoever puts a collider elsewhere does so in play.
        /// </summary>
        [Reflection.DontSerialize]
        public uint Groups
        {
            get => _groups;
            set
            {
                _groups = value;
                if (_shape != null) _shape.QueryFilterData = new FilterData(_groups, 0, 0, 0);
            }
        }

        private bool _blocks = true;

        /// <summary>
        /// Off, the shape stays where it is but takes no part: nothing bumps into it, a character's capsule walks
        /// through it, and no query finds it (a door's leaf standing open: an NPC between it and the wall had no way out
        /// round it, 2026-10-25). On again, it is solid where it then is. Not saved: set in play. A trigger keeps its
        /// flags.
        /// </summary>
        [Reflection.DontSerialize]
        public bool Blocks
        {
            get => _blocks;
            set
            {
                if (_blocks == value) return;
                _blocks = value;
                ApplyBlocks();
            }
        }

        private void ApplyBlocks()
        {
            if (_shape == null || _isTrigger) return;
            const ShapeFlag solid = ShapeFlag.SimulationShape | ShapeFlag.SceneQueryShape;
            _shape.Flags = _blocks ? _shape.Flags | solid : _shape.Flags & ~solid;
        }

        // Kept in a field because it is set before there is a shape (a scene load, the editor).
        // PhysX refuses a shape that is a trigger and a simulation shape at once, so both flags
        // change in one assignment.
        private void ApplyTrigger()
        {
            if (_shape == null) return;

            var flags = _shape.Flags;
            _shape.Flags = _isTrigger
                ? (flags & ~ShapeFlag.SimulationShape) | ShapeFlag.TriggerShape
                : (flags & ~ShapeFlag.TriggerShape) | ShapeFlag.SimulationShape;

            if (_isTrigger && (_shape.Flags & ShapeFlag.TriggerShape) == 0)
                Debug.LogWarning($"[Collider] {Entity?.Name}: a {GetType().Name} cannot be a trigger, it stays solid");
        }

        protected override void Awake()
        {
            if (Engine.IsEditor)
                return;

            try
            {
                RigidBody? rigidBody;
                using (_findBodyMarker.Auto())
                    rigidBody = Entity?.GetComponentInParents<RigidBody>();
                if (rigidBody != null)
                    return;

                using (_actorMarker.Auto())
                    CreateFullActor();

                Transform.OnChanged += OnTransformChanged;
            }
            catch (Exception ex)
            {
                Debug.Log($"[RigidBody] {Entity?.Name}: {ex.Message}");
            }
        }

        private void OnTransformChanged()
        {
            if (Actor != null)
            {
                Transform.Matrix.Deconstruct(out var scale, out var rotation, out var translation);
                var matrix = Matrix4x4.CreateFromQuaternion(rotation)
                           * Matrix4x4.CreateTranslation(translation);
                _staticActor.GlobalPose = matrix;
                _staticActor.GlobalPosePosition = translation;
                _staticActor.GlobalPoseQuat = rotation;
            }
        }

        public abstract PhysX.Geometry? CreateGeometry();

        /// <summary>Where the geometry's origin sits in the collider's frame, before <see cref="Offset"/>: the terrain's height field starts half a texel in.</summary>
        protected virtual Vector3 GeometryOffset => Vector3.Zero;

        void CreateFullActor()
        {
            // Remove old actor (its shape goes with it)
            if (Actor != null)
            {
                PhysicsWorld.Scene.RemoveActor(Actor);
                ReleaseShape(native: false);
                PhysXRelease.Release(_staticActor);
                _staticActor = null;
            }

            // Build a clean pose from position + rotation (no scale).
            // PhysX requires an orthonormal rotation — scale is applied via geometry.
            Transform.Matrix.Deconstruct(out var scale, out var rotation, out var translation);
            var matrix = Matrix4x4.CreateFromQuaternion(rotation)
                       * Matrix4x4.CreateTranslation(translation);

            _staticActor = PhysicsWorld.Physics.CreateRigidStatic();
            _staticActor.UserData = Entity;
            _staticActor.GlobalPose = matrix;
            _staticActor.GlobalPosePosition = translation;
            _staticActor.GlobalPoseQuat = rotation;
            _staticActor.Flags &= ~ActorFlag.Visualization;
            _staticActor.Flags |= ActorFlag.DisableGravity;

            CreateShape(Entity, _staticActor);
            PhysicsWorld.Scene.AddActor(Actor!);
        }

        // The shape being made: for a geometry that comes later (TerrainCollider) to call CreateShape with again
        protected Entity? ShapeRoot { get; private set; }
        protected RigidActor? ShapeActor { get; private set; }

        public void CreateShape(Entity root, RigidActor actor)
        {
            ShapeRoot = root;
            ShapeActor = actor;
            PhysX.Geometry? geometry;
            using (_geometryMarker.Auto())
                geometry = CreateGeometry();
            if (geometry == null)
                return;

            using var shapeScope = _shapeMarker.Auto();
            _shape = RigidActorExt.CreateExclusiveShape(actor, geometry, PhysicsWorld.DefaultMaterial);
            _shapeHost = root;

            // Whose shape this is: under a RigidBody all colliders share the body's actor, whose
            // UserData is the body's entity, so a query's hit can only tell the piece by its shape
            _shape.UserData = this;
            _shape.QueryFilterData = new FilterData(_groups, 0, 0, 0);

            if (_isTrigger)
                ApplyTrigger();
            if (!_blocks)
                ApplyBlocks();

            // Offset is in the collider's own frame, scaled with it; on a child of the actor's entity
            // that frame is carried into the actor's.
            var localPose = Matrix4x4.CreateTranslation(Offset * WorldScale + GeometryOffset);

            // A PhysX capsule runs along its X axis; ours stands along the entity's Y.
            if (geometry is CapsuleGeometry)
                localPose = Matrix4x4.CreateRotationZ(MathF.PI / 2) * localPose;

            if(root != Entity)
            {
                // The actor's frame is the root's place and turn in the world without its scale (a
                // PhysX pose carries none), so the shape is placed in that frame: in metres. Taken
                // relative to the root's scaled matrix, a root (or anything above it) scaled by s put
                // every shape s times too near the root's pivot.
                root.Transform.Matrix.Deconstruct(out _, out var rootRotation, out var rootTranslation);
                var rootFrame = Matrix4x4.CreateFromQuaternion(rootRotation) * Matrix4x4.CreateTranslation(rootTranslation);
                Matrix4x4.Invert(rootFrame, out var rootInverse);
                var relative = Matrix4x4.Multiply(Transform.Matrix, rootInverse);
                relative.Deconstruct(out _, out var shapeRotation, out var shapeTranslation);
                localPose *= Matrix4x4.CreateFromQuaternion(shapeRotation) * Matrix4x4.CreateTranslation(shapeTranslation);
            }

            if (!localPose.IsIdentity)
                _shape.LocalPose = localPose;
        }

        public override void Destroy()
        {
            Transform.OnChanged -= OnTransformChanged;

            if (Actor != null)
            {
                // Its own actor: the shape goes with it
                PhysicsWorld.Scene.RemoveActor(Actor);
                ReleaseShape(native: false);
                PhysXRelease.Release(_staticActor);
                _staticActor = null;
                return;
            }

            // On a RigidBody's actor: left to the actor when the body goes too (a scene clear: thousands of pieces per
            // house block, each detached from an actor still in the scene, otherwise), detached when only this goes
            ReleaseShape(native: _shapeHost == null || !_shapeHost.IsBeingDestroyed);
        }

        /// <summary>
        /// Let go of the shape's wrapper; with <paramref name="native"/> false the PhysX shape stays on its actor, for the
        /// actor's release to take with it. RigidBody calls this for the shapes on its actor before releasing it.
        /// </summary>
        internal void ReleaseShape(bool native)
        {
            PhysXRelease.Release(_shape, native);
            _shape = null;
            _shapeHost = null;
        }
    }


    public class BoxCollider : Collider, ISceneGizmo
    {
        /// <summary>Full edge lengths in the entity's frame, scaled with it.</summary>
        public Vector3 Size = Vector3.One;

        public override PhysX.Geometry? CreateGeometry()
        {
            return new BoxGeometry(Size * 0.5f * WorldScale);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            var scale = WorldScale;
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawWireBox(Offset * scale, Size * scale);
        }
    }

    public class SphereCollider : Collider, ISceneGizmo
    {
        /// <summary>Scaled by the largest of the entity's scales, so the ball holds what it is put round.</summary>
        public float Radius = 0.5f;

        private float ScaledRadius
        {
            get
            {
                var scale = WorldScale;
                return Radius * MathF.Max(scale.X, MathF.Max(scale.Y, scale.Z));
            }
        }

        public override PhysX.Geometry? CreateGeometry()
        {
            return new SphereGeometry(ScaledRadius);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawWireSphere(Offset * WorldScale, ScaledRadius);
        }
    }

    public class CapsuleCollider : Collider, ISceneGizmo
    {
        /// <summary>Scaled by the larger of the entity's X and Z scales.</summary>
        public float Radius = 0.5f;

        /// <summary>
        /// From end to end along the entity's Y, caps included, scaled by its Y scale. Never less than a ball
        /// of the radius.
        /// </summary>
        public float Height = 2.0f;

        // Radius and height as big as the entity is drawn
        private (float Radius, float Height) Scaled()
        {
            var scale = WorldScale;
            return (Radius * MathF.Max(scale.X, scale.Z), Height * scale.Y);
        }

        public override PhysX.Geometry? CreateGeometry()
        {
            // PhysX takes half the cylinder between the caps, and refuses a capsule that has
            // none: that one is a sphere.
            var (radius, height) = Scaled();
            float halfCylinder = height * 0.5f - radius;
            if (halfCylinder <= 0f)
                return new SphereGeometry(radius);

            return new CapsuleGeometry(radius, halfCylinder);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            var (radius, height) = Scaled();
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawCapsule(Offset * WorldScale, MathF.Max(height, radius * 2f), radius);
        }
    }

    /// <summary>
    /// The entity's mesh (or <see cref="Mesh"/>) as a static triangle mesh, at the size it is drawn:
    /// the entity's scale times its parents'. Cooked once per mesh.
    /// </summary>
    public class MeshCollider : Collider
    {
        public Mesh? Mesh;

        // Cooked meshes by geometry (Mesh.PhysicsGeometryKey), shared across scene loads: a reload brings new Mesh
        // objects of the same assets, and RuntimeMesh output (streets, paving) comes back the same under a new name.
        // By GUID until 2026-10-21: RuntimeMesh meshes got a new one per load, so all 75 of Eldermoor's cooked again on
        // every Play (0.8 s) and the cache gained 75 entries each time. By geometry also means a reimported mesh is
        // cooked anew (until 2026-10-17 its colliders kept the old shape: the player fell through the reworked cave
        // mouth's new floor). Entries of RuntimeMesh output that the last loads did not use are let go.
        sealed class Cooked
        {
            public required TriangleMesh Mesh;
            public bool Dynamic;
            public int LastUsed;
        }
        static readonly Dictionary<ulong, Cooked> cookedMeshes = [];
        static readonly ProfilerMarker _cookMarker = ProfilerMarker.Get("Collider mesh cook");

        // Counts the scene loads that made mesh colliders (an edit-mode load makes none, so Play, Stop, Play keeps them)
        static int _loadNumber;
        static SceneLoad? _numberedLoad;

        static int LoadNumber()
        {
            var load = SceneManager.Loading;
            if (load != null && load != _numberedLoad)
            {
                _numberedLoad = load;
                _loadNumber++;
                List<ulong>? stale = null;
                foreach (var (key, entry) in cookedMeshes)
                    if (entry.Dynamic && entry.LastUsed < _loadNumber - 2) (stale ??= []).Add(key);
                if (stale != null)
                    foreach (var key in stale)
                    {
                        PhysXRelease.Release(cookedMeshes[key].Mesh);   // shapes still on it keep the PhysX mesh
                        cookedMeshes.Remove(key);
                    }
            }
            return _loadNumber;
        }

        // The mesh CreateGeometry will use, as far as it is known yet
        Mesh? ResolveMesh()
        {
            var rendered = Entity?.GetComponent<MeshRenderer>()?.Mesh;
            return rendered != null && (Mesh == null || rendered.IsDynamic) ? rendered : Mesh;
        }

        /// <summary>
        /// Give <paramref name="mesh"/> its cooked mesh from the cache; failing that, start its cook on a worker
        /// (<paramref name="now"/> false) or finish or make it (true).
        /// </summary>
        static void Prepare(Mesh mesh, bool now)
        {
            if (mesh.CookedTriMesh is { Disposed: true }) mesh.CookedTriMesh = null;   // let go by the cache
            if (mesh.CookedTriMesh != null) return;

            if (!mesh.PhysicsCookRequested && cookedMeshes.TryGetValue(mesh.PhysicsGeometryKey(), out var hit))
            {
                hit.LastUsed = LoadNumber();
                mesh.CookedTriMesh = hit.Mesh;
                return;
            }

            if (!now)
            {
                mesh.RequestPhysicsCook();
                return;
            }

            mesh.CookPhysicsMesh();
            if (mesh.CookedTriMesh == null) return;

            ulong key = mesh.PhysicsGeometryKey();
            if (cookedMeshes.TryGetValue(key, out var same))
            {
                // Another Mesh of this geometry was cooked meanwhile: one PhysX mesh for both
                PhysXRelease.Release(mesh.CookedTriMesh);
                mesh.CookedTriMesh = same.Mesh;
                same.LastUsed = LoadNumber();
            }
            else
                cookedMeshes[key] = new Cooked { Mesh = mesh.CookedTriMesh, Dynamic = mesh.IsDynamic, LastUsed = LoadNumber() };
        }

        // Every component has Early() before any has Awake(): the cooks of a whole scene's meshes run side by side
        protected override void Early()
        {
            if (Engine.IsEditor) return;
            if (ResolveMesh() is { } mesh) Prepare(mesh, now: false);
        }

        public override PhysX.Geometry? CreateGeometry()
        {
            Mesh = ResolveMesh();
            if(Mesh == null)
            {
                Debug.Log("[RigidBody] No Mesh found for MeshCollider");
                return null;
            }

            using (_cookMarker.Auto())
                Prepare(Mesh, now: true);

            if (Mesh?.CookedTriMesh == null)
            {
                Debug.Log($"[RigidBody] Mesh cooking failed {Mesh.Name} no geometry available");
                return null;
            }

            // The mesh as big as it is drawn: the entity's scale and that of everything above it. The
            // actor's pose carries no scale (see CreateShape), so all of it goes into the geometry.
            // With the entity's own Scale only, a mesh under a scaled parent (a whole assembly
            // scaled up) collided at its unscaled size.
            Transform.Matrix.Deconstruct(out var worldScale, out _, out _);
            return new TriangleMeshGeometry(Mesh.CookedTriMesh)
            {
                Scale = new MeshScale(worldScale, Quaternion.Identity)
            };
        }
    }

    public class TerrainCollider : Collider
    {
        /// <summary>
        /// The terrain's height field (Terrain.CookedHeightFieldAsync: made once per version of the heights and holes).
        /// When it still has to be cooked, there is no shape yet: it is put on the actor once the cook is done, and a
        /// scene load waits for that, so nothing walks or falls before there is ground.
        /// </summary>
        public override PhysX.Geometry? CreateGeometry()
        {
            var terrainRenderer = Entity?.GetComponent<TerrainRenderer>();
            var terrain = terrainRenderer?.Terrain;

            if (terrain == null)
            {
                Debug.Log("[RigidBody] No Terrain found");
                return null;
            }

            if (terrain.HeightField == null)
            {
                Debug.Log("[RigidBody] No Terrain HeightField found");
                return null;
            }

            var cooked = terrain.CookedHeightFieldAsync();
            if (cooked.IsCompleted)
                return cooked.Result is { } field ? Geometry(terrain, field) : null;

            var (root, actor) = (ShapeRoot!, ShapeActor!);
            var attach = cooked.ContinueWith(_ => Engine.RunOnMainThreadAsync(() =>
            {
                // Still wanted on that actor: not destroyed, not replaced, no shape made meanwhile
                if (IsDestroyed || actor.Disposed || ShapeHost != null) return;
                if (cooked.Result != null) CreateShape(root, actor);
            })).Unwrap();
            SceneManager.Loading?.WaitFor(attach, "Terrain collision");
            return null;
        }

        /// <summary>
        /// The samples where the height map's texels are drawn: sample i at (i + 0.5) · size / N from the terrain's corner
        /// (the height bake works at texel centres; TerrainRenderer.GetHeight). Until 2026-10-24 sample i sat at
        /// i · size / (N − 1), up to half a texel off what is drawn towards the edges, so on steep ground near them a
        /// character stood up to a metre off the drawn ground.
        /// </summary>
        private static HeightFieldGeometry Geometry(Terrain terrain, HeightField field)
        {
            // The field was made from these heights (the cache is per version of them)
            float fx = terrain.TerrainSize.X / terrain.HeightField.GetLength(0);
            float fy = terrain.TerrainSize.Y / terrain.HeightField.GetLength(1);
            return new HeightFieldGeometry(field, 0, terrain.MaxHeight / short.MaxValue, fx, fy);
        }

        protected override Vector3 GeometryOffset
        {
            get
            {
                var terrain = Entity?.GetComponent<TerrainRenderer>()?.Terrain;
                var heights = terrain?.HeightField;
                if (heights == null) return Vector3.Zero;
                return new Vector3(0.5f * terrain!.TerrainSize.X / heights.GetLength(0), 0f, 0.5f * terrain.TerrainSize.Y / heights.GetLength(1));
            }
        }
    }
}