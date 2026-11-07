using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;
using System;
using System.Collections.Generic;
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
        private List<Shape>? _pieces;   // a collider of several shapes (CreateGeometries, AddPiece: the terrain's pages); else null

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
                if (_pieces != null)
                    foreach (var piece in _pieces) piece.QueryFilterData = new FilterData(_groups, 0, 0, 0);
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
            if (_isTrigger) return;
            if (_shape != null) ApplyBlocks(_shape);
            if (_pieces != null)
                foreach (var piece in _pieces) ApplyBlocks(piece);
        }

        private void ApplyBlocks(Shape shape)
        {
            const ShapeFlag solid = ShapeFlag.SimulationShape | ShapeFlag.SceneQueryShape;
            shape.Flags = _blocks ? shape.Flags | solid : shape.Flags & ~solid;
        }

        // Kept in a field because it is set before there is a shape (a scene load, the editor).
        // PhysX refuses a shape that is a trigger and a simulation shape at once, so both flags
        // change in one assignment.
        private void ApplyTrigger()
        {
            if (_shape != null) ApplyTrigger(_shape);
            if (_pieces != null)
                foreach (var piece in _pieces) ApplyTrigger(piece);
        }

        private void ApplyTrigger(Shape shape)
        {
            var flags = shape.Flags;
            shape.Flags = _isTrigger
                ? (flags & ~ShapeFlag.SimulationShape) | ShapeFlag.TriggerShape
                : (flags & ~ShapeFlag.TriggerShape) | ShapeFlag.SimulationShape;

            if (_isTrigger && (shape.Flags & ShapeFlag.TriggerShape) == 0)
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

        /// <summary>
        /// The pieces of a collider made of several shapes, each a geometry and where its origin sits in the collider's
        /// frame (TerrainCollider: a height field per page, since 2026-11-07). Null for a collider of one shape, which
        /// <see cref="CreateGeometry"/> makes; empty for one with no shape yet.
        /// </summary>
        protected virtual IReadOnlyList<(PhysX.Geometry Geometry, Vector3 Position)>? CreateGeometries() => null;

        /// <summary>Where the geometry's origin sits in the collider's frame, before <see cref="Offset"/>.</summary>
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

            IReadOnlyList<(PhysX.Geometry Geometry, Vector3 Position)>? pieces;
            using (_geometryMarker.Auto())
                pieces = CreateGeometries();
            if (pieces != null)
            {
                CreatePieces(root, actor, pieces);
                return;
            }

            PhysX.Geometry? geometry;
            using (_geometryMarker.Auto())
                geometry = CreateGeometry();
            if (geometry == null)
                return;

            using var shapeScope = _shapeMarker.Auto();
            _shape = NewShape(actor, geometry);
            _shapeHost = root;

            // Offset is in the collider's own frame, scaled with it; on a child of the actor's entity
            // that frame is carried into the actor's.
            var localPose = Matrix4x4.CreateTranslation(Offset * WorldScale + GeometryOffset);

            // A PhysX capsule runs along its X axis; ours stands along the entity's Y.
            if (geometry is CapsuleGeometry)
                localPose = Matrix4x4.CreateRotationZ(MathF.PI / 2) * localPose;

            localPose *= FrameInRoot(root);
            if (!localPose.IsIdentity)
                _shape.LocalPose = localPose;
        }

        // A shape of this collider on the actor, with its groups, trigger and blocking as set
        private Shape NewShape(RigidActor actor, PhysX.Geometry geometry)
        {
            var shape = RigidActorExt.CreateExclusiveShape(actor, geometry, PhysicsWorld.DefaultMaterial);

            // Whose shape this is: under a RigidBody all colliders share the body's actor, whose
            // UserData is the body's entity, so a query's hit can only tell the piece by its shape
            shape.UserData = this;
            shape.QueryFilterData = new FilterData(_groups, 0, 0, 0);

            if (_isTrigger)
                ApplyTrigger(shape);
            if (!_blocks && !_isTrigger)
                ApplyBlocks(shape);
            return shape;
        }

        // Every piece's shape, each where CreateGeometries put it in the collider's frame
        private void CreatePieces(Entity root, RigidActor actor, IReadOnlyList<(PhysX.Geometry Geometry, Vector3 Position)> pieces)
        {
            if (pieces.Count == 0) return;

            using var shapeScope = _shapeMarker.Auto();
            var frame = FrameInRoot(root);
            var shapes = new List<Shape>(pieces.Count);
            for (int i = 0; i < pieces.Count; i++)
            {
                var shape = NewShape(actor, pieces[i].Geometry);
                var localPose = Matrix4x4.CreateTranslation(Offset * WorldScale + GeometryOffset + pieces[i].Position) * frame;
                if (!localPose.IsIdentity)
                    shape.LocalPose = localPose;
                shapes.Add(shape);
            }
            _pieces = shapes;
            _shapeHost = root;
        }

        /// <summary>
        /// One more piece on the actor the collider's shapes are on (<see cref="ShapeActor"/>), its origin at
        /// <paramref name="position"/> in the collider's frame: for a collider whose pieces come and go (TerrainCollider:
        /// the pages round the viewers, TERRAIN_PAGES_PLAN.md 5c). Null before the collider has an actor. Main thread.
        /// </summary>
        protected Shape? AddPiece(PhysX.Geometry geometry, Vector3 position)
        {
            if (ShapeActor == null || ShapeActor.Disposed || ShapeRoot == null) return null;
            using var shapeScope = _shapeMarker.Auto();
            var shape = NewShape(ShapeActor, geometry);
            var localPose = Matrix4x4.CreateTranslation(Offset * WorldScale + GeometryOffset + position) * FrameInRoot(ShapeRoot);
            if (!localPose.IsIdentity)
                shape.LocalPose = localPose;
            (_pieces ??= new()).Add(shape);
            _shapeHost = ShapeRoot;
            return shape;
        }

        /// <summary>A piece <see cref="AddPiece"/> made, taken off its actor and released. Main thread.</summary>
        protected void RemovePiece(Shape shape)
        {
            if (_pieces == null || !_pieces.Remove(shape)) return;
            PhysXRelease.Release(shape, true);   // an exclusive shape's release detaches it
        }

        // The collider's frame in the actor's: identity on its own actor
        private Matrix4x4 FrameInRoot(Entity root)
        {
            if (root == Entity) return Matrix4x4.Identity;

            // The actor's frame is the root's place and turn in the world without its scale (a
            // PhysX pose carries none), so the shape is placed in that frame: in metres. Taken
            // relative to the root's scaled matrix, a root (or anything above it) scaled by s put
            // every shape s times too near the root's pivot.
            root.Transform.Matrix.Deconstruct(out _, out var rootRotation, out var rootTranslation);
            var rootFrame = Matrix4x4.CreateFromQuaternion(rootRotation) * Matrix4x4.CreateTranslation(rootTranslation);
            Matrix4x4.Invert(rootFrame, out var rootInverse);
            var relative = Matrix4x4.Multiply(Transform.Matrix, rootInverse);
            relative.Deconstruct(out _, out var shapeRotation, out var shapeTranslation);
            return Matrix4x4.CreateFromQuaternion(shapeRotation) * Matrix4x4.CreateTranslation(shapeTranslation);
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
            if (_pieces != null)
                foreach (var piece in _pieces) PhysXRelease.Release(piece, native);
            _pieces = null;
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

    /// <summary>
    /// The terrain's collision: a PhysX height field per page (TERRAIN_PAGES_PLAN.md 4c), only for the pages within
    /// <c>EngineSettings.TerrainCollisionRadius</c> of a viewer (Vicinity.Viewers, else the main camera; 5c), each a piece
    /// on the actor at its page's corner, the fields sharing their edge samples. Pages are cooked on workers as they come in
    /// range (<see cref="TerrainCollisionSystem"/> looks a few times a second) and dropped a quarter further out; an edit
    /// cooks again only the pages whose heights or holes changed (the store's page arrays are never changed, so a changed
    /// page is a new array). A scene load waits until the pages round the viewers are in, so nothing walks or falls before
    /// there is ground. Until 4c one field spanned the whole terrain; until 5c every page was cooked at play.
    /// </summary>
    public class TerrainCollider : Collider
    {
        private static readonly IReadOnlyList<(PhysX.Geometry, Vector3)> NoPieces = Array.Empty<(PhysX.Geometry, Vector3)>();

        /// <summary>A page that has collision: its piece, its field, and the arrays it was cooked from.</summary>
        private sealed class PagePiece
        {
            public required Shape Shape;
            public required HeightField Field;
            public object? Heights, Holes;
        }

        private readonly Dictionary<int, PagePiece> _pages = new();   // by page index (pz · across + px)
        private RigidActor? _pagesActor;      // the actor the pieces are on: another one (a body made again) starts over
        private TerrainPages? _pagesStore;    // the store whose layout the pieces follow
        private Cook? _cook;                  // a batch cooking on workers
        private float _sinceLook = float.MaxValue;
        private Vector3 _lookedFrom = new(float.NaN);

        /// <summary>Pages with collision now.</summary>
        [Reflection.DontSerialize]
        public int CollisionPages => _pages.Count;

        /// <summary>Made of pieces: <see cref="CreateGeometries"/>, then <see cref="AddPiece"/> per page.</summary>
        public override PhysX.Geometry? CreateGeometry() => null;

        /// <summary>
        /// No pieces at first: the pages come as the system finds them round the viewers. A scene load waits for the first
        /// pages round them (its waiting step, when every entity, the player's camera and its viewer included, is awake).
        /// </summary>
        protected override IReadOnlyList<(PhysX.Geometry Geometry, Vector3 Position)>? CreateGeometries()
        {
            TerrainCollisionSystem.Register(this);
            _sinceLook = float.MaxValue;
            SceneManager.Loading?.WaitFor(_ => UpdatePages(Engine.Settings.TerrainCollisionRadius, force: true), "Terrain collision");
            return NoPieces;
        }

        public override void Destroy()
        {
            TerrainCollisionSystem.Unregister(this);
            base.Destroy();   // the pieces go with the actor or are detached (Collider.ReleaseShape)
            foreach (var page in _pages.Values) PhysXRelease.Release(page.Field);
            _pages.Clear();
            _cook?.Abandon();
            _cook = null;
        }

        /// <summary>
        /// Bring the pages round the viewers in and let go of those beyond them: a finished cook is put on the actor, the
        /// pages wanted and not cooked from the store as it is now go to the workers (one batch at a time). Returns true
        /// when every wanted page has its piece. <paramref name="force"/>: look now, not only every quarter second or once
        /// the viewers have moved. Main thread (TerrainCollisionSystem, a scene load's waiting step).
        /// </summary>
        internal bool UpdatePages(float radius, bool force = false)
        {
            if (IsDestroyed || ShapeActor == null || ShapeActor.Disposed) return true;
            var terrain = Entity?.GetComponent<TerrainRenderer>()?.Terrain;
            if (terrain?.Pages is not { HasHeights: true } store || Transform == null) return true;   // no ground to make

            // Pieces made on an actor since replaced are gone with it
            if (!ReferenceEquals(_pagesActor, ShapeActor))
            {
                foreach (var page in _pages.Values) PhysXRelease.Release(page.Field);
                _pages.Clear();
                _pagesActor = ShapeActor;
            }
            // Another layout (a terrain of another size or density): every page again
            if (_pagesStore != null && (_pagesStore.Resolution != store.Resolution))
                DropAll();
            _pagesStore = store;

            // Ground under a viewer at once: a page within UnderFoot metres of one that has no collision (a teleport, a
            // portal, a bindstone) is cooked here on the main thread, ~20 ms a page, rather than fallen through while a
            // batch cooks
            var viewers = ViewersLocal(terrain);
            CookUnderViewers(store, terrain, viewers);

            if (_cook != null)
            {
                if (!_cook.Task.IsCompleted) return false;
                Apply(_cook, terrain);
                _cook = null;
                force = true;   // what changed while it cooked is seen at once
            }

            // Look a few times a second, or when asked, or when the viewers have moved a tenth of the radius
            Vector3 lead = viewers.Count > 0 ? viewers[0] : Vector3.Zero;
            _sinceLook += Time.Delta;
            if (!force && _sinceLook < 0.25f && Vector3.DistanceSquared(lead, _lookedFrom) < (radius * 0.1f) * (radius * 0.1f))
                return _cook == null;
            _sinceLook = 0f;
            _lookedFrom = lead;

            int across = store.PagesAcross;
            float fx = terrain.TerrainSize.X / (store.Resolution - 1), fz = terrain.TerrainSize.Y / (store.Resolution - 1);
            bool all = radius <= 0f;
            float keep = radius * 1.25f;
            List<int>? wanted = null;
            List<int>? gone = null;
            for (int pz = 0; pz < across; pz++)
                for (int px = 0; px < across; px++)
                {
                    int index = pz * across + px;
                    var (x0, z0, w, h) = store.PageExtent(px, pz);
                    float d = all ? 0f : Distance(viewers, x0 * fx, z0 * fz, (x0 + w - 1) * fx, (z0 + h - 1) * fz);
                    bool has = _pages.TryGetValue(index, out var piece);
                    if (has && d > keep) { (gone ??= new()).Add(index); continue; }
                    if (d > radius && !all) continue;
                    var (heights, holes) = store.PageIdentity(px, pz);
                    if (has && ReferenceEquals(piece!.Heights, heights) && ReferenceEquals(piece.Holes, holes)) continue;
                    if (heights == null) continue;   // nothing in that page yet
                    (wanted ??= new()).Add(index);
                }

            if (gone != null)
                foreach (int index in gone)
                {
                    var piece = _pages[index];
                    RemovePiece(piece.Shape);
                    PhysXRelease.Release(piece.Field);
                    _pages.Remove(index);
                }

            if (wanted == null) return true;
            _cook = Cook.Start(store, wanted, terrain.Name);
            return false;
        }

        private void DropAll()
        {
            foreach (var piece in _pages.Values)
            {
                RemovePiece(piece.Shape);
                PhysXRelease.Release(piece.Field);
            }
            _pages.Clear();
        }

        // The viewers on the terrain's ground plane, in terrain-local metres (x, _, z)
        private List<Vector3> ViewersLocal(Terrain terrain)
        {
            var origin = Transform.WorldPosition;
            var list = new List<Vector3>(Math.Max(1, Vicinity.Viewers.Count));
            foreach (var viewer in Vicinity.Viewers)
                if (viewer != null && !viewer.IsDestroyed) list.Add(viewer.WorldPosition - origin);
            if (list.Count == 0 && Camera.Main != null) list.Add(Camera.Main.Position - origin);
            return list;
        }

        // The nearest viewer's distance to a page's square on the ground (0 inside it)
        private static float Distance(List<Vector3> viewers, float x0, float z0, float x1, float z1)
        {
            float best = float.MaxValue;
            foreach (var v in viewers)
            {
                float dx = MathF.Max(0f, MathF.Max(x0 - v.X, v.X - x1));
                float dz = MathF.Max(0f, MathF.Max(z0 - v.Z, v.Z - z1));
                best = MathF.Min(best, MathF.Sqrt(dx * dx + dz * dz));
            }
            return best;
        }

        /// <summary>Metres round a viewer within which a page without collision is cooked at once.</summary>
        private const float UnderFoot = 64f;

        private void CookUnderViewers(TerrainPages store, Terrain terrain, List<Vector3> viewers)
        {
            int across = store.PagesAcross;
            float fx = terrain.TerrainSize.X / (store.Resolution - 1), fz = terrain.TerrainSize.Y / (store.Resolution - 1);
            float pageX = TerrainPages.PageCells * fx, pageZ = TerrainPages.PageCells * fz;
            List<int>? urgent = null;
            foreach (var v in viewers)
            {
                int px0 = Math.Clamp((int)MathF.Floor((v.X - UnderFoot) / pageX), 0, across - 1);
                int px1 = Math.Clamp((int)MathF.Floor((v.X + UnderFoot) / pageX), 0, across - 1);
                int pz0 = Math.Clamp((int)MathF.Floor((v.Z - UnderFoot) / pageZ), 0, across - 1);
                int pz1 = Math.Clamp((int)MathF.Floor((v.Z + UnderFoot) / pageZ), 0, across - 1);
                for (int pz = pz0; pz <= pz1; pz++)
                    for (int px = px0; px <= px1; px++)
                    {
                        int index = pz * across + px;
                        if (_pages.ContainsKey(index) || store.PageIdentity(px, pz).heights == null) continue;
                        if (urgent == null || !urgent.Contains(index)) (urgent ??= new()).Add(index);
                    }
            }
            if (urgent == null) return;

            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var cooking = PhysicsWorld.Physics.CreateCooking();
            try
            {
                foreach (int index in urgent)
                {
                    int px = index % across, pz = index / across;
                    var (_, _, w, h) = store.PageExtent(px, pz);
                    var desc = new PhysX.HeightFieldDesc { NumberOfRows = w, NumberOfColumns = h, Samples = store.PhysXSamples(px, pz) };
                    var stream = new System.IO.MemoryStream(w * h * 4 + 1024);
                    cooking.CookHeightField(desc, stream);
                    stream.Position = 0;
                    Attach(index, PhysicsWorld.Physics.CreateHeightField(stream), store, terrain);
                }
            }
            finally
            {
                PhysXRelease.Release(cooking);
            }
            Debug.LogAlways($"[Terrain] '{terrain.Name}': collision for {urgent.Count} page(s) under a viewer cooked at once in " +
                            $"{System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds:0} ms");
        }

        // A field for a page put on the actor as its piece; the piece it replaces goes
        private void Attach(int index, HeightField field, TerrainPages store, Terrain terrain)
        {
            int across = store.PagesAcross;
            float fx = terrain.TerrainSize.X / (store.Resolution - 1), fz = terrain.TerrainSize.Y / (store.Resolution - 1);
            int px = index % across, pz = index / across;
            var (x0, z0, _, _) = store.PageExtent(px, pz);
            var shape = AddPiece(new HeightFieldGeometry(field, 0, terrain.MaxHeight / short.MaxValue, fx, fz), new Vector3(x0 * fx, 0, z0 * fz));
            if (shape == null)
            {
                PhysXRelease.Release(field);
                return;
            }
            if (_pages.TryGetValue(index, out var old))
            {
                RemovePiece(old.Shape);
                PhysXRelease.Release(old.Field);   // shapes still on it keep it alive (PhysX counts references)
            }
            var (heights, holes) = store.PageIdentity(px, pz);
            _pages[index] = new PagePiece { Shape = shape, Field = field, Heights = heights, Holes = holes };
        }

        // A finished batch: each page still in the store as it was cooked gets its field and piece (one it replaces goes)
        private void Apply(Cook cook, Terrain terrain)
        {
            var streams = cook.Task.IsCompletedSuccessfully ? cook.Task.Result : null;
            cook.ReleaseCookings();
            if (streams == null)
            {
                Debug.LogWarning("Terrain", $"'{terrain.Name}': cooking the collision failed: {cook.Task.Exception?.GetBaseException().Message}");
                return;
            }
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i = 0; i < cook.Pages.Count; i++)
                if (streams[i] != null)
                    Attach(cook.Pages[i], PhysicsWorld.Physics.CreateHeightField(streams[i]), cook.Store, terrain);
            double makeMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            Debug.LogAlways($"[Terrain] '{terrain.Name}': collision for {cook.Pages.Count} page(s) cooked on {cook.Workers} workers in " +
                            $"{cook.CookMs:0} ms, {makeMs:0} ms making the fields and pieces; {_pages.Count} page(s) with collision");
        }

        /// <summary>
        /// Height fields cooked on workers, each with a cooking object of its own (PhysX.Net makes those, and the height
        /// fields, on the main thread: its object table is not thread-safe). PhysX.Net's HeightFieldSample is a class, so each
        /// worker fills one array of them again page by page.
        /// </summary>
        private sealed class Cook
        {
            public required TerrainPages Store;
            public required List<int> Pages;
            public required Task<System.IO.MemoryStream?[]> Task;
            public required PhysX.Cooking[] Cookings;
            public int Workers => Cookings.Length;
            public double CookMs;
            private bool _released;

            public static Cook Start(TerrainPages store, List<int> pages, string name)
            {
                int workers = Math.Clamp(Environment.ProcessorCount - 1, 1, pages.Count);
                var cookings = new PhysX.Cooking[workers];
                for (int k = 0; k < workers; k++)
                    cookings[k] = PhysicsWorld.Physics.CreateCooking();
                int across = store.PagesAcross;
                Cook? cook = null;
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                var task = System.Threading.Tasks.Task.Run(() =>
                {
                    var streams = new System.IO.MemoryStream?[pages.Count];
                    System.Threading.Tasks.Parallel.For(0, workers, k =>
                    {
                        PhysX.HeightFieldSample[]? reuse = null;
                        for (int i = k; i < pages.Count; i += workers)
                        {
                            int px = pages[i] % across, pz = pages[i] / across;
                            var (_, _, w, h) = store.PageExtent(px, pz);
                            var samples = store.PhysXSamples(px, pz, reuse);
                            if (samples.Length == TerrainPages.PageLength) reuse = samples;
                            var desc = new PhysX.HeightFieldDesc { NumberOfRows = w, NumberOfColumns = h, Samples = samples };
                            var stream = new System.IO.MemoryStream(w * h * 4 + 1024);
                            cookings[k].CookHeightField(desc, stream);
                            stream.Position = 0;
                            streams[i] = stream;
                        }
                    });
                    if (cook != null) cook.CookMs = System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                    return streams;
                });
                cook = new Cook { Store = store, Pages = pages, Task = task, Cookings = cookings };
                return cook;
            }

            public void ReleaseCookings()
            {
                if (_released) return;
                _released = true;
                foreach (var cooking in Cookings) PhysXRelease.Release(cooking);
            }

            /// <summary>The collider went: the cookings are released once the workers are done with them.</summary>
            public void Abandon() => Task.ContinueWith(_ => Engine.RunOnMainThreadAsync(ReleaseCookings));
        }
    }

    /// <summary>
    /// Keeps every TerrainCollider's pages round the viewers (TERRAIN_PAGES_PLAN.md 5c). Play mode only: the editor makes no
    /// colliders.
    /// </summary>
    public sealed class TerrainCollisionSystem : EntitySystem
    {
        private static readonly List<TerrainCollider> s_colliders = new();

        internal static void Register(TerrainCollider collider)
        {
            if (!s_colliders.Contains(collider)) s_colliders.Add(collider);
        }

        internal static void Unregister(TerrainCollider collider) => s_colliders.Remove(collider);

        public override void Update()
        {
            float radius = Engine.Settings.TerrainCollisionRadius;
            for (int i = s_colliders.Count - 1; i >= 0; i--)
                s_colliders[i].UpdatePages(radius);
        }
    }
}
