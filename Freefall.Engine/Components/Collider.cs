using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using PhysX;
using System;
using System.IO;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Components
{
    [Icon("icon_collider.png")]
    public abstract class Collider : Component
    {
        private RigidStatic? _staticActor;
        private Shape? _shape;

        public Actor? Actor => _staticActor;

        /// <summary>
        /// Where the shape sits, in metres in the entity's frame (rotated with it, not scaled).
        /// Read when the shape is made.
        /// </summary>
        public Vector3 Offset;

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
                var rigidBody = Entity?.GetComponentInParents<RigidBody>();
                if (rigidBody != null)
                    return;

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

        void CreateFullActor()
        {
            // Remove old actor
            if (Actor != null)
            {
                PhysicsWorld.Scene.RemoveActor(Actor);
                Actor.Dispose();
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

        public void CreateShape(Entity root, RigidActor actor)
        {
            var geometry = CreateGeometry();
            if (geometry == null)
                return;

            _shape = RigidActorExt.CreateExclusiveShape(actor, geometry, PhysicsWorld.DefaultMaterial);

            // Whose shape this is: under a RigidBody all colliders share the body's actor, whose
            // UserData is the body's entity, so a query's hit can only tell the piece by its shape
            _shape.UserData = this;
            _shape.QueryFilterData = new FilterData(_groups, 0, 0, 0);

            if (_isTrigger)
                ApplyTrigger();

            // Offset is in the collider's own frame; on a child of the actor's entity that frame
            // is carried into the actor's.
            var localPose = Matrix4x4.CreateTranslation(Offset);

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
                PhysicsWorld.Scene.RemoveActor(Actor);
                Actor.Dispose();
            }

            _shape?.Dispose();
            _staticActor?.Dispose();
        }
    }


    public class BoxCollider : Collider, ISceneGizmo
    {
        /// <summary>Full edge lengths in metres. The entity's scale does not change them.</summary>
        public Vector3 Size = Vector3.One;

        public override PhysX.Geometry? CreateGeometry()
        {
            return new BoxGeometry(Size * 0.5f);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawWireBox(Offset, Size);
        }
    }

    public class SphereCollider : Collider, ISceneGizmo
    {
        public float Radius = 0.5f;
        public override PhysX.Geometry? CreateGeometry()
        {
            return new SphereGeometry(Radius);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawWireSphere(Offset, Radius);
        }
    }

    public class CapsuleCollider : Collider, ISceneGizmo
    {
        public float Radius = 0.5f;

        /// <summary>From end to end along the entity's Y, caps included. Never less than a ball of the radius.</summary>
        public float Height = 2.0f;

        public override PhysX.Geometry? CreateGeometry()
        {
            // PhysX takes half the cylinder between the caps, and refuses a capsule that has
            // none: that one is a sphere.
            float halfCylinder = Height * 0.5f - Radius;
            if (halfCylinder <= 0f)
                return new SphereGeometry(Radius);

            return new CapsuleGeometry(Radius, halfCylinder);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawCapsule(Offset, MathF.Max(Height, Radius * 2f), Radius);
        }
    }

    /// <summary>
    /// The entity's mesh (or <see cref="Mesh"/>) as a static triangle mesh, at the size it is drawn:
    /// the entity's scale times its parents'. Cooked once per mesh.
    /// </summary>
    public class MeshCollider : Collider
    {
        public Mesh? Mesh;
        static Dictionary<string, TriangleMesh> cookedMeshes = [];

        public override PhysX.Geometry? CreateGeometry()
        {
            var meshrenderer = Entity?.GetComponent<MeshRenderer>();

            if (meshrenderer?.Mesh != null)
            {
                if(Mesh == null || meshrenderer.Mesh.IsDynamic)
                    Mesh = meshrenderer.Mesh;
            }
            
            if(Mesh == null)
            {
                Debug.Log("[RigidBody] No Mesh found for MeshCollider");
                return null;
            }


            if (Mesh?.CookedTriMesh == null)
            {
                if (!cookedMeshes.TryGetValue(Mesh.Guid, out var triMesh))
                {
                    Mesh?.CookPhysicsMesh();
                    if (Mesh?.CookedTriMesh != null)
                    {
                        cookedMeshes[Mesh.Guid] = Mesh.CookedTriMesh;
                        triMesh = Mesh.CookedTriMesh;
                    }
                }

                Mesh?.CookedTriMesh = triMesh;
            }

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

            var heightMap = terrain.HeightField;
            int rows = heightMap.GetLength(0);
            int cols = heightMap.GetLength(1);

            float fx = terrain.TerrainSize.X / (rows - 1);
            float fy = terrain.TerrainSize.Y / (cols - 1);

            Debug.Log($"[RigidBody] Cooking HeightField: {rows}x{cols}, MaxHeight={terrain.MaxHeight}, fx={fx}, fy={fy}");
            var samples = heightMap.ToSamples();

            var heightFieldDesc = new HeightFieldDesc()
            {
                NumberOfRows = rows,
                NumberOfColumns = cols,
                Samples = samples,
            };

            var cooking = PhysicsWorld.Physics.CreateCooking();
            var stream = new MemoryStream();
            cooking.CookHeightField(heightFieldDesc, stream);

            stream.Position = 0;
            HeightField heightField = PhysicsWorld.Physics.CreateHeightField(stream);

            return new HeightFieldGeometry(heightField, 0, terrain.MaxHeight / short.MaxValue, fx, fy);
        }
    }
}