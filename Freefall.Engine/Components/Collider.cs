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

        public Vector3 Offset;

        public bool IsTrigger
        {
            get => _shape != null && (_shape.Flags & ShapeFlag.TriggerShape) != 0;
            set
            {
                if (_shape == null) return;
                if (value)
                    _shape.Flags |= ShapeFlag.TriggerShape;
                else
                    _shape.Flags &= ~ShapeFlag.TriggerShape;
            }
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
            if(root != Entity)
            {
                Matrix4x4.Invert(root.Transform.Matrix, out var worldInverse);
                var relativeToParent = Matrix4x4.Multiply(Transform.Matrix, worldInverse);
                relativeToParent.Deconstruct(out var shapeScale, out var shapeRotation, out var shapeTranslation);
                var shapeWorldMatrix = Matrix4x4.CreateFromQuaternion(shapeRotation) * Matrix4x4.CreateTranslation(shapeTranslation);
                _shape.LocalPose = shapeWorldMatrix;
            }
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
        public Vector3 Size = Vector3.One;

        protected override void Awake()
        {
            var meshrenderer = Entity?.GetComponent<MeshRenderer>();
            if (meshrenderer != null && meshrenderer.Mesh != null)
            {
                var bounds = meshrenderer.Mesh.BoundingBox;
                Size = bounds.Size * 2;
                Offset = bounds.Center;
            }
        }

        public override PhysX.Geometry? CreateGeometry()
        {
            return new BoxGeometry(Size * 0.5f);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawWireBox(Offset, Size * 0.5f);
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
        public float Height = 2.0f;

        public override PhysX.Geometry? CreateGeometry()
        {
            return new CapsuleGeometry(Radius, Height * 0.5f);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            ctx.Color = new Color4(0.3f, 0.8f, 1f, 1f);
            ctx.DrawCapsule(Offset, Height, Radius);
        }
    }

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

            return new TriangleMeshGeometry(Mesh.CookedTriMesh)
            {
                Scale = new MeshScale(Transform.Scale, Quaternion.Identity)
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

            float fx = terrain.TerrainSize.X / ((terrain.HeightField?.GetLength(0) ?? 2) - 1);
            float fy = terrain.TerrainSize.Y / ((terrain.HeightField?.GetLength(1) ?? 2) - 1);

            // Fast path: use pre-cooked HeightField from import
            if (terrain.CookedHeightField != null)
            {
                Debug.Log($"[RigidBody] Using CookedHeightField: MaxHeight={terrain.MaxHeight}, scale={terrain.MaxHeight / short.MaxValue}, fx={fx}, fy={fy}");
                return new HeightFieldGeometry(terrain.CookedHeightField, 0,
                    terrain.MaxHeight / short.MaxValue, fx, fy);
            }

            // Slow fallback: cook on demand
            if (terrain.HeightField == null)
            {
                Debug.Log("[RigidBody] No Terrain HeightField found");
                return null;
            }

            var heightMap = terrain.HeightField;
            int rows = heightMap.GetLength(0);
            int cols = heightMap.GetLength(1);

            Debug.Log($"[RigidBody] Cooking HeightField on demand: {rows}x{cols}, MaxHeight={terrain.MaxHeight}, fx={fx}, fy={fy}");
            var samples = heightMap.ToSamples();

            var heightFieldDesc = new HeightFieldDesc()
            {
                NumberOfRows = rows,
                NumberOfColumns = cols,
                Samples = samples,
            };

            var cooking = PhysicsWorld.Physics.CreateCooking();
            var stream = new MemoryStream();
            bool cookResult = cooking.CookHeightField(heightFieldDesc, stream);

            stream.Position = 0;
            HeightField heightField = PhysicsWorld.Physics.CreateHeightField(stream);

            return new HeightFieldGeometry(heightField, 0, terrain.MaxHeight / short.MaxValue, fx, fy);
        }
    }
}