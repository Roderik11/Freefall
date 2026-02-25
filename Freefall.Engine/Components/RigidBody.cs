using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Vortice.Mathematics;
using PhysX;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Assets;

namespace Freefall.Components
{
    public enum ShapeType
    {
        Box,
        Sphere,
        Capsule,
        Mesh,
        StaticMesh,
        Terrain
    }

    public class RigidBody : Component
    {
        private static Dictionary<int, TriangleMesh> _triMeshCache = new();

        private bool _isStatic;
        private RigidStatic? _staticActor;
        private RigidDynamic? _dynamicActor;
        private Shape? _shape;

        public Actor? Actor => (Actor?)_staticActor ?? _dynamicActor;

        public StaticMesh? StaticMesh { get; set; }
        public Graphics.Mesh? Mesh { get; set; }
        public ShapeType Type { get; set; } = ShapeType.Box;
        public float Mass { get; set; } = 10f;

        public bool IsStatic
        {
            get => _isStatic;
            set
            {
                if (_isStatic == value) return;
                _isStatic = value;

                // Recreate actor if already alive
                if (_staticActor != null || _dynamicActor != null)
                    CreateBody();
            }
        }

        protected override void Awake()
        {
            try
            {
                // Auto-resolve StaticMesh from sibling StaticMeshRenderer if not set
                if (StaticMesh == null && Type == ShapeType.StaticMesh)
                {
                    var smr = Entity?.GetComponent<StaticMeshRenderer>();
                    if (smr?.StaticMesh != null)
                        StaticMesh = smr.StaticMesh;
                }

                CreateBody();

                Transform.OnChanged += () =>
                {
                    if (IsStatic && _staticActor != null)
                    {
                        var matrix = Matrix4x4.CreateFromQuaternion(Transform.Rotation)
                                   * Matrix4x4.CreateTranslation(Transform.WorldPosition);

                        _staticActor.GlobalPose = matrix;
                        _staticActor.GlobalPosePosition = matrix.Translation;
                        _staticActor.GlobalPoseQuat = Transform.Rotation;
                    }
                };
            }
            catch (Exception ex)
            {
                Debug.Log($"[RigidBody] {Entity?.Name}: {ex.Message}");
            }
        }

        private PhysX.Geometry? CreateGeometry()
        {
            BoundingBox bb = new BoundingBox();

            if (Mesh != null)
                bb = Mesh.BoundingBox;

            Vector3 scale = Transform?.Scale ?? Vector3.One;
            Vector3 extents = bb.Size * scale;

            switch (Type)
            {
                case ShapeType.Box:
                    return new BoxGeometry(extents * 2f);

                case ShapeType.Sphere:
                    float radius = Math.Max(Math.Max(extents.X, extents.Y), extents.Z) - 0.05f;
                    return new SphereGeometry(Math.Max(radius, 0.01f));

                case ShapeType.Capsule:
                    return new CapsuleGeometry(extents.Y, extents.X);

                case ShapeType.StaticMesh:
                {
                    // Use pre-cooked TriangleMesh if available, otherwise cook at runtime
                    if (StaticMesh?.CookedTriMesh == null)
                        StaticMesh?.CookPhysicsMesh();

                    if (StaticMesh?.CookedTriMesh == null)
                    {
                        Debug.Log("[RigidBody] StaticMesh cooking failed — no geometry available");
                        return null;
                    }

                    return new TriangleMeshGeometry(StaticMesh.CookedTriMesh)
                    {
                        Scale = new MeshScale(scale, Quaternion.Identity)
                    };
                }

                case ShapeType.Mesh:
                {
                    if (Mesh == null || Mesh.Positions == null || Mesh.CpuIndices == null)
                    {
                        Debug.Log("[RigidBody] Mesh cooking failed — no CPU vertex data retained");
                        return null;
                    }

                    var hash = Mesh.GetInstanceId();
                    if (!_triMeshCache.TryGetValue(hash, out var triangleMesh))
                    {
                        var cooking = PhysicsWorld.Physics.CreateCooking();

                        var triangleMeshDesc = new TriangleMeshDesc()
                        {
                            Flags = (MeshFlag)0,
                            Triangles = Array.ConvertAll(Mesh.CpuIndices, i => (int)i),
                            Points = Mesh.Positions
                        };

                        var stream = new MemoryStream();
                        cooking.CookTriangleMesh(triangleMeshDesc, stream);

                        stream.Position = 0;
                        triangleMesh = PhysicsWorld.Physics.CreateTriangleMesh(stream);

                        _triMeshCache.Add(hash, triangleMesh);
                    }

                    return new TriangleMeshGeometry(triangleMesh)
                    {
                        Scale = new MeshScale(scale, Quaternion.Identity)
                    };
                }

                case ShapeType.Terrain:
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

            return null;
        }

        public void CreateBody()
        {
            var geometry = CreateGeometry();
            if (geometry == null) return;

            // Remove old actor
            if (Actor != null)
            {
                PhysicsWorld.Scene.RemoveActor(Actor);
                Actor.Dispose();
                _staticActor = null;
                _dynamicActor = null;
            }

            // Build a clean pose from position + rotation (no scale).
            // PhysX requires an orthonormal rotation — scale is applied via geometry.
            var matrix = Matrix4x4.CreateFromQuaternion(Transform.Rotation)
                       * Matrix4x4.CreateTranslation(Transform.WorldPosition);

            if (IsStatic)
            {
                _staticActor = PhysicsWorld.Physics.CreateRigidStatic();
                _staticActor.UserData = Entity;
                _staticActor.GlobalPose = matrix;
                _staticActor.GlobalPosePosition = matrix.Translation;
                _staticActor.GlobalPoseQuat = Transform.Rotation;
                _staticActor.Flags &= ~ActorFlag.Visualization;
                _staticActor.Flags |= ActorFlag.DisableGravity;

                _shape = RigidActorExt.CreateExclusiveShape(_staticActor, geometry, PhysicsWorld.DefaultMaterial);
            }
            else
            {
                _dynamicActor = PhysicsWorld.Physics.CreateRigidDynamic();
                _dynamicActor.UserData = Entity;
                _dynamicActor.GlobalPose = matrix;
                _dynamicActor.AngularDamping = 1f;
                _dynamicActor.LinearDamping = 0.1f;
                _dynamicActor.SetMassAndUpdateInertia(Mass);

                _shape = RigidActorExt.CreateExclusiveShape(_dynamicActor, geometry, PhysicsWorld.DefaultMaterial);
            }

            PhysicsWorld.Scene.AddActor(Actor!);
        }

        public override void Destroy()
        {
            if (Actor != null)
            {
                PhysicsWorld.Scene.RemoveActor(Actor);
                Actor.Dispose();
                _staticActor = null;
                _dynamicActor = null;
            }
        }
    }
}
