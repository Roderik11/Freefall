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

    public class RigidBody : Component, IUpdate
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
                CreateBody();
            }
            catch (Exception ex)
            {
                Debug.Log($"[RigidBody] {Entity?.Name}: {ex.Message}");
            }
        }

        public void Update()
        {
            if (Actor == null) return;

            // Dynamic bodies: read pose from PhysX → Transform
            if (!IsStatic && _dynamicActor != null && _dynamicActor.IsSleeping == false)
            {
                Transform.Position = _dynamicActor.GlobalPosePosition;
                Transform.Rotation = _dynamicActor.GlobalPoseQuat;
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
                    // Resolve mesh from StaticMesh asset
                    var sm = StaticMesh?.Mesh ?? Mesh;
                    if (sm == null || sm.Positions == null || sm.CpuIndices == null)
                    {
                        Debug.Log("[RigidBody] StaticMesh cooking failed — no CPU vertex data retained");
                        return null;
                    }

                    var smHash = sm.GetInstanceId();

                    // Use lowest LOD indices for collision (Apex pattern) 
                    // Dramatically fewer triangles for physics
                    if (StaticMesh?.LODs?.Count > 0)
                        smHash = smHash * 31 + StaticMesh.LODs.Count; // Unique cache key for LOD variant

                    if (!_triMeshCache.TryGetValue(smHash, out var smTriMesh))
                    {
                        int[] triangles;

                        // Pick lowest LOD mesh parts for collision
                        if (StaticMesh?.LODs?.Count > 0)
                        {
                            var lowestLod = StaticMesh.LODs[StaticMesh.LODs.Count - 1];
                            var lodIndices = new List<int>();
                            foreach (var part in lowestLod.MeshParts)
                            {
                                var meshPart = sm.MeshParts[part.MeshPartIndex];
                                for (int i = 0; i < meshPart.NumIndices; i++)
                                    lodIndices.Add((int)sm.CpuIndices[meshPart.BaseIndex + i]);
                            }
                            triangles = lodIndices.ToArray();
                        }
                        else
                        {
                            triangles = Array.ConvertAll(sm.CpuIndices, i => (int)i);
                        }

                        var cooking = PhysicsWorld.Physics.CreateCooking();

                        var desc = new TriangleMeshDesc()
                        {
                            Flags = (MeshFlag)0,
                            Triangles = triangles,
                            Points = sm.Positions
                        };

                        var stream = new MemoryStream();
                        cooking.CookTriangleMesh(desc, stream);

                        stream.Position = 0;
                        smTriMesh = PhysicsWorld.Physics.CreateTriangleMesh(stream);

                        _triMeshCache.Add(smHash, smTriMesh);
                    }

                    return new TriangleMeshGeometry(smTriMesh)
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
                    var terrain = Entity?.GetComponent<GPUTerrain>();
                    if (terrain?.HeightField == null)
                    {
                        Debug.Log("[RigidBody] No GPUTerrain HeightField found");
                        return null;
                    }

                    var heightMap = terrain.HeightField;
                    int rows = heightMap.GetLength(0);
                    int cols = heightMap.GetLength(1);

                    float fx = terrain.TerrainSize.X / (rows - 1);
                    float fy = terrain.TerrainSize.Y / (cols - 1);

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

        private void CreateBody()
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

            var matrix = Transform.WorldMatrix;
            // Strip scale from pose matrix (PhysX uses scale via geometry, not pose)
            matrix.M11 = 1; matrix.M22 = 1; matrix.M33 = 1;

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
