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
    [Icon("icon_rigidbody.png")]
    public class RigidBody : Component
    {
        private bool _isStatic = true;
        private RigidStatic? _staticActor;
        private RigidDynamic? _dynamicActor;
        private Action<Message>? _onTerrainLoaded;

        public Actor? Actor => (Actor?)_staticActor ?? _dynamicActor;

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
                    CreateActor();
            }
        }

        protected override void Awake()
        {
            if (Engine.IsEditor)
                return;

            try
            {
                CreateActor();

                //// If terrain body wasn't created (data not loaded yet), retry when terrain is ready
                //if (Actor == null && Type == ShapeType.Terrain)
                //{
                //    MessageDispatcher.AddListener("TerrainLoaded", _onTerrainLoaded = _ =>
                //    {
                //        CreateBody();
                //        MessageDispatcher.RemoveListener("TerrainLoaded", _onTerrainLoaded);
                //        _onTerrainLoaded = null;
                //    });
                //}
                Transform.OnChanged += OnTransformChanged;
            }   
            catch (Exception ex)
            {
                Debug.Log($"[RigidBody] {Entity?.Name}: {ex.Message}");
            }
        }

        private void OnTransformChanged()
        {
            if (IsStatic && Actor != null)
            {
                Transform.Matrix.Deconstruct(out var scale, out var rotation, out var translation);

                var matrix = Matrix4x4.CreateFromQuaternion(rotation)
                           * Matrix4x4.CreateTranslation(translation);

                _staticActor?.GlobalPose = matrix;
                _staticActor?.GlobalPosePosition = translation;
                _staticActor?.GlobalPoseQuat = rotation;
            }
        }

        public void CreateActor()
        {
            var colliders = Entity.GetComponentsInChildren<Collider>();
            if(colliders.Count == 0)
                return;

            // Remove old actor
            if (Actor != null)
            {
                PhysicsWorld.Scene.RemoveActor(Actor);
                Actor.Dispose();
                _staticActor = null;
                _dynamicActor = null;
            }

            Transform.Matrix.Deconstruct(out var scale, out var rotation, out var translation);

            // Build a clean pose from position + rotation (no scale).
            // PhysX requires an orthonormal rotation — scale is applied via geometry.
            var matrix = Matrix4x4.CreateFromQuaternion(rotation)
                       * Matrix4x4.CreateTranslation(translation);

            RigidActor actor = null;

            if (IsStatic)
            {
                _staticActor = PhysicsWorld.Physics.CreateRigidStatic();
                _staticActor.UserData = Entity;
                _staticActor.GlobalPose = matrix;
                _staticActor.GlobalPosePosition = translation;
                _staticActor.GlobalPoseQuat = rotation;
                _staticActor.Flags &= ~ActorFlag.Visualization;
                _staticActor.Flags |= ActorFlag.DisableGravity;
                actor = _staticActor;
            }
            else
            {
                _dynamicActor = PhysicsWorld.Physics.CreateRigidDynamic();
                _dynamicActor.UserData = Entity;
                _dynamicActor.GlobalPose = matrix;
                _dynamicActor.AngularDamping = 1f;
                _dynamicActor.LinearDamping = 0.1f;
                _dynamicActor.SetMassAndUpdateInertia(Mass);
                actor = _dynamicActor;
            }

            foreach (var collider in colliders)
                collider.CreateShape(Entity, actor);

            PhysicsWorld.Scene.AddActor(Actor!);
        }

        public override void Destroy()
        {
            Transform.OnChanged -= OnTransformChanged;

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
