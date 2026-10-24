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

        private bool _isKinematic;

        /// <summary>
        /// A body that is moved, not simulated: a door's leaf, a lift. Its actor is a kinematic RigidDynamic that
        /// goes where its transform is put (a kinematic target per change, so what it pushes is pushed at its
        /// speed); nothing moves it back. It may have mesh colliders, which a simulated body may not. A body that
        /// <see cref="IsStatic"/> stays static whatever this says. Left out of the navmesh bake like any body that
        /// is not static.
        /// </summary>
        public bool IsKinematic
        {
            get => _isKinematic;
            set
            {
                if (_isKinematic == value) return;
                _isKinematic = value;

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
            else if (IsKinematic && _dynamicActor != null)
            {
                Transform.Matrix.Deconstruct(out _, out var rotation, out var translation);
                _dynamicActor.SetKinematicTarget(Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(translation));
            }
        }

        // The colliders whose shapes are on the actor: their wrappers go before the actor does (its release takes the shapes)
        private List<Collider>? _colliders;

        /// <summary>
        /// The colliders that are this body's: its own and those of the entities below it, except below another body.
        /// A collider belongs to its nearest body: a door's leaf with a kinematic body of its own under a house's
        /// static one moves with its own actor, not as a shape of the house's.
        /// </summary>
        private List<Collider> OwnColliders()
        {
            var colliders = Entity.GetComponents<Collider>();
            foreach (Transform child in Transform)
                Gather(child.Entity, colliders);
            return colliders;
        }

        private static void Gather(Entity entity, List<Collider> into)
        {
            if (entity == null || entity.GetComponent<RigidBody>() != null) return;
            foreach (var component in entity.Components)
                if (component is Collider collider) into.Add(collider);
            foreach (Transform child in entity.Transform)
                Gather(child.Entity, into);
        }

        public void CreateActor()
        {
            var colliders = OwnColliders();
            if(colliders.Count == 0)
                return;

            // Remove old actor
            if (Actor != null)
                ReleaseActor();

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
                // Kinematic before the shapes: PhysX refuses a mesh shape on a simulated actor
                if (IsKinematic) _dynamicActor.RigidBodyFlags |= RigidBodyFlag.Kinematic;
                _dynamicActor.GlobalPose = matrix;
                _dynamicActor.AngularDamping = 1f;
                _dynamicActor.LinearDamping = 0.1f;
                actor = _dynamicActor;
            }

            foreach (var collider in colliders)
                collider.CreateShape(Entity, actor);
            _colliders = colliders;

            // A simulated body's mass from its shapes once it has them (a kinematic one has no use for it)
            if (_dynamicActor != null && !IsKinematic)
                _dynamicActor.SetMassAndUpdateInertia(Mass);

            PhysicsWorld.Scene.AddActor(Actor!);
            PhysicsWorld.Simulated(this, _dynamicActor != null && !IsKinematic);
        }

        public override void Destroy()
        {
            Transform.OnChanged -= OnTransformChanged;

            if (Actor != null)
                ReleaseActor();
        }

        // Out of the scene, the shapes' wrappers first (left on the actor), then the actor with its shapes in one release
        private void ReleaseActor()
        {
            PhysicsWorld.Simulated(this, false);
            PhysicsWorld.Scene.RemoveActor(Actor!);
            if (_colliders != null)
                foreach (var collider in _colliders)
                    if (collider.ShapeHost == Entity)
                        collider.ReleaseShape(native: false);
            _colliders = null;

            PhysXRelease.Release((PhysX.IDisposable?)_staticActor ?? _dynamicActor);
            _staticActor = null;
            _dynamicActor = null;
        }
    }
}
