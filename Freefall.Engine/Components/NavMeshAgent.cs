using System;
using System.Numerics;
using DotRecast.Core.Numerics;
using DotRecast.Detour.Crowd;
using Freefall.Base;
using Freefall.Reflection;

using Component = Freefall.Base.Component;

namespace Freefall.Components
{
    /// <summary>
    /// Steers an entity along the navmesh toward a destination.
    /// Uses DtCrowd for local avoidance between agents.
    /// </summary>
    [Icon("icon_collider.png")]
    public class NavMeshAgent : Component, IUpdate
    {
        // ── Configuration ──

        public bool DriveTransform = false;

        [ValueRange(0.5f, 20f)]
        public float Speed = 3.5f;

        /// <summary>Turn speed in degrees per second.</summary>
        [ValueRange(30f, 720f)]
        public float AngularSpeed = 360f;

        [ValueRange(0.1f, 2f)]
        public float Radius = 0.35f;

        [ValueRange(0.5f, 5f)]
        public float Height = 2.0f;

        [ValueRange(0.1f, 5f)]
        public float StoppingDistance = 0.5f;

        [ValueRange(1f, 50f)]
        public float Acceleration = 8f;

        // ── State ──

        [DontSerialize] 
        public bool HasPath { get; private set; }

        [DontSerialize] 
        public bool IsMoving { get; private set; }

        [DontSerialize]
        public Vector3 Position { get; private set; }

        [DontSerialize] 
        public Vector3 Velocity { get; private set; }

        [DontSerialize] 
        public Vector3 Destination { get; private set; }

        [DontSerialize]
        public float RemainingDistance { get; private set; }

        // ── Events ──

        public event Action? OnDestinationReached;
        public event Action? OnPathFailed;

        // ── Internal ──

        private DtCrowdAgent? _crowdAgent;
        private bool _destinationPending;
        private Vector3 _pendingTarget;
        private bool _reachedFired;

        // ── Lifecycle ──

        protected override void Awake()
        {
            // Registration happens lazily in Update — NavMeshWorld may not be ready yet
        }

        /// <summary>
        /// Lazily register with the crowd when NavMeshWorld becomes available.
        /// Handles any Awake ordering between NavMeshAgent and NavMeshSurface.
        /// </summary>
        private void EnsureRegistered()
        {
            if (_crowdAgent != null) return;
            if (!NavMeshWorld.IsReady) return;

            var agentParams = new DtCrowdAgentParams
            {
                radius = Radius,
                height = Height,
                maxAcceleration = Acceleration,
                maxSpeed = Speed,
                collisionQueryRange = Radius * 12f,
                pathOptimizationRange = Radius * 30f,
                separationWeight = 2f,
                updateFlags = DtCrowdAgentUpdateFlags.DT_CROWD_ANTICIPATE_TURNS
                            | DtCrowdAgentUpdateFlags.DT_CROWD_OPTIMIZE_VIS
                            | DtCrowdAgentUpdateFlags.DT_CROWD_OPTIMIZE_TOPO
                            | DtCrowdAgentUpdateFlags.DT_CROWD_OBSTACLE_AVOIDANCE,
                obstacleAvoidanceType = 3,
                queryFilterType = 0,
            };

            _crowdAgent = NavMeshWorld.AddCrowdAgent(
                Transform?.WorldPosition ?? Vector3.Zero, agentParams);

            if (_crowdAgent == null)
            {
                Debug.Log($"[NavMeshAgent] {Entity?.Name}: Failed to add crowd agent.");
            }
        }

        public override void Destroy()
        {
            if (_crowdAgent != null)
            {
                NavMeshWorld.RemoveCrowdAgent(_crowdAgent);
                _crowdAgent = null;
            }
        }

        // ── API ──

        /// <summary>Set a new destination. Path is computed by the crowd.</summary>
        public void SetDestination(Vector3 target)
        {
            Destination = target;
            _reachedFired = false;

            if (_crowdAgent != null && NavMeshWorld.IsReady)
            {
                HasPath = NavMeshWorld.SetAgentTarget(_crowdAgent, target);
                if (!HasPath)
                    OnPathFailed?.Invoke();
            }
            else
            {
                _destinationPending = true;
                _pendingTarget = target;
            }
        }

        /// <summary>Stop moving and clear the current path.</summary>
        public void Stop()
        {
            HasPath = false;
            IsMoving = false;
            Velocity = Vector3.Zero;

            if (_crowdAgent != null)
                NavMeshWorld.ResetAgentTarget(_crowdAgent);
        }

        /// <summary>Warp to a position on the navmesh (teleport).</summary>
        public void Warp(Vector3 position)
        {
            Stop();
            
            if(DriveTransform)
                Transform.Position = position;
            // TODO: re-add crowd agent at new position
        }

        // ── Update ──

        public void Update()
        {
            if (Engine.IsEditor) return;

            EnsureRegistered();

            if (_crowdAgent == null) return;

            // Handle pending destination (navmesh loaded after Awake)
            if (_destinationPending && NavMeshWorld.IsReady)
            {
                _destinationPending = false;
                SetDestination(_pendingTarget);
            }

            // Read crowd-computed position and velocity
            var crowdPos = NavMeshWorld.FromRc(_crowdAgent.npos);
            var crowdVel = NavMeshWorld.FromRc(_crowdAgent.vel);

            Position = crowdPos;
            Velocity = crowdVel;
            float speed = crowdVel.Length();
            IsMoving = speed > 0.01f;

            if (DriveTransform)
            {
                // Update transform position
                Transform.Position = crowdPos;

                // Face movement direction
                if (IsMoving)
                {
                    var dir = Vector3.Normalize(new Vector3(crowdVel.X, 0, crowdVel.Z));
                    if (dir.LengthSquared() > 0.001f)
                    {
                        float targetYaw = MathF.Atan2(dir.X, dir.Z);
                        var targetRot = Quaternion.CreateFromYawPitchRoll(targetYaw, 0, 0);
                        float t = MathF.Min(1f, Time.SmoothDelta * AngularSpeed * (MathF.PI / 180f));
                        Transform.Rotation = Quaternion.Slerp(Transform.Rotation, targetRot, t);
                    }
                }
            }
            else
            {
                // External controller (e.g. CapsuleController) moves the entity.
                // Sync actual position back to crowd agent so it steers from
                // the correct location next frame.
                _crowdAgent.npos = NavMeshWorld.ToRc(Transform.Position);
            }

            // Check arrival
            if (HasPath)
            {
                RemainingDistance = Vector3.Distance(crowdPos, Destination);
                if (RemainingDistance <= StoppingDistance && !_reachedFired)
                {
                    _reachedFired = true;
                    HasPath = false;
                    IsMoving = false;
                    OnDestinationReached?.Invoke();
                }
            }
        }
    }
}
