using System;
using System.Numerics;
using DotRecast.Core.Numerics;
using DotRecast.Detour.Crowd;
using Freefall.Base;
using Freefall.Reflection;

using Component = Freefall.Base.Component;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// Steers an entity along the navmesh toward a destination.
    /// Uses DtCrowd for local avoidance between agents.
    /// </summary>
    [Icon("icon_nav_agent.png")]
    public class NavMeshAgent : Component, IUpdate
    {
        [Category("Agent")]
        [ValueRange(0.1f, 2f)]
        public float Radius = 0.35f;

        [ValueRange(0.5f, 5f)]
        public float Height = 2.0f;

        /// <summary>
        /// Off, the crowd only knows it is there, where its transform stands and as fast as that moves (the player):
        /// the others plan their ways round it as round each other, and it is never given a way or moved. Agents not in
        /// the crowd are walked into: the player stood in the way of NPCs that did not know of him (2026-10-25).
        /// </summary>
        public bool Steered = true;

        [Category("Movement")]
        public bool DriveTransform = false;

        [ValueRange(0.5f, 20f)]
        public float Speed = 3.5f;

        /// <summary>Turn speed in degrees per second.</summary>
        [ValueRange(30f, 720f)]
        public float AngularSpeed = 360f;

        [ValueRange(1f, 50f)]
        public float Acceleration = 8f;

        [ValueRange(0.1f, 5f)]
        public float StoppingDistance = 0.5f;

        [Category("State")]
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

        /// <summary>Metres left to where its way ends: the destination, or as near to it as a way leads.</summary>
        [DontSerialize]
        public float RemainingDistance => MathF.Sqrt(_remainingSquared);

        /// <summary>
        /// Seconds it has had a way to go while its transform has not got <see cref="StallWithin"/> metres from where
        /// it was: a body walking into a shut door, or one standing under a platform its way climbs to by no stair it
        /// can take. The crowd can say it is moving all that while (a few centimetres a second). 0 without a way.
        /// </summary>
        [DontSerialize]
        public float Stalled { get; private set; }

        /// <summary>How often the crowd's agent was put again where its transform stands, or back on the navmesh (see Update).</summary>
        [DontSerialize]
        public int Reseated { get; private set; }

        /// <summary>The crowd has its agent on the navmesh and moves it (Detour's state "walking").</summary>
        [DontSerialize]
        public bool OnNavMesh => _crowdAgent != null && _crowdAgent.state == DtCrowdAgentState.DT_CROWDAGENT_STATE_WALKING;

        /// <summary>Metres to the nearest other agent of the crowd round it (within about 4 m), or infinity.</summary>
        [DontSerialize]
        public float NearestOther
        {
            get
            {
                if (_crowdAgent == null) return float.PositiveInfinity;
                float least = float.MaxValue;
                var me = _crowdAgent.npos;
                for (int i = 0; i < _crowdAgent.nneis; i++)
                {
                    var other = _crowdAgent.neis[i].agent;
                    if (other == null) continue;
                    float dx = other.npos.X - me.X, dz = other.npos.Z - me.Z;
                    least = MathF.Min(least, dx * dx + dz * dz);
                }
                return least == float.MaxValue ? float.PositiveInfinity : MathF.Sqrt(least);
            }
        }

        /// <summary>
        /// Somebody of the crowd within <paramref name="within"/> metres whose way runs into this one: it moves at a walk
        /// or more, comes nearer, and its line passes within the two bodies' breadth and a little. The nearest such, where
        /// it is and how it moves (level), for making way. False when nobody is.
        /// </summary>
        public bool Approaching(float within, out Vector3 from, out Vector3 velocity)
        {
            from = velocity = default;
            if (_crowdAgent == null) return false;
            var me = _crowdAgent.npos;
            float best = within * within;
            bool found = false;
            for (int i = 0; i < _crowdAgent.nneis; i++)
            {
                var other = _crowdAgent.neis[i].agent;
                if (other == null) continue;
                float dx = me.X - other.npos.X, dz = me.Z - other.npos.Z;
                float away = dx * dx + dz * dz;
                if (away > best) continue;

                float vx = other.vel.X, vz = other.vel.Z;
                float speed = vx * vx + vz * vz;
                if (speed < ApproachSpeed * ApproachSpeed) continue;
                float toward = dx * vx + dz * vz;              // > 0: coming nearer
                if (toward <= 0f) continue;
                float aside = dx * vz - dz * vx;               // |aside| / |v|: how far off its line this one stands
                float breadth = Radius + other.option.radius + ApproachBreadth;
                if (aside * aside > breadth * breadth * speed) continue;

                best = away;
                found = true;
                from = new Vector3(other.npos.X, other.npos.Y, other.npos.Z);
                velocity = new Vector3(vx, 0f, vz);
            }
            return found;
        }

        // Approaching: metres a second at which somebody is on its way, and metres beside its line that still count as in it
        private const float ApproachSpeed = 0.4f, ApproachBreadth = 0.3f;

        // ── Events ──

        public event Action? OnDestinationReached;
        public event Action? OnPathFailed;

        // ── Internal ──

        private DtCrowdAgent? _crowdAgent;
        private bool _destinationPending;
        private Vector3 _pendingTarget;
        private Vector3 _goal;            // Destination put on the navmesh: what the crowd steers for
        private float _remainingSquared;
        private Vector3 _headwayFrom;     // where its transform was when it last made headway (Stalled)

        // Metres its transform has to go to count as headway (Stalled)
        private const float StallWithin = 0.5f;

        // Metres apart, up or down and level, at which an external controller's body and the crowd's agent have
        // parted (see Update): more than a stair's detail mesh lies off its steps, or the capsule off the navmesh
        private const float PartedAbove = 1.5f, PartedBeyond = 1.5f;

        // Seconds between two re-seats at most (Reseat), and when it was last re-seated (Time.TotalTime); the same for
        // putting an agent the crowd gave up on back on the navmesh
        private const double ReseatEvery = 2.0, PlaceEvery = 0.5;
        private double _reseatAt = double.NegativeInfinity, _placedAt = double.NegativeInfinity;

        // Not steered: where its transform was last frame, for its velocity; and the most that counts (a teleport)
        private Vector3 _lastAt;
        private bool _hadLast;
        private const float FastestKnown = 10f;

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
                // Not steered, it plans nothing: the others plan round it
                updateFlags = !Steered ? 0 : DtCrowdAgentUpdateFlags.DT_CROWD_ANTICIPATE_TURNS
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
            LeaveCrowd();
        }

        /// <summary>
        /// Disabled (a body put away in a PrefabPool), it leaves the crowd, which would steer it on and steer others
        /// round it; enabled again it joins at the place it is then, in its next update.
        /// </summary>
        protected override void OnEnabledChanged()
        {
            if (!Enabled)
            {
                LeaveCrowd();
                HasPath = false;
                IsMoving = false;
                Velocity = Vector3.Zero;
                Stalled = 0f;
                _destinationPending = false;
            }
        }

        private void LeaveCrowd()
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

            if (_crowdAgent != null && NavMeshWorld.IsReady)
            {
                HasPath = NavMeshWorld.SetAgentTarget(_crowdAgent, target, out _goal);
                if (!HasPath)
                {
                    // Nowhere to go: not on to where it was going before, either
                    NavMeshWorld.ResetAgentTarget(_crowdAgent);
                    OnPathFailed?.Invoke();
                }
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
            Stalled = 0f;

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

            if (!Steered)
            {
                Known();
                return;
            }

            // Speed set after it joined reaches the crowd too (a run, a walk): it was read once, at the join
            if (_crowdAgent.option.maxSpeed != Speed) _crowdAgent.option.maxSpeed = Speed;

            // Handle pending destination (navmesh loaded after Awake)
            if (_destinationPending && NavMeshWorld.IsReady)
            {
                _destinationPending = false;
                SetDestination(_pendingTarget);
            }

            // The crowd gives an agent up for good once it stands where the crowd finds no polygon within a little over
            // its radius (a body added before it was put down, a moment off the navmesh): it is never moved again, and
            // its body stood with a way to go, on the wall walk, on the fountain's steps (2026-10-25). It is put on the
            // nearest polygon within a larger reach, every PlaceEvery seconds until that works.
            if (_crowdAgent.state == DtCrowdAgentState.DT_CROWDAGENT_STATE_INVALID && Time.TotalTime - _placedAt >= PlaceEvery)
            {
                _placedAt = Time.TotalTime;
                if (NavMeshWorld.PlaceAgent(_crowdAgent, Transform.Position))
                {
                    Reseated++;
                    if (HasPath) SetDestination(Destination);
                }
            }

            // Read crowd-computed position and velocity
            var crowdPos = NavMeshWorld.FromRc(_crowdAgent.npos);
            var crowdVel = NavMeshWorld.FromRc(_crowdAgent.vel);

            // Without a path it is going nowhere, whatever speed the crowd still has it at
            Position = crowdPos;
            Velocity = HasPath ? crowdVel : Vector3.Zero;
            IsMoving = HasPath && crowdVel.LengthSquared() > 0.0001f;

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
                var body = Transform.Position;

                // The crowd follows the body by sliding its agent along the navmesh's surface, level only: a body
                // put somewhere else (a teleport), or one on a storey over the crowd's (an NPC on a gatehouse's top
                // whose agent was on the street through the gate below it, 2026-10-25), is never caught up with, and
                // the crowd steers it from where it is not. Parted that far, the agent is put again where the body
                // stands and asks for its way anew; not again within ReseatEvery, since a body further off the
                // navmesh than the crowd reaches stays apart from where it puts the agent (1431 times in 4 minutes).
                var apart = crowdPos - body;
                if ((MathF.Abs(apart.Y) > PartedAbove || apart.X * apart.X + apart.Z * apart.Z > PartedBeyond * PartedBeyond)
                    && Time.TotalTime - _reseatAt >= ReseatEvery)
                {
                    Reseat();
                    return;
                }
                _crowdAgent.npos = NavMeshWorld.ToRc(body);
            }

            // Arrived: within StoppingDistance of where its way ends. That is the destination put
            // on the navmesh (not the point asked for, which may be off it or at another height),
            // or, once the crowd has found that no way leads there, the nearest point one does
            // lead to. The crowd is told to stop: left with its target it keeps making for the
            // exact point, and a character that follows the agent's velocity but moves by its
            // clips walks past it, is sent back, and turns to and fro on the spot.
            if (HasPath)
            {
                var end = _crowdAgent.targetState == DtMoveRequestState.DT_CROWDAGENT_TARGET_VALID
                    ? NavMeshWorld.FromRc(_crowdAgent.corridor.GetTarget())
                    : _goal;

                _remainingSquared = Vector3.DistanceSquared(crowdPos, end);
                if (_remainingSquared <= StoppingDistance * StoppingDistance)
                {
                    Stop();
                    OnDestinationReached?.Invoke();
                }
            }

            // Headway: how long it has had a way to go and its transform has stayed where it was (Stalled). Asked for
            // a new destination it keeps counting: a brain that asks again every half second is no nearer for it.
            var at = Transform.WorldPosition;
            if (!HasPath || Vector3.DistanceSquared(at, _headwayFrom) > StallWithin * StallWithin)
            {
                _headwayFrom = at;
                Stalled = 0f;
            }
            else Stalled += Time.SmoothDelta;
        }

        /// <summary>
        /// Not steered: the crowd is told where its transform stands and how fast it goes, for the others to plan round.
        /// The crowd's own update would move it on by its velocity and push it out of the others' way: both are undone
        /// here every frame.
        /// </summary>
        private void Known()
        {
            var at = Transform.Position;
            var velocity = Vector3.Zero;
            float delta = Time.SmoothDelta;
            if (_hadLast && delta > 0f)
            {
                velocity = (at - _lastAt) / delta;
                velocity.Y = 0f;
                if (velocity.LengthSquared() > FastestKnown * FastestKnown) velocity = Vector3.Zero;   // a teleport
            }
            _lastAt = at;
            _hadLast = true;

            var v = NavMeshWorld.ToRc(velocity);
            _crowdAgent!.npos = NavMeshWorld.ToRc(at);
            _crowdAgent.vel = v;
            _crowdAgent.dvel = v;
            _crowdAgent.nvel = v;
            Position = at;
            Velocity = velocity;
            IsMoving = velocity.LengthSquared() > 0.01f;
        }

        /// <summary>The crowd's agent again, where the transform stands now, on its way to where it was going.</summary>
        private void Reseat()
        {
            bool going = HasPath;
            var destination = Destination;
            LeaveCrowd();
            Reseated++;
            _reseatAt = Time.TotalTime;
            EnsureRegistered();
            if (going && _crowdAgent != null) SetDestination(destination);
        }
    }
}
