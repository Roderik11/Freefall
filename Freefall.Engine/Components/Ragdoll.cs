using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;
using PhysX;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// A humanoid character that falls as a body: <see cref="Fall"/> makes a physics body of each part of the
    /// Animator's <see cref="Animator.Body"/> (capsules, joined by joints that bend as knees, elbows, shoulders, hips,
    /// the neck and the spine bend), starting from the pose drawn this frame, and from then on the parts' bones take
    /// the bodies' poses while the bones between follow as the clip has them. When the bodies come to rest (or after
    /// <see cref="SettleAfter"/>) the pose is kept and the bodies go: a corpse costs nothing. <see cref="Rise"/> blends
    /// back to the clips. Goes on the entity with the Animator. Play only: in the editor there is no ground to fall on.
    ///
    /// The parts meet the world (terrain, buildings, rocks) and other bodies, but not each other and not kinematic
    /// actors: the characters' capsules walk through a corpse rather than kick it about (<see cref="PhysXFilterShader"/>).
    /// No ray finds them: hitboxes are the Animator's, posed as the ragdoll lies.
    /// </summary>
    [Icon("icon_rigidbody.png")]
    public class Ragdoll : Component
    {
        [Category("Body")]
        [Description("What the whole body weighs, in kilograms: shared out between the parts by their size.")]
        public float Mass = 75f;

        [Description("How quickly a part's turning dies down by itself: more, and a body flops less.")]
        public float AngularDamping = 0.6f;

        [Description("Seconds a body falls at most: then the pose is kept as it is, whether or not it has come to rest.")]
        public float SettleAfter = 6f;

        [Description("Seconds Rise takes to blend from the body as it lies to the clips.")]
        public float RiseSeconds = 0.4f;

        [Category("Joints")]
        [Description("Degrees the spine bends at each of its joints (pelvis to belly, belly to chest), any way, and twists.")]
        public float SpineBend = 20f;

        public float SpineTwist = 15f;

        [Description("Degrees the head bends on the neck, any way, and turns.")]
        public float NeckBend = 40f;

        public float NeckTwist = 45f;

        [Description("Degrees an upper arm swings away from where it hangs in the rig's own pose, any way, and twists.")]
        public float ShoulderSwing = 85f;

        public float ShoulderTwist = 60f;

        [Description("Degrees an elbow bends (forwards only).")]
        public float ElbowBend = 140f;

        [Description("Degrees a thigh swings, any way, round a line tilted forwards from where it hangs in the rig's own pose by Hip Forward, and twists.")]
        public float HipSwing = 60f;

        public float HipForward = 25f;

        public float HipTwist = 30f;

        [Description("Degrees a knee bends (backwards only).")]
        public float KneeBend = 140f;

        [Description("Degrees a foot bends at the ankle, any way, and twists.")]
        public float AnkleBend = 30f;

        public float AnkleTwist = 15f;

        private enum State { Off, Falling, Lying, Rising }
        private State _state;

        /// <summary>It lies as a body: falling, or come to rest and kept.</summary>
        [Freefall.Reflection.DontSerialize]
        public bool Down => _state == State.Falling || _state == State.Lying;

        /// <summary>Its bodies are being simulated.</summary>
        [Freefall.Reflection.DontSerialize]
        public bool Falling => _state == State.Falling;

        /// <summary>Its bones are the ragdoll's, wholly or in part (rising).</summary>
        internal bool Posing => _state != State.Off;

        private Animator _animator;
        private BodyShape _body;
        private RigidDynamic[] _actors;
        private Shape[] _shapes;
        private D6Joint[] _joints;
        private float[] _radii;

        // The parts' world poses after the step before the last and after the last, and their model matrices drawn now
        private Vector3[] _wasAt, _nowAt;
        private Quaternion[] _wasTurn, _nowTurn;
        private Matrix4x4[] _models;
        private long _step;
        private float _fallen, _rise;

        // Per skeleton among the character's renderers: which part each bone takes its pose from, and room for the
        // bones' poses against their parents
        private sealed class BoneMap
        {
            public int[] Part;
            public Matrix4x4[] Local;
        }
        private readonly Dictionary<Skeleton, BoneMap> _maps = new();

        private static PhysX.Material s_material;
        private static uint s_ids;

        // A part never starts faster than this (m/s, rad/s): two poses either side of a teleport are no motion
        private const float MostSpeed = 12f, MostTurn = 20f;

        private static Vector3 Capped(Vector3 value, float most)
        {
            float length = value.Length();
            return length > most ? value * (most / length) : value;
        }

        protected override void Awake()
        {
            _animator = Entity.GetComponent<Animator>();
            if (_animator != null) _animator.Ragdoll = this;
        }

        public override void Destroy()
        {
            Release();
            if (_animator != null && _animator.Ragdoll == this) _animator.Ragdoll = null;
            _state = State.Off;
        }

        /// <summary>
        /// Fall as a body from the pose drawn this frame, every part moving at <paramref name="velocity"/> (the
        /// character's own, m/s) and an <paramref name="impulse"/> (N·s, world space) at <paramref name="at"/> on the
        /// part nearest it: the blow that killed. Falling already, only the impulse. False where it cannot (in the
        /// editor, no Animator, a rig that is not a humanoid's, a character not posed yet): the caller keeps its clip.
        /// </summary>
        public bool Fall(Vector3 impulse, Vector3 at, Vector3 velocity)
        {
            if (_state == State.Falling)
            {
                Push(impulse, at);
                return true;
            }

            if (Engine.IsEditor || PhysicsWorld.Scene == null) return false;
            _animator ??= Entity.GetComponent<Animator>();
            if (_animator == null || !_animator.TryGetBodyFrames(out var body, out var frames)) return false;
            _animator.Ragdoll = this;

            Release();
            Build(body, frames, velocity);
            Push(impulse, at);
            _state = State.Falling;
            _fallen = 0f;
            _step = PhysicsWorld.Steps;
            Sample();
            return true;
        }

        /// <summary>Back to the clips, blended over <see cref="RiseSeconds"/> from the pose it lies in: respawned, raised.</summary>
        public void Rise()
        {
            if (_state == State.Off || _state == State.Rising) return;
            Release();
            _rise = 0f;
            _state = State.Rising;
        }

        /// <summary>Back to the clips at once (a body taken up again by a pool).</summary>
        public void Stop()
        {
            Release();
            _state = State.Off;
        }

        // ---- Making the bodies ----

        private void Build(BodyShape body, Matrix4x4[] frames, Vector3 velocity)
        {
            var parts = body.Parts;
            int count = parts.Length;
            _body = body;
            _actors = new RigidDynamic[count];
            _shapes = new Shape[count];
            _joints = new D6Joint[count];
            _radii = new float[count];
            _wasAt = new Vector3[count];
            _nowAt = new Vector3[count];
            _wasTurn = new Quaternion[count];
            _nowTurn = new Quaternion[count];
            if (_models == null || _models.Length != count) _models = new Matrix4x4[count];

            var physics = PhysicsWorld.Physics;
            s_material ??= physics.CreateMaterial(0.8f, 0.7f, 0f);
            uint id = ++s_ids;

            float volume = 0f;
            foreach (var part in parts) volume += part.Volume;
            volume = MathF.Max(volume, 1e-6f);

            // Each part goes on as it was moving in the clip (a runner's trunk forwards, an arm with its swing); without
            // two recent poses, all of it at the character's own velocity
            var linear = new Vector3[count];
            var angular = new Vector3[count];
            bool moving = _animator.TryGetBodyMotion(body, linear, angular);

            for (int i = 0; i < count; i++)
            {
                var part = parts[i];
                Matrix4x4.Decompose(frames[i], out _, out var turn, out var at);
                _wasAt[i] = _nowAt[i] = at;
                _wasTurn[i] = _nowTurn[i] = turn;

                var actor = physics.CreateRigidDynamic();
                actor.GlobalPosePosition = at;
                actor.GlobalPoseQuat = turn;
                actor.UserData = this;

                // A round capsule as thick as the part's oval on average, along its axis
                float radius = MathF.Max(0.02f, 0.5f * (part.RadiusFrom + part.RadiusTo) * MathF.Sqrt(part.Depth));
                var span = part.To - part.From;
                float length = span.Length();
                Geometry geometry = length > 0.01f ? new CapsuleGeometry(radius, length * 0.5f) : new SphereGeometry(radius);
                var shape = RigidActorExt.CreateExclusiveShape(actor, geometry, s_material, ShapeFlag.SimulationShape);
                var along = length > 0.01f ? span / length : Vector3.UnitX;
                shape.LocalPose = Matrix4x4.CreateFromQuaternion(FromTo(Vector3.UnitX, along))
                                * Matrix4x4.CreateTranslation((part.From + part.To) * 0.5f);
                shape.SimulationFilterData = new FilterData(PhysXFilterShader.RagdollPart, id, 0, 0);

                actor.SetMassAndUpdateInertia(Mass * part.Volume / volume);
                actor.LinearDamping = 0.05f;
                actor.AngularDamping = AngularDamping;
                actor.SolverIterationCounts = new SolverIterationCounts(12, 4);
                actor.MaxDepenetrationVelocity = 3f;
                actor.LinearVelocity = moving ? Capped(linear[i], MostSpeed) : velocity;
                if (moving) actor.AngularVelocity = Capped(angular[i], MostTurn);

                PhysicsWorld.Scene.AddActor(actor);
                _actors[i] = actor;
                _shapes[i] = shape;
                _radii[i] = radius;
            }

            for (int i = 0; i < count; i++)
            {
                var part = parts[i];
                if (part.Parent < 0) continue;
                _joints[i] = Join(part, parts[part.Parent], _actors[part.Parent], _actors[i]);
            }
        }

        private enum Kind { Spine, Neck, Shoulder, Elbow, Hip, Knee, Ankle }

        private static Kind KindOf(HumanoidBone bone) => bone switch
        {
            HumanoidBone.Head => Kind.Neck,
            HumanoidBone.LeftUpperArm or HumanoidBone.RightUpperArm => Kind.Shoulder,
            HumanoidBone.LeftLowerArm or HumanoidBone.RightLowerArm => Kind.Elbow,
            HumanoidBone.LeftUpperLeg or HumanoidBone.RightUpperLeg => Kind.Hip,
            HumanoidBone.LeftLowerLeg or HumanoidBone.RightLowerLeg => Kind.Knee,
            HumanoidBone.LeftFoot or HumanoidBone.RightFoot => Kind.Ankle,
            _ => Kind.Spine,
        };

        /// <summary>
        /// The joint between a part and the one it hangs from, where its bone starts. A hinge (elbow, knee) turns about
        /// the axis across the way it bends; anything else bends any way within a cone round the part's own axis and
        /// twists about it. All of it from the rig's rest pose (BodyPart.Bind): a bend is measured from how the parts lie
        /// there, not from the pose the character died in. From that pose a knee bent mid-stride could bend as far
        /// again backwards, and every cone stood round whatever the limb was doing.
        /// </summary>
        private D6Joint Join(BodyPart part, BodyPart parent, RigidDynamic parentActor, RigidDynamic actor)
        {
            var kind = KindOf(part.Bone);
            var bind = part.Bind;
            var anchor = Vector3.Transform(part.Joint, bind);
            var along = Vector3.TransformNormal(part.To - part.From, bind);
            along = along.LengthSquared() > 1e-10f ? Vector3.Normalize(along) : Vector3.UnitY;

            // The way the character faces, in the model space of the rig (+Z of the entity)
            var forward = _body.Skeleton.FlipXZ ? -Vector3.UnitZ : Vector3.UnitZ;

            // The joint's X: the part's axis, or for a hinge the axis it bends about (a positive turn about it takes the
            // part from the line of its parent the way an elbow bends, forwards, or a knee, backwards)
            var x = along;
            bool hinge = kind == Kind.Elbow || kind == Kind.Knee;
            if (hinge)
            {
                var upper = Vector3.TransformNormal(parent.To - parent.From, parent.Bind);
                var bend = kind == Kind.Elbow ? forward : -forward;
                x = Vector3.Cross(upper, bend);
                if (x.LengthSquared() < 1e-8f) x = Vector3.TransformNormal(part.Across, bind);
                x = Vector3.Normalize(x);
            }
            var joint = JointFrame(x, anchor);

            // A thigh swings further forwards than back: its cone stands tilted forwards on the pelvis
            var onParent = joint;
            if (kind == Kind.Hip)
            {
                var ahead = forward - x * Vector3.Dot(forward, x);
                if (ahead.LengthSquared() > 1e-6f)
                {
                    float tilt = HipForward * MathF.PI / 180f;
                    onParent = JointFrame(Vector3.Normalize(x * MathF.Cos(tilt) + Vector3.Normalize(ahead) * MathF.Sin(tilt)), anchor);
                }
            }

            Matrix4x4.Invert(parent.Bind, out var toParent);
            Matrix4x4.Invert(bind, out var toPart);
            var d6 = PhysicsWorld.Physics.CreateJoint<D6Joint>(parentActor, onParent * toParent, actor, joint * toPart);

            const float Degree = MathF.PI / 180f;
            float twist, swing;
            switch (kind)
            {
                case Kind.Elbow:
                case Kind.Knee:
                    float most = (kind == Kind.Elbow ? ElbowBend : KneeBend) * Degree;
                    d6.SetMotion(D6Axis.Twist, D6Motion.Limited);
                    d6.TwistLimit = new JointAngularLimitPair(-5f * Degree, Math.Clamp(most, 0.1f, 3f));
                    return d6;
                case Kind.Neck: swing = NeckBend; twist = NeckTwist; break;
                case Kind.Shoulder: swing = ShoulderSwing; twist = ShoulderTwist; break;
                case Kind.Hip: swing = HipSwing; twist = HipTwist; break;
                case Kind.Ankle: swing = AnkleBend; twist = AnkleTwist; break;
                default: swing = SpineBend; twist = SpineTwist; break;
            }

            swing = Math.Clamp(swing * Degree, 0.02f, 3f);
            twist = Math.Clamp(twist * Degree, 0.02f, 3f);
            d6.SetMotion(D6Axis.Twist, D6Motion.Limited);
            d6.SetMotion(D6Axis.Swing1, D6Motion.Limited);
            d6.SetMotion(D6Axis.Swing2, D6Motion.Limited);
            d6.TwistLimit = new JointAngularLimitPair(-twist, twist);
            d6.SwingLimit = new JointLimitCone(swing, swing);
            return d6;
        }

        // A joint's frame: X along the given unit axis, at the anchor
        private static Matrix4x4 JointFrame(Vector3 x, Vector3 anchor)
        {
            var y = Vector3.Normalize(Vector3.Cross(MathF.Abs(x.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitZ, x));
            var z = Vector3.Cross(x, y);
            return new Matrix4x4(
                x.X, x.Y, x.Z, 0f,
                y.X, y.Y, y.Z, 0f,
                z.X, z.Y, z.Z, 0f,
                anchor.X, anchor.Y, anchor.Z, 1f);
        }

        private void Push(Vector3 impulse, Vector3 at)
        {
            if (_actors == null || impulse.LengthSquared() < 1e-8f) return;

            // On the part whose capsule is nearest the place
            int nearest = 0;
            float best = float.MaxValue;
            for (int i = 0; i < _actors.Length; i++)
            {
                var part = _body.Parts[i];
                var pose = Matrix4x4.CreateFromQuaternion(_nowTurn[i]) * Matrix4x4.CreateTranslation(_nowAt[i]);
                var from = Vector3.Transform(part.From, pose);
                var to = Vector3.Transform(part.To, pose);
                var span = to - from;
                float length = span.LengthSquared();
                float t = length > 1e-10f ? Math.Clamp(Vector3.Dot(at - from, span) / length, 0f, 1f) : 0f;
                float away = Vector3.DistanceSquared(at, from + span * t);
                if (away < best) { best = away; nearest = i; }
            }
            _actors[nearest].AddForceAtPosition(impulse, at, ForceMode.Impulse, true);
        }

        // ---- Every frame (AnimatorSystem, main thread, before the poses) ----

        /// <summary>
        /// The parts' poses for this frame: read after each step, drawn between the last two, the entity put under
        /// the pelvis. Comes to rest when every body sleeps or the time is up. Rising: on towards the clips.
        /// </summary>
        internal void Sample()
        {
            if (_state == State.Rising)
            {
                _rise += Time.SmoothDelta / MathF.Max(RiseSeconds, 0.01f);
                if (_rise >= 1f) _state = State.Off;
                return;
            }
            if (_state != State.Falling || _actors == null) return;

            if (PhysicsWorld.Steps != _step)
            {
                _fallen += (PhysicsWorld.Steps - _step) * PhysicsWorld.StepSeconds;
                _step = PhysicsWorld.Steps;
                bool asleep = true;
                for (int i = 0; i < _actors.Length; i++)
                {
                    _wasAt[i] = _nowAt[i];
                    _wasTurn[i] = _nowTurn[i];
                    _nowAt[i] = _actors[i].GlobalPosePosition;
                    _nowTurn[i] = _actors[i].GlobalPoseQuat;
                    asleep &= _actors[i].IsSleeping;
                }

                if (asleep || _fallen >= SettleAfter)
                {
                    Lie();
                    return;
                }
            }

            Place(PhysicsWorld.StepFraction);
        }

        // At rest: the last pose is kept, the bodies go
        private void Lie()
        {
            Array.Copy(_nowAt, _wasAt, _nowAt.Length);
            Array.Copy(_nowTurn, _wasTurn, _nowTurn.Length);
            Place(1f);
            Release();
            _state = State.Lying;
        }

        /// <summary>The parts <paramref name="share"/> of the way from their poses a step ago to their last, as model matrices.</summary>
        private void Place(float share)
        {
            int count = _models.Length;
            Span<Vector3> at = count <= 32 ? stackalloc Vector3[count] : new Vector3[count];
            Span<Quaternion> turn = count <= 32 ? stackalloc Quaternion[count] : new Quaternion[count];
            float lowest = float.MaxValue;
            for (int i = 0; i < count; i++)
            {
                at[i] = Vector3.Lerp(_wasAt[i], _nowAt[i], share);
                turn[i] = Quaternion.Slerp(_wasTurn[i], _nowTurn[i], share);

                // The lowest the body reaches: its capsules' ends less their radius
                var part = _body.Parts[i];
                var pose = Matrix4x4.CreateFromQuaternion(turn[i]) * Matrix4x4.CreateTranslation(at[i]);
                float low = MathF.Min(Vector3.Transform(part.From, pose).Y, Vector3.Transform(part.To, pose).Y) - _radii[i];
                lowest = MathF.Min(lowest, low);
            }

            // The entity under the pelvis, on the ground the body lies on: whatever goes by the character's place
            // (looting, culling, the sounds) goes by where the body is
            Transform.Position = new Vector3(at[0].X, lowest, at[0].Z);

            Matrix4x4.Invert(Rigid(Transform.Matrix), out var toEntity);
            if (_body.Skeleton.FlipXZ) toEntity *= Matrix4x4.CreateScale(-1f, 1f, -1f);
            for (int i = 0; i < count; i++)
                _models[i] = Matrix4x4.CreateFromQuaternion(turn[i]) * Matrix4x4.CreateTranslation(at[i]) * toEntity;

            MapSkeletons();
        }

        // Which part each bone of each skeleton among the character's renderers takes its pose from (main thread: the
        // poses on workers only read these)
        private void MapSkeletons()
        {
            var parts = _animator?.Parts;
            if (parts == null) return;
            for (int r = 0; r < parts.Count; r++)
            {
                var skeleton = parts[r].Mesh?.Skeleton;
                if (skeleton == null || _maps.ContainsKey(skeleton)) continue;

                var map = new BoneMap { Part = new int[skeleton.Bones.Length], Local = new Matrix4x4[skeleton.Bones.Length] };
                Array.Fill(map.Part, -1);
                var own = _body.Skeleton;
                for (int p = 0; p < _body.Parts.Length; p++)
                {
                    int bone = ReferenceEquals(skeleton, own) ? _body.Parts[p].BoneIndex : skeleton.FindBone(own.BoneNames[_body.Parts[p].BoneIndex]);
                    if (bone >= 0) map.Part[bone] = p;
                }
                _maps[skeleton] = map;
            }
        }

        // ---- In the pose (Animator.GetPose, on a worker) ----

        /// <summary>
        /// The parts' bones of one skeleton take the ragdoll's poses (keeping their own scale), every other bone
        /// follows its parent as the clip has it. Rising: from the ragdoll's poses towards the clip's.
        /// </summary>
        internal void Apply(Skeleton skeleton, Matrix4x4[] bones)
        {
            if (!_maps.TryGetValue(skeleton, out var map) || _models == null) return;
            var hierarchy = skeleton.Bones;
            int count = Math.Min(hierarchy.Length, bones.Length);
            float clip = _state == State.Rising ? _rise * _rise * (3f - 2f * _rise) : 0f;

            var local = map.Local;
            for (int i = 0; i < count; i++)
            {
                int parent = hierarchy[i].Parent;
                if (parent >= 0 && Matrix4x4.Invert(bones[parent], out var inverse)) local[i] = bones[i] * inverse;
                else local[i] = bones[i];
            }

            for (int i = 0; i < count; i++)
            {
                int part = map.Part[i];
                int parent = hierarchy[i].Parent;
                if (part >= 0)
                {
                    Matrix4x4.Decompose(bones[i], out var scale, out var clipTurn, out var clipAt);
                    Matrix4x4.Decompose(_models[part], out _, out var turn, out var at);
                    if (clip > 0f)
                    {
                        turn = Quaternion.Slerp(turn, clipTurn, clip);
                        at = Vector3.Lerp(at, clipAt, clip);
                    }
                    bones[i] = Matrix4x4.CreateScale(scale) * Matrix4x4.CreateFromQuaternion(turn) * Matrix4x4.CreateTranslation(at);
                }
                else if (parent >= 0)
                    bones[i] = local[i] * bones[parent];
            }
        }

        // ---- Small things ----

        private void Release()
        {
            if (_actors == null) return;
            var scene = PhysicsWorld.Scene;
            if (_joints != null)
                foreach (var joint in _joints)
                    if (joint != null) PhysXRelease.Release(joint);
            for (int i = 0; i < _actors.Length; i++)
            {
                if (_actors[i] == null) continue;
                if (_shapes[i] != null) PhysXRelease.Release(_shapes[i], releaseNative: false);
                scene?.RemoveActor(_actors[i]);
                PhysXRelease.Release(_actors[i]);
            }
            _actors = null;
            _shapes = null;
            _joints = null;
        }

        private static Matrix4x4 Rigid(in Matrix4x4 matrix)
        {
            Matrix4x4.Decompose(matrix, out _, out var turn, out var at);
            return Matrix4x4.CreateFromQuaternion(turn) * Matrix4x4.CreateTranslation(at);
        }

        /// <summary>The turn that takes the unit <paramref name="from"/> to the unit <paramref name="to"/>.</summary>
        private static Quaternion FromTo(Vector3 from, Vector3 to)
        {
            float dot = Vector3.Dot(from, to);
            if (dot > 0.99999f) return Quaternion.Identity;
            if (dot < -0.99999f)
            {
                var axis = Vector3.Cross(Vector3.UnitY, from);
                if (axis.LengthSquared() < 1e-6f) axis = Vector3.Cross(Vector3.UnitZ, from);
                return Quaternion.CreateFromAxisAngle(Vector3.Normalize(axis), MathF.PI);
            }
            var cross = Vector3.Cross(from, to);
            return Quaternion.Normalize(new Quaternion(cross, 1f + dot));
        }
    }
}
