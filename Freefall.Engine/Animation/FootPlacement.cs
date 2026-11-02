using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using Freefall.Base;
using Freefall.Components;
using PhysX;

namespace Freefall.Animation
{
    /// <summary>
    /// The ground under one foot, in the space of the character's entity: its height at a place and
    /// how it rises from there, per metre along the entity's X and Z.
    /// </summary>
    public struct FootGround
    {
        public Vector2 At;
        public float Height;
        public Vector2 Slope;
    }

    /// <summary>
    /// The bones of a humanoid skeleton's legs, for moving them after the pose has gone up the
    /// hierarchy: each part with what hangs on it. Null for a skeleton without two whole legs.
    /// </summary>
    internal sealed class LegChain
    {
        internal sealed class Leg
        {
            public int Upper, Lower, Foot;
            public Footfalls.Sole Sole;

            /// <summary>The bones that go with the thigh, with the shin and with the foot.</summary>
            public int[] Thigh, Shin, Ankle;
        }

        public int Hips;
        public Leg Left, Right;

        private static readonly ConditionalWeakTable<Skeleton, LegChain> s_legs = new();

        public static LegChain Of(Skeleton skeleton)
        {
            var legs = s_legs.GetValue(skeleton, static s => Build(s));
            return legs.Left != null ? legs : null;
        }

        private static LegChain Build(Skeleton skeleton)
        {
            var legs = new LegChain();
            var humanoid = HumanoidTemplate.Detect(skeleton);
            if (humanoid == null || !humanoid.TryGetValue(HumanoidBone.Hips, out legs.Hips)) return legs;

            var bones = skeleton.Bones;
            var bind = new Matrix4x4[bones.Length];
            for (int i = 0; i < bones.Length; i++)
            {
                var pose = bones[i].BindPose;
                var local = Matrix4x4.CreateScale(pose.Scale) * Matrix4x4.CreateFromQuaternion(pose.Rotation) * Matrix4x4.CreateTranslation(pose.Position);
                bind[i] = bones[i].Parent >= 0 ? local * bind[bones[i].Parent] : local;
            }
            float hipHeight = MathF.Max(MathF.Abs(bind[legs.Hips].Translation.Y), 1e-4f);

            var left = Side(bones, humanoid, bind, hipHeight, HumanoidBone.LeftUpperLeg, HumanoidBone.LeftLowerLeg, HumanoidBone.LeftFoot, HumanoidBone.LeftToes);
            var right = Side(bones, humanoid, bind, hipHeight, HumanoidBone.RightUpperLeg, HumanoidBone.RightLowerLeg, HumanoidBone.RightFoot, HumanoidBone.RightToes);
            if (left != null && right != null)
            {
                legs.Left = left;
                legs.Right = right;
            }
            return legs;
        }

        private static Leg Side(Bone[] bones, Dictionary<HumanoidBone, int> humanoid, Matrix4x4[] bind, float hipHeight,
            HumanoidBone upper, HumanoidBone lower, HumanoidBone foot, HumanoidBone toes)
        {
            if (!humanoid.TryGetValue(upper, out int upperBone) || !humanoid.TryGetValue(lower, out int lowerBone)
                || !humanoid.TryGetValue(foot, out int footBone))
                return null;

            if (!humanoid.TryGetValue(toes, out int toesBone)) toesBone = -1;

            // Parents come before their children: each bone goes with the part its parent goes with
            var part = new byte[bones.Length];
            var thigh = new List<int>();
            var shin = new List<int>();
            var ankle = new List<int>();
            for (int i = 0; i < bones.Length; i++)
            {
                int parent = bones[i].Parent;
                part[i] = i == upperBone ? (byte)1 : i == lowerBone ? (byte)2 : i == footBone ? (byte)3
                        : parent >= 0 ? part[parent] : (byte)0;
                if (part[i] == 1) thigh.Add(i);
                else if (part[i] == 2) shin.Add(i);
                else if (part[i] == 3) ankle.Add(i);
            }

            return new Leg
            {
                Upper = upperBone,
                Lower = lowerBone,
                Foot = footBone,
                Sole = new Footfalls.Sole(footBone, toesBone, bind, hipHeight),
                Thigh = thigh.ToArray(),
                Shin = shin.ToArray(),
                Ankle = ankle.ToArray(),
            };
        }
    }

    /// <summary>
    /// Stands a humanoid character's feet on what is under them, whichever way the ground lies: the
    /// pelvis comes down as far as the lower foot needs, each knee bends, and a foot that is down
    /// lies along the ground. Ground is measured from the height the clip stands on, so a foot
    /// keeps the lift the clip gives it, and on level ground at the character's feet the clip plays
    /// exactly as made.
    ///
    /// Which foot is down comes from the clips (<see cref="Footfalls"/>), through the same blend as
    /// the pose (<see cref="StateSamples.Down"/>): only a foot that is down has a say in how far
    /// the pelvis drops, lies along the ground, or keeps the ground it landed on.
    ///
    /// In two halves, because the pose runs on a worker before the scripts have moved the
    /// character. <see cref="Update"/> runs on the main thread after the move, from whoever moves
    /// it: it finds the ground and hands it on, in the entity's space as the entity will stand when
    /// the next pose is drawn. The pose then moves the legs (<see cref="Solve"/>). Set an instance
    /// as <see cref="Animator.Feet"/>; without one the Animator pays for nothing.
    /// </summary>
    public sealed class FootPlacement
    {
        /// <summary>How far above and below the character's feet ground is looked for, in metres. Above: no higher than what it can step on, or a foot climbs a bench it stands beside.</summary>
        public float ReachUp = 0.42f, ReachDown = 0.6f;

        /// <summary>The steepest a foot lies, as the rise per metre (0.5: about 27Â°). Over a step's edge it stops there, with the rest of it over the edge.</summary>
        public float MaxSlope = 0.5f;

        /// <summary>The furthest the pelvis comes down, in metres.</summary>
        public float MaxDrop = 0.3f;

        /// <summary>Seconds for the whole of it to come in or go, when <see cref="Update"/> starts or stops wanting it.</summary>
        public float FadeTime = 0.15f;

        // Seconds to close most of the gap: a foot in the air going up over something and coming down
        // again, one that is down, the stance, and the pelvis going down and coming back. The pelvis
        // is slow on purpose: what it is asked for changes with every step, and followed closely it
        // bobs 10 cm a stride on a 20° climb. At these it bobs 5, with the feet as near the ground.
        private const float LiftEase = 0.04f, LowerEase = 0.09f, PlantedEase = 0.05f, StanceEase = 0.05f, DropEase = 0.15f, RiseEase = 0.45f;

        // How far a foot goes before its ground is looked for again, in metres: one in the air, and
        // one that is down, which keeps the ground it landed on. Not much further than that: a
        // clip slides a foot that is down by a few centimetres now and then, and on stairs that
        // takes it off a step's edge, where it lay tilted, and onto a tread (8 cm had it 5 cm in the tread)
        private const float SwingCast = 0.015f, HeldCast = 0.03f;

        // A correction smaller than this leaves the leg as it is (metres)
        private const float NoRise = 0.0005f;

        // How far short of a straight leg it starts to slow down, as a share of its length
        private const float Soft = 0.03f;

        // In a leg that stands straight the way the foot points says where the knee goes, as much as
        // a knee this far (metres) off the leg's line says so itself: a bent knee goes by the clip
        private const float KneeFollowsFoot = 0.02f;

        private struct Foot
        {
            public bool Looked, Found, Placed;
            public Vector3 LookedAt;    // the ball of the foot when its ground was last looked for
            public Vector3 Anchor;      // on the ground then
            public Vector2 Gradient;    // how the ground rises from there, world X and Z
            public float Height;        // where the foot is put in the world, eased
            public Vector2 Slope;
            public FootGround Ground;   // the same for the pose
        }

        private readonly Animator _animator;
        private Foot _left, _right;
        private float _weight;
        private Vector3 _lastPosition;
        private bool _hasPosition;

        // The ray that is looking
        private Func<RaycastHit[], bool> _onHits;
        private float _hitAt;
        private Vector3 _hitNormal;
        private bool _hitAny;

        public FootPlacement(Animator animator)
        {
            _animator = animator;
        }

        /// <summary>How much each foot is down now, 0 to 1, as the clips have it.</summary>
        public float LeftDown { get; private set; } = 1f;
        public float RightDown { get; private set; } = 1f;

        /// <summary>How far the pelvis is lowered now, in metres: 0 or less.</summary>
        public float Drop { get; private set; }

        /// <summary>How much of it applies now, 0 to 1.</summary>
        public float Weight => _weight;

        /// <summary>Whether the next pose has anything to do.</summary>
        internal bool Active => _weight > 0f || Drop != 0f;

        /// <summary>
        /// Find the ground under both feet and hand it on, on the main thread once the character
        /// has moved. With <paramref name="wanted"/> false (in the air, seated, out of sight) it
        /// casts nothing and what it did fades out.
        /// </summary>
        public void Update(bool wanted)
        {
            float dt = Time.SmoothDelta;
            _weight = wanted ? MathF.Min(1f, _weight + dt / FadeTime) : MathF.Max(0f, _weight - dt / FadeTime);

            var world = _animator.Transform.WorldMatrix;
            var position = world.Translation;

            // The next pose is drawn after the next move: the entity will be about as far on as this
            // frame took it. Without this a foot that is down sinks or rises by a frame's move.
            var motion = _hasPosition ? position - _lastPosition : Vector3.Zero;
            _lastPosition = position;
            _hasPosition = true;

            // Put somewhere else (a teleport, a respawn): what was under the feet is no more, and
            // nothing eases from there
            if (motion.LengthSquared() > 1f)
            {
                motion = Vector3.Zero;
                _left.Looked = _left.Placed = false;
                _right.Looked = _right.Placed = false;
            }

            if (_weight <= 0f)
            {
                _left.Looked = _left.Placed = false;
                _right.Looked = _right.Placed = false;
                return;
            }

            var next = world;
            next.Translation = position + motion;
            if (!Matrix4x4.Invert(next, out var toEntity)) return;

            var right = new Vector2(world.M11, world.M13);
            var forward = new Vector2(world.M31, world.M33);
            if (right.LengthSquared() < 1e-6f || forward.LengthSquared() < 1e-6f) return;
            right = Vector2.Normalize(right);
            forward = Vector2.Normalize(forward);

            Place(true, ref _left, world, toEntity, position.Y, motion.Y, right, forward, dt);
            Place(false, ref _right, world, toEntity, position.Y, motion.Y, right, forward, dt);
        }

        /// <summary>
        /// Hand on the ground by other means than the rays of <see cref="Update"/>: a character whose
        /// ground is known, a test.
        /// </summary>
        public void Stand(in FootGround left, in FootGround right, float weight = 1f)
        {
            _left.Ground = left;
            _right.Ground = right;
            _weight = weight;
        }

        private void Place(bool left, ref Foot foot, in Matrix4x4 world, in Matrix4x4 toEntity, float feetY, float rise, Vector2 right, Vector2 forward, float dt)
        {
            if (!_animator.TryGetSole(left, out var heel, out var ball, out var ankle)) return;
            heel = Vector3.Transform(heel, world);
            ball = Vector3.Transform(ball, world);
            ankle = Vector3.Transform(ankle, world);

            // A foot that is down keeps the ground it landed on: nothing new to find under it,
            // however long it stands
            bool down = (left ? LeftDown : RightDown) >= 0.5f;
            float apart = down ? HeldCast : SwingCast;
            float x = ball.X - foot.LookedAt.X, z = ball.Z - foot.LookedAt.Z;
            if (!foot.Looked || x * x + z * z > apart * apart)
                Look(ref foot, heel, ball, feetY);

            // No ground within reach: the foot plays as the clip has it
            float height = feetY + rise;
            var slope = Vector2.Zero;
            if (foot.Found)
            {
                slope = foot.Gradient;
                height = foot.Anchor.Y + slope.X * (ankle.X - foot.Anchor.X) + slope.Y * (ankle.Z - foot.Anchor.Z);
            }

            if (!foot.Placed)
            {
                foot.Placed = true;
                foot.Height = height;
                foot.Slope = slope;
            }
            else
            {
                float ease = down ? PlantedEase : height > foot.Height ? LiftEase : LowerEase;
                float share = MathF.Min(1f, dt / ease);
                foot.Height += (height - foot.Height) * share;
                foot.Slope += (slope - foot.Slope) * share;
            }

            var at = Vector3.Transform(new Vector3(ankle.X, foot.Height, ankle.Z), toEntity);
            foot.Ground = new FootGround
            {
                At = new Vector2(at.X, at.Z),
                Height = at.Y,
                Slope = new Vector2(Vector2.Dot(foot.Slope, right), Vector2.Dot(foot.Slope, forward)),
            };
        }

        // What the two rays of a foot meet: the higher of the two carries it, and it lies from there
        // the way the ground lies between them, no steeper than MaxSlope
        private void Look(ref Foot foot, Vector3 heel, Vector3 ball, float feetY)
        {
            foot.Looked = true;
            foot.LookedAt = ball;

            bool onBall = Ray(ball.X, ball.Z, feetY, out float ballY, out var ballNormal);
            bool onHeel = Ray(heel.X, heel.Z, feetY, out float heelY, out var heelNormal);
            foot.Found = onBall || onHeel;
            if (!foot.Found) return;

            var normal = onBall && onHeel ? ballNormal + heelNormal : onBall ? ballNormal : heelNormal;
            float up = MathF.Max(normal.Y, 0.1f);
            var gradient = new Vector2(-normal.X / up, -normal.Z / up);

            if (onBall && onHeel)
            {
                // Along the foot the two heights say it, also across a step's edge, where both normals are level
                var along = new Vector2(ball.X - heel.X, ball.Z - heel.Z);
                float length = along.Length();
                if (length > 0.01f)
                {
                    along /= length;
                    gradient += along * ((ballY - heelY) / length - Vector2.Dot(gradient, along));
                }
            }

            float steepest = MaxSlope * MaxSlope;
            float steep = gradient.LengthSquared();
            if (steep > steepest) gradient *= MathF.Sqrt(steepest / steep);

            foot.Gradient = gradient;
            foot.Anchor = !onHeel || (onBall && ballY >= heelY)
                ? new Vector3(ball.X, ballY, ball.Z)
                : new Vector3(heel.X, heelY, heel.Z);
        }

        private bool Ray(float x, float z, float feetY, out float y, out Vector3 normal)
        {
            float top = feetY + ReachUp, length = ReachUp + ReachDown;
            _hitAt = length;
            _hitNormal = Vector3.UnitY;
            _hitAny = false;
            _onHits ??= OnHits;
            PhysicsWorld.Scene?.Raycast(new Vector3(x, top, z), -Vector3.UnitY, length, 8, _onHits, HitFlag.Normal);
            y = top - _hitAt;
            normal = _hitNormal;
            return _hitAny;
        }

        // The nearest solid thing a Collider made, what a character stands on: never a capsule, its
        // own least of all, nor a trigger. A ray that starts inside something (a wall it brushes)
        // meets it where it starts: no ground.
        private bool OnHits(RaycastHit[] hits)
        {
            if (hits == null) return true;
            foreach (var hit in hits)
            {
                if (hit == null || hit.Distance >= _hitAt || hit.Distance <= 0.001f) continue;
                var shape = hit.Shape;
                if (shape == null || (shape.Flags & ShapeFlag.TriggerShape) != 0 || shape.UserData is not Collider) continue;
                _hitAt = hit.Distance;
                _hitNormal = hit.Normal;
                _hitAny = true;
            }
            return true;
        }

        /// <summary>
        /// Move the legs of a pose that has gone up the hierarchy (the pelvis is lowered by then, by
        /// <see cref="Drop"/>): bones in the skeleton's model space, <paramref name="flipX"/>
        /// for a skeleton turned half round from its entity. <paramref name="leftDown"/> and
        /// <paramref name="rightDown"/> are what the clips of this pose say. <paramref name="own"/>:
        /// this is the skeleton the character stands on, the one that moves the eased values on;
        /// any other takes them as they are.
        /// </summary>
        internal void Solve(LegChain legs, Matrix4x4[] bones, bool flipX, float leftDown, float rightDown, bool own, float dt)
        {
            if (own)
            {
                float share = MathF.Min(1f, dt / StanceEase);
                LeftDown += (leftDown - LeftDown) * share;
                RightDown += (rightDown - RightDown) * share;
            }

            float sign = flipX ? -1f : 1f;
            float leftNeed = Leg(legs.Left, _left.Ground, LeftDown, bones, sign);
            float rightNeed = Leg(legs.Right, _right.Ground, RightDown, bones, sign);
            if (!own) return;

            // The foot in the air has no say: nothing has to reach down for it
            float want = MathF.Max(-MaxDrop, MathF.Min(leftNeed * LeftDown, rightNeed * RightDown));
            Drop += (want - Drop) * MathF.Min(1f, dt / (want < Drop ? DropEase : RiseEase));
            if (MathF.Abs(Drop) < 0.0001f && want == 0f) Drop = 0f;
        }

        /// <summary>One leg. Returns how far the pelvis has to come down for this foot to reach: 0 or less.</summary>
        private float Leg(LegChain.Leg leg, in FootGround ground, float down, Matrix4x4[] bones, float sign)
        {
            var hip = bones[leg.Upper].Translation;
            var knee = bones[leg.Lower].Translation;
            var ankle = bones[leg.Foot].Translation;
            var heel = Vector3.Transform(leg.Sole.Heel, bones[leg.Foot]);
            var ball = Vector3.Transform(leg.Sole.Ball, bones[leg.Foot]);

            // The ground in the skeleton's space, by how much of it applies
            float atX = ground.At.X * sign, atZ = ground.At.Y * sign;
            float riseX = ground.Slope.X * sign, riseZ = ground.Slope.Y * sign;
            float height = ground.Height;
            float weight = _weight;

            // A foot that is down lies along the ground
            var tilt = Quaternion.Identity;
            float lay = weight * down;
            if (lay > 0f && (riseX != 0f || riseZ != 0f))
            {
                var normal = Vector3.Normalize(new Vector3(-riseX * lay, 1f, -riseZ * lay));
                tilt = Quaternion.Normalize(new Quaternion(normal.Z, 0f, -normal.X, 1f + normal.Y));
            }

            // The foot goes where the lowest of its sole is as far above the ground as the clip has
            // it above its own
            var heelFrom = Vector3.Transform(heel - ankle, tilt);
            var ballFrom = Vector3.Transform(ball - ankle, tilt);
            float overHeel = ankle.Y + heelFrom.Y - weight * (height + riseX * (ankle.X + heelFrom.X - atX) + riseZ * (ankle.Z + heelFrom.Z - atZ));
            float overBall = ankle.Y + ballFrom.Y - weight * (height + riseX * (ankle.X + ballFrom.X - atX) + riseZ * (ankle.Z + ballFrom.Z - atZ));
            float rise = MathF.Min(heel.Y, ball.Y) - Drop - MathF.Min(overHeel, overBall);

            if (MathF.Abs(rise) < NoRise && tilt == Quaternion.Identity) return 0f;

            var target = new Vector3(ankle.X, ankle.Y + rise, ankle.Z);
            float length = Vector3.Distance(hip, knee) + Vector3.Distance(knee, ankle);
            float soft = length * Soft;

            // How far the pelvis has to come down from where the clip has it for the leg to reach
            float free = TwoBoneIK.FreeReach(Vector3.Distance(hip, ankle), length, soft);
            float x = target.X - hip.X, z = target.Z - hip.Z;
            float room = free * free - x * x - z * z;
            float need = room > 0f ? MathF.Min(0f, target.Y + MathF.Sqrt(room) - (hip.Y - Drop)) : -MaxDrop;

            // The knee keeps the side the clip has it on, also in a leg that stands all but straight,
            // where the way the foot points decides it
            var line = Vector3.Normalize(ankle - hip);
            var kneeOut = knee - hip;
            kneeOut -= Vector3.Dot(kneeOut, line) * line;
            var pole = hip + kneeOut + (ball - heel) * KneeFollowsFoot / MathF.Max(Vector3.Distance(ball, heel), 0.01f);

            if (!TwoBoneIK.Solve(hip, knee, ankle, target, pole, soft, out var thigh, out var shin, out var kneeTo, out var ankleTo))
                return need;

            // Each part turns about its upper joint, and carries on what the part above did
            var upper = About(hip, thigh);
            var lower = upper * About(kneeTo, shin);
            var foot = lower * About(ankleTo, Quaternion.Normalize(tilt * Quaternion.Inverse(Quaternion.Normalize(shin * thigh))));

            foreach (int bone in leg.Thigh) bones[bone] *= upper;
            foreach (int bone in leg.Shin) bones[bone] *= lower;
            foreach (int bone in leg.Ankle) bones[bone] *= foot;

            return need;
        }

        // A turn about a point of the model space, to go after a bone's model matrix
        private static Matrix4x4 About(Vector3 point, Quaternion turn)
            => Matrix4x4.CreateTranslation(-point) * Matrix4x4.CreateFromQuaternion(turn) * Matrix4x4.CreateTranslation(point);
    }
}
