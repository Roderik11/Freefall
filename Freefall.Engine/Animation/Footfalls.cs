using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Freefall.Animation
{
    /// <summary>
    /// A foot coming down in a clip, found from what the clip does with the foot
    /// (<see cref="AnimationClip.GetFootfalls"/>): nobody has to mark it.
    /// </summary>
    public readonly struct Footfall
    {
        /// <summary>When, as a fraction of the clip's length.</summary>
        public readonly float Time;

        public readonly bool Left;

        /// <summary>
        /// How fast the sole comes down, in hip heights per second: about 0.5 for a walk, 1 to 2
        /// for a run, more for a landing. Sizes do not matter to it.
        /// </summary>
        public readonly float Strength;

        public Footfall(float time, bool left, float strength)
        {
            Time = time;
            Left = left;
            Strength = strength;
        }
    }

    /// <summary>The bones of a humanoid skeleton that say where its feet are; a part it does not have is -1.</summary>
    internal sealed class FootBones
    {
        public int Hips = -1;
        public int LeftFoot = -1, LeftToes = -1, RightFoot = -1, RightToes = -1;

        private static readonly ConditionalWeakTable<Skeleton, FootBones> s_feet = new();

        public static FootBones Of(Skeleton skeleton) => s_feet.GetValue(skeleton, static s =>
        {
            var feet = new FootBones();
            var humanoid = HumanoidTemplate.Detect(s);
            if (humanoid == null) return feet;

            int Find(HumanoidBone bone) => humanoid.TryGetValue(bone, out int index) ? index : -1;
            feet.Hips = Find(HumanoidBone.Hips);
            feet.LeftFoot = Find(HumanoidBone.LeftFoot);
            feet.LeftToes = Find(HumanoidBone.LeftToes);
            feet.RightFoot = Find(HumanoidBone.RightFoot);
            feet.RightToes = Find(HumanoidBone.RightToes);
            return feet;
        });
    }

    /// <summary>Which humanoid bone is each one's counterpart on the other side.</summary>
    internal static class HumanoidSides
    {
        private static readonly HumanoidBone[] s_other = Build();

        /// <summary>The same bone on the other side; a bone of the middle is its own.</summary>
        public static HumanoidBone Other(HumanoidBone bone)
            => (uint)bone < (uint)s_other.Length ? s_other[(int)bone] : bone;

        private static HumanoidBone[] Build()
        {
            var other = new HumanoidBone[(int)HumanoidBone.Count];
            for (int i = 0; i < other.Length; i++)
            {
                var bone = (HumanoidBone)i;
                string name = bone.ToString();
                string swapped = name.StartsWith("Left", StringComparison.Ordinal) ? "Right" + name.Substring(4)
                               : name.StartsWith("Right", StringComparison.Ordinal) ? "Left" + name.Substring(5)
                               : null;
                other[i] = swapped != null && Enum.TryParse<HumanoidBone>(swapped, out var found) ? found : bone;
            }
            return other;
        }
    }

    internal static class Footfalls
    {
        // In hip heights (0.04 is about 4 cm on a person), above the ground the clip stands on.
        //
        // A foot is off the ground once its sole is more than Lifted above it: a crouched
        // shuffle lifts the ball of the foot 5 cm, a walk 9. It has come down when the sole is
        // back within Down, or when, within Near, it stops coming down: a walk meets the ground
        // heel first and rocks on it for a fifth of the stride with the sole still 3 cm up,
        // and the step is heard when the heel arrives, not when the foot lies flat.
        private const float Lifted = 0.04f;
        private const float Down = 0.02f;
        private const float Near = 0.06f;

        // Coming down slower than this, in hip heights per second, the sole has stopped
        private const float Stopped = 0.15f;

        // A foot does not come down twice within this many seconds: the heel and then the rest
        // of the same step are one
        private const float Apart = 0.3f;

        // How far behind the ankle the heel meets the ground, as a share of ankle to ball
        private const float HeelBack = 0.3f;

        // A rig whose toe joints are within this of y = 0 in the bind pose stands on y = 0
        private const float StandsOnZero = 0.25f;

        /// <summary>
        /// Where each foot comes down in a clip, in the order they do. The sole is two points
        /// that move with the foot bone and are on the ground in the bind pose: one under the
        /// ball of the foot (the toe joint) and one under the heel. Whichever is lower counts,
        /// so a heel strike in a walk and a landing on the forefoot in a sprint are both found.
        /// </summary>
        public static Footfall[] Find(Bone[] bones, AnimationChannel[] channels, FootBones feet, float duration, float ticksPerSecond)
        {
            if (feet == null || duration <= 0f || ticksPerSecond <= 0f) return Array.Empty<Footfall>();
            if (feet.LeftFoot < 0 && feet.RightFoot < 0) return Array.Empty<Footfall>();

            int count = bones.Length;

            // Only the bones that carry the feet
            var needed = new bool[count];
            void Need(int bone)
            {
                for (; bone >= 0 && !needed[bone]; bone = bones[bone].Parent)
                    needed[bone] = true;
            }
            Need(feet.Hips);
            Need(feet.LeftFoot);
            Need(feet.LeftToes);
            Need(feet.RightFoot);
            Need(feet.RightToes);

            // Model matrices at a key time (parents come before their children)
            var model = new Matrix4x4[count];
            void Pose(float ticks, bool bind)
            {
                for (int i = 0; i < count; i++)
                {
                    if (!needed[i]) continue;

                    var pose = bones[i].BindPose;
                    if (!bind) channels[i]?.GetBoneTransform(bones[i], ticks, ref pose);
                    var local = Matrix4x4.CreateScale(pose.Scale)
                              * Matrix4x4.CreateFromQuaternion(pose.Rotation)
                              * Matrix4x4.CreateTranslation(pose.Position);
                    model[i] = bones[i].Parent >= 0 ? local * model[bones[i].Parent] : local;
                }
            }

            Pose(0f, bind: true);

            float hipHeight = feet.Hips >= 0 ? MathF.Abs(model[feet.Hips].Translation.Y) : 0f;
            if (hipHeight < 1e-4f) hipHeight = 1f;

            var left = new Sole(feet.LeftFoot, feet.LeftToes, model, hipHeight);
            var right = new Sole(feet.RightFoot, feet.RightToes, model, hipHeight);

            // One sample per key frame
            int steps = Math.Clamp((int)MathF.Ceiling(duration), 8, 600);
            var leftHeight = new float[steps + 1];
            var rightHeight = new float[steps + 1];
            for (int k = 0; k <= steps; k++)
            {
                Pose(duration * k / steps, bind: false);
                leftHeight[k] = left.Height(model);
                rightHeight[k] = right.Height(model);
            }

            var found = new List<Footfall>();
            float perSecond = steps / (duration / ticksPerSecond);
            if (left.Foot >= 0) Steps(leftHeight, true, hipHeight, perSecond, found);
            if (right.Foot >= 0) Steps(rightHeight, false, hipHeight, perSecond, found);

            found.Sort(static (a, b) => a.Time.CompareTo(b.Time));
            return found.ToArray();
        }

        private static void Steps(float[] height, bool left, float hipHeight, float samplesPerSecond, List<Footfall> found)
        {
            int last = height.Length - 1;

            // The ground the clip stands on: planted feet sit a centimetre or two lower than
            // the bind pose has them. A foot that never gets down there (a fall) is measured
            // from where the bind pose stands.
            float ground = 0f;
            for (int k = 0; k <= last; k++)
                ground = MathF.Min(ground, height[k]);
            for (int k = 0; k <= last; k++)
                height[k] -= ground;

            float lifted = Lifted * hipHeight, down = Down * hipHeight, near = Near * hipHeight;
            float stopped = Stopped * hipHeight / samplesPerSecond;
            int apart = (int)MathF.Ceiling(Apart * samplesPerSecond);

            // A clip that ends where it starts goes round
            bool cyclic = MathF.Abs(height[0] - height[last]) < down;

            // Walk the clip with the foot up or down. `peak` is how high it got since it left
            // the ground, `since` how many samples ago it last came down.
            bool up = height[0] > lifted;
            float peak = height[0];
            int since = apart;

            // Whether the foot, being up, has come down at sample k
            bool Lands(int k)
            {
                float h = height[k];
                if (h < down) return true;
                if (h > near || h > (peak + down) * 0.5f) return false;

                float next = k < last ? height[k + 1] : cyclic ? height[1] : h;
                return h - next < stopped;
            }

            // Once round first, for the state the clip is entered in: that of its own end
            if (cyclic)
            {
                for (int k = 1; k <= last; k++)
                    Step(k, record: false);
            }

            for (int k = 1; k <= last; k++)
                Step(k, record: true);

            void Step(int k, bool record)
            {
                since++;
                if (!up)
                {
                    if (height[k] > lifted && since >= apart)
                    {
                        up = true;
                        peak = height[k];
                    }
                    return;
                }

                peak = MathF.Max(peak, height[k]);
                if (!Lands(k)) return;

                up = false;
                since = 0;
                if (!record) return;

                // Between two samples: where the sole passes the line, if that is how it came down
                float time = k;
                float fall = height[k - 1] - height[k];
                if (height[k] < down && height[k - 1] >= down && fall > 1e-6f)
                    time = k - 1 + (height[k - 1] - down) / fall;

                found.Add(new Footfall(Math.Clamp(time / last, 0f, 1f), left, Speed(height, k, samplesPerSecond) / hipHeight));
            }
        }

        // How fast the sole was coming down over the two samples before one
        private static float Speed(float[] height, int at, float samplesPerSecond)
        {
            int from = Math.Max(at - 2, 0);
            return at > from ? MathF.Max(0f, (height[from] - height[at]) / (at - from)) * samplesPerSecond : 0f;
        }

        // The two points of a foot that meet the ground, fixed to the foot bone
        private readonly struct Sole
        {
            public readonly int Foot;
            private readonly bool _twoPoints;
            private readonly Vector3 _heel, _ball;   // in the foot bone's space
            private readonly float _ground;          // where both stand in the bind pose

            public Sole(int foot, int toes, Matrix4x4[] bind, float hipHeight)
            {
                Foot = foot;
                _twoPoints = false;
                _heel = _ball = Vector3.Zero;
                _ground = 0f;
                if (foot < 0) return;

                // The ground of the bind pose: y = 0 for a rig that stands on it, else the
                // height of its lowest joint
                var ankle = bind[foot].Translation;
                var ball = toes >= 0 ? bind[toes].Translation : ankle;
                _ground = ball.Y >= 0f && ball.Y < StandsOnZero * hipHeight ? 0f : ball.Y;

                if (!Matrix4x4.Invert(bind[foot], out var toFoot))
                {
                    // The joint itself, by how far it is from where the bind pose has it
                    _ground = ankle.Y;
                    return;
                }

                // Under the ball, and under the heel a little behind the ankle. With no toe
                // joint to say which way the foot points: under the ankle alone.
                _ball = Vector3.Transform(new Vector3(ball.X, _ground, ball.Z), toFoot);
                _heel = Vector3.Transform(new Vector3(ankle.X - HeelBack * (ball.X - ankle.X), _ground, ankle.Z - HeelBack * (ball.Z - ankle.Z)), toFoot);
                _twoPoints = true;
            }

            /// <summary>How far the sole is above the ground of the bind pose.</summary>
            public float Height(Matrix4x4[] model)
            {
                if (Foot < 0) return 0f;
                if (!_twoPoints) return model[Foot].Translation.Y - _ground;

                float heel = Vector3.Transform(_heel, model[Foot]).Y;
                float ball = Vector3.Transform(_ball, model[Foot]).Y;
                return MathF.Min(heel, ball) - _ground;
            }
        }
    }
}
