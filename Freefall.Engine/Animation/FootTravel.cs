using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Freefall.Animation
{
    /// <summary>
    /// The travel a clip played in place stands for, worked out from its feet. An animal pack's walk and run keep the
    /// body where it is while the feet slide back under it; moved at an even pace such a clip slides its feet within
    /// every step, because a body surges on each push-off and slows as each foot lands. <see cref="Add"/> puts the
    /// travel the feet imply on the clip's root bone, frame by frame: the body goes on by as much as the feet that are
    /// down slide back, and through a moment with none down (a gallop's leap) at the pace it had. The clip is then
    /// like one captured moving, and root motion (<see cref="RootMotionTrack"/>) takes it out as usual: the even pace
    /// moves the character, the surge stays in the pose, and the feet that are down stay where they are.
    ///
    /// The feet are found as on any rig: the leaf bones in the lowest part of the bind pose that are not a tail. The
    /// root is a humanoid's hips, or else the bone where the body splits (<see cref="RootOf"/>), which is also where
    /// an Animator takes root motion from on a rig without hips.
    /// </summary>
    public static class FootTravel
    {
        // The feet: leaf bones in this lowest share of the rig's height (lowest to highest joint)
        const float FeetBelow = 0.15f;

        // A foot is down while it is within this share of the rig's height of its ground, no higher over it than this
        // share above the lowest foot of the moment (each over its own ground), and moving up or down at under this
        // share a second. Its ground is the height it is at or under this share of the time: the lowest point can be
        // a swing that dips through the ground, and a gallop's foot is in the air most of the time. Without the
        // second test a chicken's swinging foot, which it lifts by a centimetre or two, counted as down: it went the
        // other way and halved the pace (0.22 m/s where the foot that is down slides at 0.47).
        const float DownWithin = 0.03f;
        const float LowestWithin = 0.012f;
        const float DownStillerThan = 0.1f;
        const float GroundShare = 0.1f;

        // Travel shorter than this share of the rig's height over the whole clip is none
        const float LeastTravel = 0.02f;

        // How many harmonics of the cycle the surge keeps (a walk's four footfalls are the fourth)
        const int SurgeHarmonics = 4;

        /// <summary>How far a body's pace swings either way of its mean over a stride, as a share of it, by default.</summary>
        public const float DefaultSurge = 0.2f;

        /// <summary>
        /// The share of the frames that had a foot down in the last <see cref="Add"/> on this thread (0 when it found
        /// no feet): for tools that judge how much of a clip the travel was read from.
        /// </summary>
        [ThreadStatic] public static float LastDownShare;

        /// <summary>The body's speed over the ground frame by frame in the last <see cref="Add"/> on this thread, for tools.</summary>
        [ThreadStatic] public static float[] LastSpeeds;

        static readonly ConditionalWeakTable<Skeleton, StrongBox<int>> s_roots = new();

        // Going down from the top for the root: on into a child that holds at least this share of the bones below
        const float BodyShare = 0.9f;

        /// <summary>
        /// The bone a skeleton travels by: a humanoid's hips; for another rig with feet, the bone where its body
        /// splits (the pelvis or spine root of a four-legged rig); -1 for one with neither. Found by the hierarchy
        /// alone, going down from the top while one child holds nearly all the bones below: the files of a rig agree
        /// on it whatever their bind pose. The bone the feet hang from did not: in the horse's run file the hind
        /// toes are posed too high to count as feet, its feet's common bone was the chest, and the travel put there
        /// pulled the front half away from the back (2026-10-23).
        /// </summary>
        public static int RootOf(Skeleton skeleton) => s_roots.GetValue(skeleton, static s =>
        {
            var humanoid = HumanoidTemplate.Detect(s);
            if (humanoid != null && humanoid.TryGetValue(HumanoidBone.Hips, out int hips))
                return new StrongBox<int>(hips);
            return new StrongBox<int>(FeetOf(s.Bones, out _).Length >= 2 ? Split(s.Bones) : -1);
        }).Value;

        // From the top bone with the most below it, down into the child holding at least BodyShare of what is below,
        // to the bone where that no longer holds. Parents come before their children in a skeleton.
        static int Split(Bone[] bones)
        {
            int n = bones.Length;
            var below = new int[n];
            for (int i = n - 1; i >= 0; i--)
                if (bones[i].Parent >= 0) below[bones[i].Parent] += below[i] + 1;

            int at = -1;
            for (int i = 0; i < n; i++)
                if (bones[i].Parent < 0 && (at < 0 || below[i] > below[at])) at = i;

            while (at >= 0)
            {
                int biggest = -1;
                for (int i = 0; i < n; i++)
                    if (bones[i].Parent == at && (biggest < 0 || below[i] > below[biggest])) biggest = i;
                if (biggest < 0 || below[biggest] + 1 < BodyShare * below[at]) break;
                at = biggest;
            }
            return at;
        }

        /// <summary>
        /// Put the travel its feet imply on a clip played in place, as root motion's keys on <see cref="RootOf"/>.
        /// With <paramref name="even"/> (m/s, in the skeleton's model space) an even travel at that velocity instead,
        /// for a clip whose feet are hardly ever seen down. <paramref name="surge"/>: how far the pace may swing either
        /// way of its mean over a stride, as a share of it (<see cref="Surge"/>; 1 or more keeps a hop's bursts).
        /// False, and the clip as it was, when the skeleton has no feet or root, no foot is ever down, or the feet do
        /// not slide.
        /// </summary>
        public static bool Add(AnimationClip clip, Skeleton skeleton, Vector3 even = default, float surge = DefaultSurge)
        {
            LastDownShare = 0f;
            var bones = skeleton.Bones;
            bool byFeet = even.X == 0f && even.Z == 0f;
            var feet = FeetOf(bones, out float height);

            // The root by the hierarchy, as RootOf finds it on the rig, without asking for feet first: a clip file's
            // bind pose need not stand on them (the hound's gallop file), and an even travel does not read them
            var humanoid = HumanoidTemplate.Detect(skeleton);
            int root = humanoid != null && humanoid.TryGetValue(HumanoidBone.Hips, out int hips) ? hips : Split(bones);
            if (root < 0 || (byFeet && feet.Length < 2) || clip.Duration <= 0f || clip.TicksPerSecond <= 0f) return false;

            // On the clip's own frames: between two keys a leg turns and its foot arcs up, which would read as a foot
            // leaving the ground in the middle of a gallop's short contact
            var channels = clip.ChannelsFor(bones);
            int frames = Math.Max(2, (int)MathF.Round(clip.Duration));
            float dt = clip.Duration / frames / clip.TicksPerSecond;

            var model = new Matrix4x4[bones.Length];
            var at = new Vector3[feet.Length, frames + 1];
            var aboveRoot = new Matrix4x4[frames + 1];
            var rootPosition = new Vector3[frames + 1];
            for (int s = 0; s <= frames; s++)
            {
                float ticks = clip.Duration * s / frames;
                for (int i = 0; i < bones.Length; i++)
                {
                    var pose = bones[i].BindPose;
                    channels[i]?.GetBoneTransform(bones[i], ticks, ref pose);
                    var local = Matrix4x4.CreateScale(pose.Scale) * Matrix4x4.CreateFromQuaternion(pose.Rotation) * Matrix4x4.CreateTranslation(pose.Position);
                    model[i] = bones[i].Parent >= 0 ? local * model[bones[i].Parent] : local;
                    if (i == root) rootPosition[s] = pose.Position;
                }
                for (int f = 0; f < feet.Length; f++) at[f, s] = model[feet[f]].Translation;
                aboveRoot[s] = bones[root].Parent >= 0 ? model[bones[root].Parent] : Matrix4x4.Identity;
            }

            var travel = new Vector2[frames + 1];
            if (!byFeet)
            {
                for (int s = 0; s <= frames; s++)
                    travel[s] = new Vector2(even.X, even.Z) * (s * dt);
                return Put(clip, bones[root].Name, travel, aboveRoot, rootPosition, frames);
            }

            // Each foot's ground
            var ground = new float[feet.Length];
            var heights = new float[frames + 1];
            for (int f = 0; f < feet.Length; f++)
            {
                for (int s = 0; s <= frames; s++) heights[s] = at[f, s].Y;
                Array.Sort(heights);
                ground[f] = heights[(int)(GroundShare * frames)];
            }

            // Frame to frame, the body goes on by as much as the feet that are down slide back
            var velocity = new Vector2[frames];
            var known = new bool[frames];
            float within = DownWithin * height, lowestWithin = LowestWithin * height, stiller = DownStillerThan * height;
            int anyKnown = 0;
            for (int s = 0; s < frames; s++)
            {
                float lowest = float.MaxValue;
                for (int f = 0; f < feet.Length; f++)
                    lowest = MathF.Min(lowest, at[f, s].Y - ground[f]);

                var slide = Vector2.Zero;
                int down = 0;
                for (int f = 0; f < feet.Length; f++)
                {
                    var a = at[f, s];
                    var b = at[f, s + 1];
                    float over = a.Y - ground[f];
                    if (MathF.Abs(over) > within || over > lowest + lowestWithin) continue;
                    if (MathF.Abs(b.Y - a.Y) / dt > stiller) continue;
                    slide += new Vector2(b.X - a.X, b.Z - a.Z);
                    down++;
                }
                if (down == 0) continue;
                velocity[s] = -slide / (down * dt);
                known[s] = true;
                anyKnown++;
            }
            LastDownShare = (float)anyKnown / frames;
            if (anyKnown == 0) return false;

            // With no foot down (a leap), the pace between the last frame one was and the next, round the loop
            for (int s = 0; s < frames; s++)
            {
                if (known[s]) continue;
                int back = 1, ahead = 1;
                while (!known[(s - back + frames) % frames]) back++;
                while (!known[(s + ahead) % frames]) ahead++;
                float t = (float)back / (back + ahead);
                velocity[s] = Vector2.Lerp(velocity[(s - back + frames) % frames], velocity[(s + ahead) % frames], t);
            }

            var smooth = Surge(velocity, surge);

            LastSpeeds = new float[frames];
            for (int s = 0; s < frames; s++)
            {
                travel[s + 1] = travel[s] + smooth[s] * dt;
                LastSpeeds[s] = smooth[s].Length();
            }
            if (travel[frames].Length() < LeastTravel * height) return false;

            return Put(clip, bones[root].Name, travel, aboveRoot, rootPosition, frames);
        }

        /// <summary>
        /// The surge the feet show, as a body has it: round the loop, only the first <see cref="SurgeHarmonics"/>
        /// harmonics of the cycle (a body swells and eases with each step, it does not jerk where a foot lifts), and
        /// swinging at most <paramref name="surge"/> of the mean pace either way. The mean, the clip's travel, is kept.
        /// Hand-made in-place walks hold both feet still where the weight passes from one to the other: read as it
        /// is, the deer's walk stopped dead twice a stride and rushed between (0 to 2 times its pace), which the user
        /// saw as robots (2026-10-23).
        /// </summary>
        static Vector2[] Surge(Vector2[] velocity, float surge)
        {
            int n = velocity.Length;

            // A hop keeps its bursts: only smoothed a little round the loop (a foot settling or lifting at the edge of
            // the ground band is not the body)
            if (surge >= 1f)
            {
                var hop = new Vector2[n];
                for (int s = 0; s < n; s++)
                    hop[s] = velocity[(s - 1 + n) % n] * 0.25f + velocity[s] * 0.5f + velocity[(s + 1) % n] * 0.25f;
                return hop;
            }

            var mean = Vector2.Zero;
            for (int s = 0; s < n; s++) mean += velocity[s];
            mean /= n;

            // The low harmonics of what swings about the mean
            var swing = new Vector2[n];
            for (int k = 1; k <= Math.Min(SurgeHarmonics, n / 2); k++)
            {
                Vector2 re = Vector2.Zero, im = Vector2.Zero;
                for (int s = 0; s < n; s++)
                {
                    var (sin, cos) = MathF.SinCos(2f * MathF.PI * k * s / n);
                    re += velocity[s] * cos;
                    im += velocity[s] * sin;
                }
                re *= 2f / n;
                im *= 2f / n;
                for (int s = 0; s < n; s++)
                {
                    var (sin, cos) = MathF.SinCos(2f * MathF.PI * k * s / n);
                    swing[s] += re * cos + im * sin;
                }
            }

            // No further either way than the surge allows
            float pace = mean.Length(), widest = 0f;
            for (int s = 0; s < n; s++) widest = MathF.Max(widest, swing[s].Length());
            float keep = pace > 0f && widest > surge * pace ? surge * pace / widest : 1f;

            var result = new Vector2[n];
            for (int s = 0; s < n; s++) result[s] = mean + swing[s] * keep;
            return result;
        }

        // The travel on the root's position keys, in the space they are in (its parent's), a key a frame
        static bool Put(AnimationClip clip, string root, Vector2[] travel, Matrix4x4[] aboveRoot, Vector3[] rootPosition, int frames)
        {
            var keys = new List<VectorKey>(frames + 1);
            for (int s = 0; s <= frames; s++)
            {
                if (!Matrix4x4.Invert(aboveRoot[s], out var toParent)) return false;
                var add = Vector3.TransformNormal(new Vector3(travel[s].X, 0f, travel[s].Y), toParent);
                keys.Add(new VectorKey { Time = clip.Duration * s / frames, Value = rootPosition[s] + add });
            }
            clip.SetPositionKeys(root, keys);
            return true;
        }

        /// <summary>
        /// The ends of the legs: leaf bones in the lowest part of the bind pose that are not a tail (a tail's tip lies
        /// low too, and goes along with the body). <paramref name="height"/>: lowest to highest joint.
        /// </summary>
        internal static int[] FeetOf(Bone[] bones, out float height)
        {
            int n = bones.Length;
            var model = new Matrix4x4[n];
            var hasChild = new bool[n];
            float bottom = float.MaxValue, top = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                var pose = bones[i].BindPose;
                var local = Matrix4x4.CreateScale(pose.Scale) * Matrix4x4.CreateFromQuaternion(pose.Rotation) * Matrix4x4.CreateTranslation(pose.Position);
                model[i] = bones[i].Parent >= 0 ? local * model[bones[i].Parent] : local;
                if (bones[i].Parent >= 0) hasChild[bones[i].Parent] = true;
                bottom = MathF.Min(bottom, model[i].Translation.Y);
                top = MathF.Max(top, model[i].Translation.Y);
            }
            height = MathF.Max(top - bottom, 1e-3f);

            var feet = new List<int>();
            for (int i = 0; i < n; i++)
                if (!hasChild[i] && model[i].Translation.Y < bottom + FeetBelow * height
                    && !(bones[i].Name ?? "").Contains("tail", StringComparison.OrdinalIgnoreCase))
                    feet.Add(i);
            return feet.ToArray();
        }

    }
}
