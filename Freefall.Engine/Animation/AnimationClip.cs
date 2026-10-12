using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Numerics;
using Freefall.Assets;

namespace Freefall.Animation
{
    /// <summary>
    /// Contains animation data for a skeleton.
    /// Loaded from FBX/DAE files.
    /// </summary>
    public class AnimationClip : Asset
    {
        private readonly List<AnimationChannel> _channels = new List<AnimationChannel>();
        public readonly ReadOnlyCollection<AnimationChannel> Channels;

        /// <summary>Animation events triggered at specific times.</summary>
        public List<AnimationEvent> Events = new List<AnimationEvent>();

        /// <summary>The skeleton this clip was imported from (for retargeting).</summary>
        public Skeleton Skeleton { get; set; }

        /// <summary>Duration in ticks.</summary>
        public float Duration;

        /// <summary>Ticks per second (usually 24 or 30).</summary>
        public float TicksPerSecond = 30;

        /// <summary>Duration in seconds.</summary>
        public float DurationSeconds => Duration / TicksPerSecond;

        public float DurationSecondsInverse => 1f / DurationSeconds;

        /// <summary>
        /// Which channel moves each bone of one skeleton (null: none). Channels are matched to
        /// bones by name once per skeleton, so sampling a bone is an array read.
        /// </summary>
        private sealed class Binding
        {
            public Bone[] Bones;
            public AnimationChannel[] Channels;

            // Worked out when root motion is first asked for; replaced as a whole
            public RootBinding Root;

            // Worked out when first asked for
            public Footfall[] Footfalls;
        }

        /// <summary>
        /// The clip's root motion on one skeleton, and the channels that play it in place:
        /// those of the binding, with the travel taken out of the hips' positions.
        /// </summary>
        private sealed class RootBinding
        {
            public int RootBone;
            public RootMotionTrack Track;
            public AnimationChannel[] Channels;

            /// <summary>The same with the clip's turn taken out as well (<see cref="RootMotionTrack.Turn"/>).</summary>
            public AnimationChannel[] TurnChannels;

            /// <summary>The in-place channels with the clip's lift taken out as well; built when first asked for.</summary>
            public AnimationChannel[] LiftChannels;
        }

        // A clip whose lowest point never rises further than this above the ground has no lift (model units)
        private const float LiftThreshold = 0.02f;

        // A clip that ends facing less than this away from where it starts does not turn (radians)
        private const float TurnThreshold = 0.35f;

        // Both in hip heights (0.1 is about 10 cm on a person).
        //
        // A clip that ends less than NetTravel from where it starts does not travel: it is left
        // as it is, with whatever its hips do in between (in-place clips drift a few centimetres,
        // and a jump made for a moving character leans half a metre out and back).
        //
        // A clip that does travel moves in bursts when its hips stray further than PathSurge from
        // an even pace. Walk and run cycles stray 3 to 11 cm, lunges and knock-backs half a metre
        // and more.
        private const float NetTravel = 0.1f;
        private const float PathSurge = 0.2f;

        // Newest first, and replaced as a whole when one is added: Animators update in parallel
        // and share clips, so a reader must never see a list that is still being changed.
        private Binding[] _bindings = [];

        // Skeletons are replaced on reimport: forget the oldest rather than hold on to all of them
        private const int MaxBindings = 16;

        public AnimationClip()
        {
            Channels = new ReadOnlyCollection<AnimationChannel>(_channels);
        }

        public void AddChannel(AnimationChannel channel)
        {
            lock (_channels)
            {
                _channels.Add(channel);
                _bindings = [];
            }
        }

        public void RemoveChannel(AnimationChannel channel)
        {
            lock (_channels)
            {
                _channels.Remove(channel);
                _bindings = [];
            }
        }

        /// <summary>
        /// This clip's channel for each bone of a skeleton, null where it does not move the bone.
        /// The same array comes back for the same <paramref name="bones"/>: do not change it.
        /// </summary>
        internal AnimationChannel[] ChannelsFor(Bone[] bones) => BindingFor(bones).Channels;

        /// <summary>
        /// The channels that play this clip in place on a skeleton whose hips are
        /// <paramref name="rootBone"/>: the travel over the ground is not in them (see
        /// <see cref="GetRootMotion(Bone[], int)"/>). With no root bone, or for a clip that does
        /// not travel, the clip's own channels.
        /// </summary>
        internal AnimationChannel[] ChannelsFor(Bone[] bones, int rootBone, bool turning = false, bool lifting = false)
        {
            var binding = BindingFor(bones);
            if ((uint)rootBone >= (uint)bones.Length) return binding.Channels;

            var root = RootFor(binding, rootBone);
            if (lifting) return LiftFor(binding, root);
            return turning ? root.TurnChannels : root.Channels;
        }

        /// <summary>
        /// The in-place channels with the clip's lift taken out as well: how far the lowest point
        /// of the body is above where it stands in the bind pose (the flight of a jump, the drop
        /// before a landing). The pose then keeps its lowest point on the ground, and whoever
        /// moves the character decides how high it is.
        /// </summary>
        private AnimationChannel[] LiftFor(Binding binding, RootBinding root)
        {
            var lifted = root.LiftChannels;
            if (lifted != null) return lifted;

            lock (_channels)
            {
                return root.LiftChannels ??= BuildLift(binding, root);
            }
        }

        private static AnimationChannel[] BuildLift(Binding binding, RootBinding root)
        {
            var bones = binding.Bones;
            int rootBone = root.RootBone;
            var hips = root.Channels[rootBone];
            var positions = hips?.Position;
            if (positions == null || positions.Count < 1) return root.Channels;

            // The body: the hips and everything below them. What is above them in the hierarchy
            // (a root bone on the ground) does not count.
            var body = new bool[bones.Length];
            for (int i = 0; i < bones.Length; i++)
                body[i] = i == rootBone || (bones[i].Parent >= 0 && body[bones[i].Parent]);

            // Model matrices at a key time (parents come before their children), and the height
            // of the lowest bone of the body
            var model = new Matrix4x4[bones.Length];
            float Lowest(float ticks, bool bind)
            {
                float lowest = float.MaxValue;
                for (int i = 0; i < bones.Length; i++)
                {
                    var pose = bones[i].BindPose;
                    if (!bind) binding.Channels[i]?.GetBoneTransform(bones[i], ticks, ref pose);
                    var local = Matrix4x4.CreateScale(pose.Scale)
                              * Matrix4x4.CreateFromQuaternion(pose.Rotation)
                              * Matrix4x4.CreateTranslation(pose.Position);
                    model[i] = bones[i].Parent >= 0 ? local * model[bones[i].Parent] : local;
                    if (body[i]) lowest = MathF.Min(lowest, model[i].Translation.Y);
                }
                return lowest;
            }

            float ground = Lowest(0f, bind: true);
            int parent = bones[rootBone].Parent;

            int count = positions.Count;
            var keys = new List<VectorKey>(count);
            float highest = 0f;
            for (int k = 0; k < count; k++)
            {
                float lift = MathF.Max(0f, Lowest(positions[k].Time, bind: false) - ground);
                highest = MathF.Max(highest, lift);

                var above = parent >= 0 ? model[parent] : Matrix4x4.Identity;
                if (!Matrix4x4.Invert(above, out var toParent)) return root.Channels;

                var place = Vector3.Transform(positions[k].Value, above);
                place.Y -= lift;
                keys.Add(new VectorKey { Time = positions[k].Time, Value = Vector3.Transform(place, toParent) });
            }

            if (highest < LiftThreshold) return root.Channels;

            var channels = (AnimationChannel[])root.Channels.Clone();
            channels[rootBone] = new AnimationChannel { Target = hips.Target, Position = new VectorKeys(keys), Rotation = hips.Rotation, Scale = hips.Scale };
            return channels;
        }

        /// <summary>The travel over the ground this clip carries on the hips of a skeleton.</summary>
        public RootMotionTrack GetRootMotion(Bone[] bones, int rootBone)
        {
            if ((uint)rootBone >= (uint)bones.Length) return RootMotionTrack.Nothing;
            return RootFor(BindingFor(bones), rootBone).Track;
        }

        /// <summary>
        /// The travel over the ground this clip carries on a humanoid skeleton: its kind, how far
        /// it goes and its pace. For tools; an Animator works it out for its own skeleton.
        /// </summary>
        public RootMotionTrack GetRootMotion(Skeleton skeleton)
        {
            var humanoid = HumanoidTemplate.Detect(skeleton);
            if (humanoid == null || !humanoid.TryGetValue(HumanoidBone.Hips, out int hips))
                return RootMotionTrack.Nothing;
            return GetRootMotion(skeleton.Bones, hips);
        }

        /// <summary>
        /// Where each foot comes down in this clip on a humanoid skeleton, in the order they do
        /// (empty for a skeleton without feet, and for a clip that keeps them where they are).
        /// Found from the clip itself, once per skeleton. The same array comes back every time:
        /// do not change it.
        /// </summary>
        public Footfall[] GetFootfalls(Skeleton skeleton)
        {
            var binding = BindingFor(skeleton.Bones);
            var found = binding.Footfalls;
            if (found != null) return found;

            lock (_channels)
            {
                return binding.Footfalls ??= Footfalls.Find(binding.Bones, binding.Channels, FootBones.Of(skeleton), Duration, TicksPerSecond);
            }
        }

        /// <summary>
        /// A playback time in seconds as the time of this clip's keys: inside the clip as it is,
        /// held on the last key once it is over.
        /// </summary>
        internal float TicksAt(float seconds)
            => seconds < DurationSeconds ? (seconds * TicksPerSecond) % Duration : Duration;

        private Binding BindingFor(Bone[] bones)
        {
            var bindings = _bindings;
            for (int i = 0; i < bindings.Length; i++)
                if (ReferenceEquals(bindings[i].Bones, bones)) return bindings[i];

            return Bind(bones);
        }

        private RootBinding RootFor(Binding binding, int rootBone)
        {
            var root = binding.Root;
            if (root != null && root.RootBone == rootBone) return root;

            lock (_channels)
            {
                root = binding.Root;
                if (root == null || root.RootBone != rootBone)
                    binding.Root = root = BuildRoot(binding, rootBone);
                return root;
            }
        }

        /// <summary>
        /// Split the hips' travel over the ground from the rest of the clip. The travel is
        /// measured in model space, through whatever the bones above the hips do, and taken out
        /// of the hips' position keys in the space those are in.
        /// </summary>
        private RootBinding BuildRoot(Binding binding, int rootBone)
        {
            var bones = binding.Bones;
            var result = new RootBinding
            {
                RootBone = rootBone,
                Track = RootMotionTrack.Nothing,
                Channels = binding.Channels,
                TurnChannels = binding.Channels,
            };

            var hips = binding.Channels[rootBone];
            var positions = hips?.Position;
            if (positions == null || positions.Count < 2) return result;

            int count = positions.Count;
            float firstTicks = positions[0].Time;
            float span = positions[count - 1].Time - firstTicks;
            if (span <= 0f) return result;

            // Model matrix of the bone above the hips: world(bone) = local(bone) * world(parent)
            Matrix4x4 ParentModel(float ticks, bool bind)
            {
                var model = Matrix4x4.Identity;
                for (int bone = bones[rootBone].Parent; bone >= 0; bone = bones[bone].Parent)
                {
                    var pose = bones[bone].BindPose;
                    if (!bind) binding.Channels[bone]?.GetBoneTransform(bones[bone], ticks, ref pose);
                    model *= Matrix4x4.CreateScale(pose.Scale)
                           * Matrix4x4.CreateFromQuaternion(pose.Rotation)
                           * Matrix4x4.CreateTranslation(pose.Position);
                }
                return model;
            }

            var parents = new Matrix4x4[count];
            var ground = new Vector3[count];
            for (int k = 0; k < count; k++)
            {
                parents[k] = ParentModel(positions[k].Time, bind: false);
                ground[k] = Vector3.Transform(positions[k].Value, parents[k]);
            }

            var net = ground[count - 1] - ground[0];
            net.Y = 0f;

            float surgeSquared = 0f;
            for (int k = 0; k < count; k++)
            {
                var off = ground[k] - ground[0] - net * ((positions[k].Time - firstTicks) / span);
                surgeSquared = MathF.Max(surgeSquared, off.X * off.X + off.Z * off.Z);
            }

            float hipHeight = MathF.Abs(Vector3.Transform(bones[rootBone].BindPose.Position, ParentModel(0f, bind: true)).Y);
            if (hipHeight < 1e-4f) hipHeight = 1f;

            float pathSurge = PathSurge * hipHeight, netTravel = NetTravel * hipHeight;
            var kind = RootMotionKind.None;
            if (net.X * net.X + net.Z * net.Z > netTravel * netTravel)
                kind = surgeSquared > pathSurge * pathSurge ? RootMotionKind.Path : RootMotionKind.Steady;

            // How far the hips turn about the vertical from the first key to the last, summed
            // key to key so that more than half a turn is counted right
            Quaternion ParentRotation(float ticks)
            {
                var rotation = Quaternion.Identity;
                for (int bone = bones[rootBone].Parent; bone >= 0; bone = bones[bone].Parent)
                {
                    var pose = bones[bone].BindPose;
                    binding.Channels[bone]?.GetBoneTransform(bones[bone], ticks, ref pose);
                    rotation = pose.Rotation * rotation;
                }
                return rotation;
            }

            float turn = 0f;
            List<VectorKey> headings = null;
            var rotations = hips.Rotation;
            if (rotations != null && rotations.Count >= 2)
            {
                headings = new List<VectorKey>(rotations.Count) { new VectorKey { Time = rotations[0].Time } };
                var previous = ParentRotation(rotations[0].Time) * rotations[0].Value;
                for (int j = 1; j < rotations.Count; j++)
                {
                    var current = ParentRotation(rotations[j].Time) * rotations[j].Value;
                    var step = current * Quaternion.Inverse(previous);
                    if (step.W < 0f) step = Quaternion.Negate(step);
                    turn += 2f * MathF.Atan2(step.Y, step.W);
                    headings.Add(new VectorKey { Time = rotations[j].Time, Value = new Vector3(turn, 0f, 0f) });
                    previous = current;
                }
                if (MathF.Abs(turn) < TurnThreshold)
                {
                    turn = 0f;
                    headings = null;
                }
            }

            if (kind == RootMotionKind.None && turn == 0f) return result;

            Vector3 TravelAt(int k)
            {
                if (kind == RootMotionKind.Path)
                {
                    var travel = ground[k] - ground[0];
                    travel.Y = 0f;
                    return travel;
                }
                return kind == RootMotionKind.Steady ? net * ((positions[k].Time - firstTicks) / span) : Vector3.Zero;
            }

            var track = new RootMotionTrack
            {
                Kind = kind,
                Net = kind == RootMotionKind.None ? Vector3.Zero : TravelAt(count - 1),
                Turn = turn,
                Heading = headings != null ? new VectorKeys(headings) : null,
                Duration = DurationSeconds,
                StartTicks = firstTicks,
                EndTicks = firstTicks + span,
            };

            // In place: the travel leaves the hips' position keys, in the space those are in
            var inPlace = new List<VectorKey>(count);
            var path = kind == RootMotionKind.Path ? new List<VectorKey>(count) : null;
            var toParent = new Matrix4x4[count];
            for (int k = 0; k < count; k++)
            {
                if (!Matrix4x4.Invert(parents[k], out toParent[k])) return result;

                var travel = TravelAt(k);
                path?.Add(new VectorKey { Time = positions[k].Time, Value = travel });
                inPlace.Add(new VectorKey { Time = positions[k].Time, Value = positions[k].Value - Vector3.TransformNormal(travel, toParent[k]) });
            }
            if (path != null) track.Path = new VectorKeys(path);

            AnimationChannel[] WithHips(VectorKeys position, QuaternionKeys rotation)
            {
                var channels = (AnimationChannel[])binding.Channels.Clone();
                channels[rootBone] = new AnimationChannel { Target = hips.Target, Position = position, Rotation = rotation, Scale = hips.Scale };
                return channels;
            }

            result.Track = track;
            result.Channels = kind == RootMotionKind.None ? binding.Channels : WithHips(new VectorKeys(inPlace), hips.Rotation);
            result.TurnChannels = result.Channels;

            if (turn != 0f)
            {
                // Facing ahead as well: the character has turned by TurnAt so far (as far as the
                // hips have), so the pose is turned back by that much about where the character
                // stands and keeps the heading of its first key
                var facing = new List<VectorKey>(count);
                for (int k = 0; k < count; k++)
                {
                    var back = Quaternion.CreateFromAxisAngle(Vector3.UnitY, -track.TurnAt(positions[k].Time));
                    var model = Vector3.Transform(ground[k] - TravelAt(k), back);
                    facing.Add(new VectorKey { Time = positions[k].Time, Value = Vector3.Transform(model, toParent[k]) });
                }

                var heading = new List<QuaternionKey>(rotations.Count);
                for (int j = 0; j < rotations.Count; j++)
                {
                    float ticks = rotations[j].Time;
                    var parent = ParentRotation(ticks);
                    var back = Quaternion.CreateFromAxisAngle(Vector3.UnitY, -track.TurnAt(ticks));
                    heading.Add(new QuaternionKey
                    {
                        Time = ticks,
                        Value = Quaternion.Normalize(Quaternion.Inverse(parent) * back * parent * rotations[j].Value),
                    });
                }

                result.TurnChannels = WithHips(new VectorKeys(facing), new QuaternionKeys(heading));
            }

            return result;
        }

        private Binding Bind(Bone[] bones)
        {
            lock (_channels)
            {
                var bindings = _bindings;
                for (int i = 0; i < bindings.Length; i++)
                    if (ReferenceEquals(bindings[i].Bones, bones)) return bindings[i];

                var byName = new Dictionary<string, AnimationChannel>(_channels.Count);
                for (int i = 0; i < _channels.Count; i++)
                    byName[_channels[i].Target] = _channels[i];

                var channels = new AnimationChannel[bones.Length];
                for (int i = 0; i < bones.Length; i++)
                {
                    string name = bones[i].Name;
                    if (name != null && byName.TryGetValue(name, out var channel))
                        channels[i] = channel;
                }

                int kept = Math.Min(bindings.Length, MaxBindings - 1);
                var grown = new Binding[kept + 1];
                grown[0] = new Binding { Bones = bones, Channels = channels };
                Array.Copy(bindings, 0, grown, 1, kept);
                _bindings = grown;

                return grown[0];
            }
        }

        /// <summary>
        /// The pose this clip gives one bone of a skeleton at a time in seconds. Leaves
        /// <paramref name="result"/> as it is where the clip does not move the bone.
        /// </summary>
        public void GetBonePose(Bone[] bones, int boneIndex, float time, ref BonePose result)
        {
            ChannelsFor(bones)[boneIndex]?.GetBoneTransform(bones[boneIndex], TicksAt(time), ref result);
        }
    }

    /// <summary>
    /// Something that happens at a point of a clip: the blow of a swing lands, a hand lets go.
    /// The Animator reports it in the update in which playback passes that point
    /// (<c>Animator.GetEvents</c>).
    /// </summary>
    public class AnimationEvent
    {
        /// <summary>
        /// The name a foot coming down is reported under. Those are not marked on clips: they
        /// are found from what the clip does with the feet (<c>Animator.FootfallEvents</c>).
        /// </summary>
        public const string Footfall = "footfall";

        /// <summary>When, as a fraction of the clip's length.</summary>
        public float Time;

        public string Name;

        /// <summary>
        /// Where on the body it happens, when that matters. A clip played left for right
        /// reports the bone of the other side.
        /// </summary>
        public HumanoidBone? Bone;
    }

    /// <summary>An <see cref="AnimationEvent"/> as it was passed on one Animator.</summary>
    public struct AnimationEventInfo
    {
        public string Name;

        /// <summary>Where on the body, the way round the clip was played; null when the event names none.</summary>
        public HumanoidBone? Bone;

        /// <summary>
        /// How much of what the character shows there comes from the state that passed it, 0..1:
        /// the weight of its layer and the layer's mask at <see cref="Bone"/>, less what the
        /// layers above cover. A step of the legs under an upper-body attack counts in full; the
        /// steps in the attack's own clip, whose legs are masked out, count for nothing.
        /// </summary>
        public float Weight;

        /// <summary>A footfall: how fast the foot came down (<see cref="Footfall.Strength"/>). Otherwise 1.</summary>
        public float Strength;

        /// <summary>The layer, state and clip it comes from.</summary>
        public int Layer;
        public AnimationState State;
        public AnimationClip Clip;
    }
}
