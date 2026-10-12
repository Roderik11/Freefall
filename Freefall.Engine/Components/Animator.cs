using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Animation;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public delegate void BoneTransformHandler(Bone bone, ref Matrix4x4 matrix);
    public delegate void PostHierarchyHandler(Bone[] skeleton, Matrix4x4[] bones);

    /// <summary>
    /// Controls animation playback for skinned meshes using an animation state machine.
    /// Owns per-Skeleton bone buffers (StreamingBuffer) that are shared across all
    /// SkinnedMeshRenderers on this entity. Each unique Skeleton gets posed once per frame.
    /// </summary>
    [Icon("icon_animator.png")]
    public class Animator : Component, IUpdate, IParallel
    {
        /// <summary>The animation state machine (shared definition).</summary>
        public Animation.Animation Animation;

        /// <summary>Per-instance mutable state (time, weights, parameters).</summary>
        public readonly AnimationPlayback Playback = new();

        private Skeleton _retargetSource;

        /// <summary>
        /// The source skeleton that the animation clips were made for.
        /// Set this when the clips come from a different character than the mesh.
        /// </summary>
        public Skeleton RetargetSource
        {
            get => _retargetSource;
            set
            {
                _retargetSource = value;
                _retargetInitialized = false;
            }
        }

        private FingerRetargeting _fingers = FingerRetargeting.KeepShape;

        /// <summary>
        /// How the fingers follow the clips of a <see cref="RetargetSource"/> rig: keeping the
        /// character's own hand shape and bending from there, or pointing where the source's
        /// fingers point. See <see cref="FingerRetargeting"/>.
        /// </summary>
        public FingerRetargeting Fingers
        {
            get => _fingers;
            set
            {
                if (_fingers == value) return;
                _fingers = value;
                _retargetInitialized = false;
            }
        }

        /// <summary>
        /// How much the fingers of a retargeted character follow the clips: 1 = in full, 0 = the
        /// hands stay as the character is bound, which on most rigs is a relaxed open hand.
        /// Clips made for a hand that grips look clenched on a hand that holds nothing.
        /// </summary>
        public float FingerWeight { get; set; } = 1f;

        // FingerWeight × what the graph or its animation set asks for, for the pose being built
        private float _poseFingerWeight = 1f;

        /// <summary>
        /// Replaces the clips of <see cref="Animation"/> state by state (weapon type, character
        /// style) without another graph. Changing it while playing cross-fades to the new clips
        /// at the same point of each cycle. Ignored when the set was made for another graph.
        /// </summary>
        public AnimationSet AnimationSet { get; set; }

        private const float AnimationSetFadeSeconds = 0.2f;
        private bool _animationSetApplied;

        /// <summary>Event fired when a bone transform is calculated.</summary>
        public event BoneTransformHandler OnBoneTransform;

        /// <summary>Event fired after hierarchy multiplication — bones are in model space.</summary>
        public event PostHierarchyHandler OnPostHierarchy;

        /// <summary>
        /// Report each foot coming down as an event named <see cref="AnimationEvent.Footfall"/>,
        /// with the foot as its bone and how hard it lands as its strength. Nobody marks these
        /// on clips: they are found from what each clip does with the feet
        /// (<see cref="AnimationClip.GetFootfalls"/>), the first time the clip is played. Set
        /// by whoever wants them (footstep sounds, dust).
        /// </summary>
        [Freefall.Reflection.DontSerialize]
        public bool FootfallEvents { get; set; }

        // What the last update passed. Filled in that update, which runs in parallel with other
        // Animators', and read afterwards by whoever asks: nothing is called from in there.
        private readonly List<AnimationEventInfo> _events = new();

        // The layer whose state is being updated
        internal AnimationLayer EventLayer;

        // Retargeting
        private BoneMap _boneMap;
        private bool _retargetInitialized;
        private bool _debugLogged;
        private Quaternion[] _srcAnimModelRot;
        private BonePose[] _srcPose; // the layers blended on the source skeleton, per source bone
        private Quaternion[] _retargetedLocalRot;
        private Quaternion[] _retargetedModelRot;

        // Un-retargeted pose: model-space rotations while it is blended (see BlendLayers)
        private Bone[] _poseBones;
        private Quaternion[] _poseModelRot;
        private bool _trackPoseModelRot;
        private bool _sourceEvaluated;

        // Parameter name → index lookup (built once when Animation is set)
        private Dictionary<string, AnimationParameter> _paramMap;

        // Per-Skeleton bone buffers: each unique Skeleton gets its own GPU buffer.
        // Multiple SMRs sharing the same Skeleton share the same buffer (no redundant posing).
        private readonly Dictionary<Skeleton, BoneBufferEntry> _boneBuffers = new();
        
        private class BoneBufferEntry
        {
            public StreamingBuffer<Matrix4x4> Buffer;
            public Matrix4x4[] StagingMatrices;
        }

        protected override void Awake()
        {
        }

        private void EnsureParamMap()
        {
            if (_paramMap != null || Animation == null) return;

            _paramMap = new Dictionary<string, AnimationParameter>(Animation.Parameters.Count);
            foreach (var p in Animation.Parameters)
                _paramMap[p.Name] = p;
        }

        // --- Parameter API ---

        public float GetParam(string name)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param))
                return Playback.Get(PK.Param(param.Index), param.DefaultValue);
            return 0f;
        }

        public void SetParam(string name, float value)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param))
                Playback.Set(PK.Param(param.Index), value);
        }

        internal void ConsumeTrigger(string name)
        {
            EnsureParamMap();
            if (_paramMap != null && _paramMap.TryGetValue(name, out var param) && param.IsTrigger)
                Playback.Set(PK.Param(param.Index), 0);
        }

        // --- Events ---

        /// <summary>
        /// The events the playing states passed in the last update: those marked on their clips
        /// (<see cref="AnimationClip.Events"/>) and, with <see cref="FootfallEvents"/>, the feet
        /// coming down. Read it once per frame from an Update of your own; each event is in it
        /// for one frame. There is no callback on purpose: Animators update in parallel, and
        /// code called from in there could touch nothing safely.
        /// <code>
        /// foreach (ref readonly var e in animator.GetEvents())
        ///     if (e.Name == "hit" &amp;&amp; e.Weight > 0.5f) ...
        /// </code>
        /// </summary>
        public ReadOnlySpan<AnimationEventInfo> GetEvents()
            => System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_events);

        /// <summary>The skeleton the clips are played on: the source skeleton when retargeting.</summary>
        internal bool TryGetClipSkeleton(out Skeleton skeleton)
        {
            skeleton = _boneMap != null ? _boneMap.Source : _poseRootSkeleton;
            return skeleton != null;
        }

        /// <summary>A state passed an event of its clip in this update.</summary>
        internal void ReportEvent(AnimationState state, AnimationClip clip, string name, HumanoidBone? bone, float strength)
        {
            var layer = EventLayer;
            _events.Add(new AnimationEventInfo
            {
                Name = name,
                Bone = bone,
                Strength = strength,
                Weight = layer != null ? ShareOf(layer, bone) : 1f,
                Layer = layer?.Index ?? 0,
                State = state,
                Clip = clip,
            });
        }

        /// <summary>
        /// How much of what the character shows at a bone comes from one layer: its weight and
        /// its mask there, less what the layers above put over it. Without a bone only layers
        /// that take the whole body count as being over it.
        /// </summary>
        private float ShareOf(AnimationLayer layer, HumanoidBone? bone)
        {
            Skeleton skeleton = null;
            int index = -1;
            if (bone.HasValue && TryGetClipSkeleton(out skeleton))
            {
                var humanoid = s_humanoid.GetValue(skeleton, static s => HumanoidTemplate.Detect(s) ?? new Dictionary<HumanoidBone, int>());
                if (!humanoid.TryGetValue(bone.Value, out index)) index = -1;
            }

            float share = layer.GetWeight(this);
            if (layer.Mask != null && index >= 0)
                share *= layer.Mask.GetWeights(skeleton).Weight[index];

            var layers = Animation.Layers;
            for (int i = layer.Index + 1; i < layers.Count && share > 0f; i++)
            {
                var above = layers[i];
                float over = above.GetWeight(this);
                if (over <= 0f) continue;

                // A state that plays nothing lets the layers below through
                var current = above.GetCurrent(Playback);
                if (current == null || !current.Plays(Playback)) continue;

                if (above.Mask != null)
                    over *= index >= 0 ? above.Mask.GetWeights(skeleton).Weight[index] : 0f;
                share *= 1f - over;
            }

            return share;
        }

        // --- State queries ---

        /// <summary>
        /// The name of the state a layer is in (the one a running transition leads to), or null.
        /// For game code that has to know what the character is doing: no movement input while
        /// "Roll" plays, the next hit of a combo only while "Attack1" is under way.
        /// </summary>
        public string GetStateName(int layer = 0)
        {
            var layers = Animation?.Layers;
            if (layers == null || (uint)layer >= (uint)layers.Count) return null;
            return layers[layer].GetCurrent(Playback)?.Name;
        }

        /// <summary>How far the state of a layer is through its clip, 0 to 1 (0 when it plays none).</summary>
        public float GetStateTime(int layer = 0)
        {
            var layers = Animation?.Layers;
            if (layers == null || (uint)layer >= (uint)layers.Count) return 0f;
            return layers[layer].GetCurrent(Playback)?.GetTimeNormalized(Playback) ?? 0f;
        }

        /// <summary>
        /// Where the state of a layer stands in its cycle, 0 to 1 counted from where its clip is
        /// taken up (a blend tree: its heaviest child); negative when it plays nothing. The same
        /// for every clip of a tree, whichever foot each of them begins on.
        /// </summary>
        public float GetStatePhase(int layer = 0)
        {
            var layers = Animation?.Layers;
            if (layers == null || (uint)layer >= (uint)layers.Count) return -1f;
            return layers[layer].GetCurrent(Playback)?.GetPhase(Playback) ?? -1f;
        }

        /// <summary>Whether the state of a layer plays left for right at the moment (<see cref="AnimationState.Mirror"/>).</summary>
        public bool IsStateMirrored(int layer = 0)
        {
            var layers = Animation?.Layers;
            if (layers == null || (uint)layer >= (uint)layers.Count) return false;
            return layers[layer].GetCurrent(Playback)?.IsMirrored(Playback) ?? false;
        }

        // --- Update ---

        public void Update()
        {
            _events.Clear();

            if (Entity == null) return;
            if (Animation == null) return;

            if (!_retargetInitialized)
                InitRetargeting();

            UpdateAnimationSet();
            UpdateLayers();

            _sourceEvaluated = false;
            WalkChildren(Entity.Transform);
        }

        // --- Root motion ---

        /// <summary>
        /// The travel over the ground that the playing states with
        /// <see cref="AnimationState.RootMotion"/> asked for in the last update, in this entity's
        /// local space: +Z forward, in metres on a character of scale 1. Zero for states without
        /// the flag, which play in place.
        ///
        /// It is not applied here. Whoever moves the character adds it to its own movement, in
        /// full, scaled, or with more on top (a knock-back that carries further than the clip):
        /// <c>move = velocity * dt + Transform.Rotation * animator.RootMotionDelta</c>.
        /// </summary>
        public Vector3 RootMotionDelta { get; private set; }

        /// <summary>
        /// The turn about the vertical that the playing states with
        /// <see cref="AnimationState.RootTurn"/> asked for in the last update, in radians, positive
        /// like a positive yaw. Applied by whoever turns the character:
        /// <c>Transform.Rotation *= Quaternion.CreateFromAxisAngle(Vector3.UnitY, animator.RootTurnDelta)</c>.
        /// <see cref="RootMotionDelta"/> of the same update is in the heading before this turn.
        /// </summary>
        public float RootTurnDelta { get; private set; }

        /// <summary>
        /// Turns the upper body away from the way the entity faces, on top of whatever the graph
        /// plays: X to the right and Y upwards, in radians, spread over the spine. For a body that
        /// points its weapon where the player looks while its feet stay where they are. Whoever
        /// drives the character sets it every frame (and eases it: it is applied as it is).
        /// </summary>
        [Freefall.Reflection.DontSerialize]
        public Vector2 TorsoAim { get; set; }

        /// <summary>The same for the neck and head, on top of what <see cref="TorsoAim"/> gives them.</summary>
        [Freefall.Reflection.DontSerialize]
        public Vector2 HeadLook { get; set; }

        /// <summary>
        /// The share of <see cref="TorsoAim"/>'s turn to the side that the hips take, 0..1: with
        /// the feet where they are a body turns in the hips as well, not at the waist alone. The
        /// legs keep their place and the spine turns by that much less, so the shoulders point
        /// where they are asked to either way. See <see cref="AnimationLayer.HipsFollow"/> for the
        /// same on a masked layer.
        /// </summary>
        [Freefall.Reflection.DontSerialize]
        public float TorsoAimHips { get; set; }

        /// <summary>The most the hips are turned against the legs by aim and layers together, in radians.</summary>
        private const float MaxHipsTurn = 0.5f;

        // What the masked layers of the pose built last ask the hips to come round by (HipsFollow)
        private float _layerHipsTurn;

        // Per bone of a skeleton, its share of TorsoAim and of HeadLook (0 for all but the spine,
        // the neck and the head), and the bones a turn of the hips involves
        private sealed class AimChain
        {
            public float[] Torso, Head;
            public int Hips = -1, LeftLeg = -1, RightLeg = -1;
        }

        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Skeleton, AimChain> s_aimChains = new();

        private static AimChain AimChainOf(Skeleton skeleton) => s_aimChains.GetValue(skeleton, static s =>
        {
            int count = s.Bones.Length;
            var chain = new AimChain { Torso = new float[count], Head = new float[count] };
            var humanoid = HumanoidTemplate.Detect(s);
            if (humanoid == null) return chain;

            // Each bone of a chain turns by an equal part: the turn adds up along it
            static void Share(Dictionary<HumanoidBone, int> humanoid, float[] shares, params HumanoidBone[] bones)
            {
                int found = 0;
                foreach (var bone in bones)
                    if (humanoid.ContainsKey(bone)) found++;
                foreach (var bone in bones)
                    if (humanoid.TryGetValue(bone, out int index)) shares[index] = 1f / found;
            }

            Share(humanoid, chain.Torso, HumanoidBone.Spine, HumanoidBone.Chest, HumanoidBone.UpperChest);
            Share(humanoid, chain.Head, HumanoidBone.Neck, HumanoidBone.Head);

            // The hips can only take a turn where the legs can give it back
            if (humanoid.TryGetValue(HumanoidBone.Hips, out int hips)
                && humanoid.TryGetValue(HumanoidBone.LeftUpperLeg, out int leftLeg)
                && humanoid.TryGetValue(HumanoidBone.RightUpperLeg, out int rightLeg))
            {
                chain.Hips = hips;
                chain.LeftLeg = leftLeg;
                chain.RightLeg = rightLeg;
            }
            return chain;
        });

        /// <summary>
        /// Turn a bone's model matrix about the bone's own place: sideways about the vertical,
        /// then up or down about the axis that lies across the way the body faces by then.
        /// </summary>
        private static void TurnInModelSpace(ref Matrix4x4 model, float yaw, float pitch, float yawSoFar, bool flipXZ)
        {
            var place = model.Translation;
            model.Translation = Vector3.Zero;

            var turn = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yaw);
            if (pitch != 0f)
            {
                // The entity's right is -X in the model space of a skeleton whose mesh is turned round
                var across = Vector3.Transform(flipXZ ? -Vector3.UnitX : Vector3.UnitX,
                    Quaternion.CreateFromAxisAngle(Vector3.UnitY, yawSoFar + yaw));
                turn = Quaternion.CreateFromAxisAngle(across, -pitch) * turn;
            }

            model *= Matrix4x4.CreateFromQuaternion(turn);
            model.Translation = place;
        }

        // ---- Where the bones are: for what hangs on the character ----

        // Model matrices of the character's skeleton as last posed; kept once someone has asked
        private bool _keepBonePoses;
        private Matrix4x4[] _bonePoses;
        private Skeleton _bonePoseSkeleton;

        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Skeleton, Dictionary<HumanoidBone, int>> s_humanoid = new();

        /// <summary>
        /// Where a humanoid bone of the character was when it was last posed, in the entity's own
        /// space and without scale: for hanging something on it (a sword in the right hand, a
        /// shield on the left forearm). Put the thing under the character's entity and give it
        /// this as its local transform, with its own offset in front:
        /// <c>local = grip * pose</c>. False until the character has been posed once after the
        /// first call, and for a bone the skeleton does not have.
        /// </summary>
        public bool TryGetBonePose(HumanoidBone bone, out Matrix4x4 pose)
        {
            pose = Matrix4x4.Identity;
            _keepBonePoses = true;

            var skeleton = _bonePoseSkeleton;
            var poses = _bonePoses;
            if (skeleton == null || poses == null) return false;

            var humanoid = s_humanoid.GetValue(skeleton, static s => HumanoidTemplate.Detect(s) ?? new Dictionary<HumanoidBone, int>());
            if (!humanoid.TryGetValue(bone, out int index) || (uint)index >= (uint)poses.Length) return false;

            // The bone's own axes stay as the rig has them (what a grip is measured against); only
            // the model space is turned round to the entity's for a skeleton whose mesh is flipped.
            // (Skinning conjugates instead, F·M·F, because its vertices are flipped as well: that
            // would hand out a frame with X and Z the other way round.)
            var model = poses[index];
            if (skeleton.FlipXZ) model *= Matrix4x4.CreateScale(-1f, 1f, -1f);
            if (!Matrix4x4.Decompose(model, out _, out var rotation, out var position)) return false;

            pose = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position);
            return true;
        }

        // In the model space of the skeleton root motion is taken from
        private Vector3 _frameRootMotion; // blended over the layers updated so far this frame
        private Vector3 _layerRootMotion; // what the layer being updated asked for
        private float _frameRootTurn, _layerRootTurn;
        private bool _layerRootPlays;

        // Un-retargeted: the first humanoid skeleton this Animator posed, and its hips
        private Skeleton _poseRootSkeleton;
        private int _poseRootBone = -1;

        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Skeleton, System.Runtime.CompilerServices.StrongBox<int>> s_hips = new();

        /// <summary>The hips of a humanoid skeleton, or -1.</summary>
        private static int HipsOf(Skeleton skeleton) => s_hips.GetValue(skeleton, static s =>
        {
            var humanoid = HumanoidTemplate.Detect(s);
            return new System.Runtime.CompilerServices.StrongBox<int>(
                humanoid != null && humanoid.TryGetValue(HumanoidBone.Hips, out int hips) ? hips : -1);
        }).Value;

        /// <summary>
        /// The skeleton the clips are played on and its hips: the source skeleton when retargeting.
        /// False for a skeleton without hips, which has no root motion.
        /// </summary>
        internal bool TryGetRootSource(out Bone[] bones, out int rootBone)
        {
            var skeleton = _boneMap != null ? _boneMap.Source : _poseRootSkeleton;
            bones = skeleton?.Bones;
            rootBone = _boneMap != null ? _boneMap.SourceRoot : _poseRootBone;
            return bones != null && rootBone >= 0;
        }

        /// <summary>A state reports the travel and turn of its clip for this update (zero when it plays in place).</summary>
        internal void AddRootMotion(Vector3 travel, float turned = 0f)
        {
            _layerRootMotion += travel;
            _layerRootTurn += turned;
            _layerRootPlays = true;
        }

        internal void BeginLayerRootMotion()
        {
            _layerRootMotion = Vector3.Zero;
            _layerRootTurn = 0f;
            _layerRootPlays = false;
        }

        /// <summary>
        /// A layer's travel goes over that of the layers below the way its pose does at the hips:
        /// by layer weight × mask weight. An upper-body layer has no say, a full-body one takes over.
        /// </summary>
        internal void EndLayerRootMotion(AnimationLayer layer)
        {
            // A state that plays nothing lets the layers below through
            if (!_layerRootPlays) return;

            float share = layer.GetWeight(this);
            if (share > 0f && layer.Mask != null)
            {
                var skeleton = _boneMap != null ? _boneMap.Source : _poseRootSkeleton;
                int rootBone = _boneMap != null ? _boneMap.SourceRoot : _poseRootBone;
                share *= skeleton != null && rootBone >= 0 ? layer.Mask.GetWeights(skeleton).Weight[rootBone] : 0f;
            }
            if (share <= 0f) return;

            if (share >= 1f)
            {
                _frameRootMotion = _layerRootMotion;
                _frameRootTurn = _layerRootTurn;
            }
            else
            {
                _frameRootMotion = Vector3.Lerp(_frameRootMotion, _layerRootMotion, share);
                _frameRootTurn += (_layerRootTurn - _frameRootTurn) * share;
            }
        }

        private void UpdateLayers()
        {
            _frameRootMotion = Vector3.Zero;
            _frameRootTurn = 0f;

            foreach (AnimationLayer layer in Animation.Layers)
                layer.Update(this, Playback);

            // Into the entity's space: at the character's size, and turned the way its meshes are
            var travel = _frameRootMotion;
            bool flip;
            if (_boneMap != null)
            {
                travel *= _boneMap.PositionScale;
                flip = _boneMap.Target.FlipXZ;
            }
            else
                flip = _poseRootSkeleton != null && _poseRootSkeleton.FlipXZ;

            RootMotionDelta = flip ? new Vector3(-travel.X, travel.Y, -travel.Z) : travel;

            // Turning the meshes round by half a turn does not change which way a turn goes
            RootTurnDelta = _frameRootTurn;
        }

        /// <summary>
        /// Hand the animation set to the playback state. A change while playing keeps every state
        /// at the same point of its cycle under the new clips and starts the cross-fade.
        /// </summary>
        private void UpdateAnimationSet()
        {
            var set = AnimationSet;
            if (set != null && set.Base != null && set.Base != Animation)
                set = null;

            if (!_animationSetApplied)
            {
                Playback.AnimationSet = set;
                _animationSetApplied = true;
            }
            else if (set != Playback.AnimationSet)
            {
                foreach (AnimationLayer layer in Animation.Layers)
                    foreach (AnimationState state in layer.States)
                        RescaleStateTime(state, Playback.AnimationSet, set);

                Playback.PreviousAnimationSet = Playback.AnimationSet;
                Playback.AnimationSet = set;
                Playback.SetFade = 0f;
            }
            else if (Playback.SetFade < 1f)
            {
                Playback.SetFade = MathF.Min(1f, Playback.SetFade + Time.SmoothDelta / AnimationSetFadeSeconds);
                if (Playback.SetFade >= 1f)
                    Playback.PreviousAnimationSet = null;
            }
        }

        private void RescaleStateTime(AnimationState state, AnimationSet from, AnimationSet to)
        {
            var oldClip = from != null ? from.Resolve(state) : state.Clip;
            var newClip = to != null ? to.Resolve(state) : state.Clip;
            if (oldClip != null && newClip != null && oldClip != newClip)
            {
                // The same point of the new clip, each counted from where its set takes it up
                float elapsed = state.GetTimeElapsed(Playback);
                if (elapsed > 0f)
                {
                    float point = AnimationState.Aligned(elapsed * oldClip.DurationSecondsInverse,
                        from != null ? from.ResolveStart(state) : state.Start,
                        to != null ? to.ResolveStart(state) : state.Start, state.Loop);
                    state.SetTimeElapsed(Playback, point * newClip.DurationSeconds);
                }
            }

            int count = state.GetStateCount();
            for (int i = 0; i < count; i++)
                RescaleStateTime(state.GetState(i), from, to);
        }

        // Walk child SMRs: pose each unique Skeleton once, set BoneBufferIdx
        void WalkChildren(Transform parent)
        {
            var smr = parent?.Entity?.GetComponent<SkinnedMeshRenderer>();
            if (smr?.Mesh?.Skeleton != null)
            {
                var skeleton = smr.Mesh.Skeleton;

                var entry = EnsureBoneBuffer(skeleton);

                // Pose + upload (only once per unique Skeleton per frame)
                if (entry.Buffer.LastWriteFrame != Engine.FrameIndex)
                {
                    GetPose(skeleton, entry.StagingMatrices);
                    entry.Buffer.BulkWrite(entry.StagingMatrices);
                    entry.Buffer.LastWriteFrame = Engine.FrameIndex;
                }

                // Set BoneBufferIdx on SMR so it passes it through to Enqueue
                smr.BoneBufferIdx = entry.Buffer.SrvIndex;
            }

            foreach (Transform child in parent)
                WalkChildren(child);
        }

        /// <summary>
        /// The bone buffer indices this character's renderers draw with in the current frame: what
        /// a shader can tell its skin by (InstanceDescriptor.BoneBufferIdx). One per skeleton among
        /// its parts, the first four; 0 = none.
        /// </summary>
        internal Vortice.Mathematics.Int4 BoneBufferIndices()
        {
            Span<int> found = stackalloc int[4];
            int count = 0;
            foreach (var entry in _boneBuffers.Values)
            {
                if (count == found.Length) break;
                found[count++] = (int)entry.Buffer.SrvIndex;
            }
            return new Vortice.Mathematics.Int4(found[0], found[1], found[2], found[3]);
        }

        /// <summary>
        /// Get or create a bone buffer for a Skeleton. Lazily allocated.
        /// </summary>
        private BoneBufferEntry EnsureBoneBuffer(Skeleton skeleton)
        {
            if (_boneBuffers.TryGetValue(skeleton, out var existing))
                return existing;

            int boneCount = skeleton.Bones.Length;
            var buffer = new StreamingBuffer<Matrix4x4>(Engine.Device, boneCount);
            var entry = new BoneBufferEntry
            {
                Buffer = buffer,
                StagingMatrices = new Matrix4x4[boneCount]
            };
            _boneBuffers[skeleton] = entry;
            return entry;
        }

        public override void Destroy()
        {
            // The frames in flight still read these through the bone buffer index of this character's
            // draws. Releasing them here removed the device when many characters were deleted at once,
            // so they go to the device to release once those frames are done.
            foreach (var entry in _boneBuffers.Values)
                Engine.Device.DeferDispose(entry.Buffer);
            _boneBuffers.Clear();
        }

        // --- Retargeting ---

        private void InitRetargeting()
        {
            _retargetInitialized = true;
            _debugLogged = false;
            _boneMap = null;

            if (RetargetSource == null) return;

            var renderer = Entity?.GetComponentInChildren<SkinnedMeshRenderer>();
            var meshSkeleton = renderer?.Mesh?.Skeleton;
            if (meshSkeleton == null) return;
            if (RetargetSource == meshSkeleton) return;

            // Try humanoid template first, fall back to name matching
            _boneMap = HumanoidTemplate.TryMap(RetargetSource, meshSkeleton, _fingers)
                    ?? BoneMap.CreateFromNames(RetargetSource, meshSkeleton);

            Debug.Log($"[Animator] Retarget: {RetargetSource.Name} → {meshSkeleton.Name}, BoneMap created");

            // One-time: compare model-space positions of key bones
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[BonePos] {RetargetSource.Name} → {meshSkeleton.Name}");
            for (int i = 0; i < meshSkeleton.Bones.Length; i++)
            {
                int srcIdx = _boneMap.TargetToSource[i];
                if (srcIdx < 0) continue;

                Vector3 srcPos = Vector3.Zero, dstPos = Vector3.Zero;
                if (Matrix4x4.Invert(RetargetSource.Bones[srcIdx].OffsetMatrix, out var srcInv))
                    srcPos = srcInv.Translation;
                if (Matrix4x4.Invert(meshSkeleton.Bones[i].OffsetMatrix, out var dstInv))
                    dstPos = dstInv.Translation;

                sb.AppendLine($"  dst[{i}] {meshSkeleton.BoneNames[i],-22} src=({srcPos.X,7:F2},{srcPos.Y,7:F2},{srcPos.Z,7:F2})  dst=({dstPos.X,7:F2},{dstPos.Y,7:F2},{dstPos.Z,7:F2})");
            }
            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Evaluate the source skeleton's animation to get full model-space rotations.
        /// These are accumulated from the root through all bones (including structural nodes).
        /// </summary>
        private void EvaluateSourceModelRotations()
        {
            var source = _boneMap.Source;
            int count = source.Bones.Length;

            if (_srcAnimModelRot == null || _srcAnimModelRot.Length != count)
            {
                _srcAnimModelRot = new Quaternion[count];
                _srcPose = new BonePose[count];
            }

            _sourcePoseState.RootBone = _boneMap.SourceRoot;
            PrepareLayers(_sourcePoseState, source);

            for (int j = 0; j < count; j++)
            {
                var srcBone = source.Bones[j];
                BonePose blendPose = srcBone.BindPose;

                BlendLayers(source.Bones, j, _srcAnimModelRot, ref blendPose);
                _srcPose[j] = blendPose;

                int parent = srcBone.Parent;
                _srcAnimModelRot[j] = parent >= 0
                    ? _srcAnimModelRot[parent] * blendPose.Rotation
                    : blendPose.Rotation;
            }

            FinishLayers();
        }

        /// <summary>
        /// What one layer contributes to a pose, and what it showed last time. The first part is
        /// read once per pose instead of per bone; the rest carries a transition across frames.
        /// </summary>
        private struct LayerPose
        {
            /// <summary>Layer weight on this Animator; at 0 the layer is skipped.</summary>
            public float Weight;

            /// <summary>Mask weight per bone of the skeleton being posed; null = every bone in full.</summary>
            public float[] MaskWeights;

            /// <summary>Masked bones and their ancestors.</summary>
            public bool[] MaskChain;

            /// <summary>The layer keeps its own model-space orientation (masked, ParentInfluence below 1).</summary>
            public bool ModelSpace;
            public float ParentInfluence;

            /// <summary>The share of the layer's turn the hips take (AnimationLayer.HipsFollow), for ModelSpace.</summary>
            public float HipsFollow;

            /// <summary>The state the layer plays. A transition changes it at once; the pose follows over time.</summary>
            public AnimationState State;

            /// <summary>
            /// The clips of <see cref="State"/> for this pose, and while capturing those of
            /// <see cref="Previous"/>: resolved once, so a bone only reads keys.
            /// </summary>
            public StateSamples Samples, PreviousSamples;

            /// <summary>Progress of the running transition, 1 when settled.</summary>
            public float Fade;

            /// <summary>A transition started since the last pose: this pose records what is carried over.</summary>
            public bool Capturing;
            public int Serial;

            /// <summary>While capturing: the state the transition left, and how far the one it interrupted had come.</summary>
            public AnimationState Previous;
            public float PreviousFade;

            /// <summary>The layer's own model-space rotations along <see cref="MaskChain"/>, for ModelSpace.</summary>
            public Quaternion[] ModelRot;

            /// <summary>The pose the layer showed per bone, last time it was posed.</summary>
            public BonePose[] Last;
            public bool HasLast;

            /// <summary>
            /// Per bone, what the layer showed when the transition started relative to its new state:
            /// rotation as a factor after the state's own, position and scale as differences.
            /// Valid once a pose has captured it for the running transition.
            /// </summary>
            public BonePose[] Offset;
            public bool HasOffset;

            /// <summary>How much of the layer was playing last time, and when the transition started.</summary>
            public float Coverage, FromCoverage;
        }

        /// <summary>
        /// The layers of one pose. A pose is built per skeleton: the source skeleton when
        /// retargeting, otherwise each skeleton this Animator drives.
        /// </summary>
        private sealed class PoseState
        {
            public LayerPose[] Layers = Array.Empty<LayerPose>();

            /// <summary>The hips of the skeleton, -1 when it has none: clips are posed without their travel there.</summary>
            public int RootBone = -1;
        }

        private PoseState _pose; // the one being built
        private readonly PoseState _sourcePoseState = new PoseState();
        private readonly Dictionary<Skeleton, PoseState> _poseStates = new();

        /// <summary>
        /// Read layer weights, masks and the current state for a pose on <paramref name="skeleton"/>.
        /// True when a layer needs the model-space rotations of the pose as it is built.
        /// </summary>
        private bool PrepareLayers(PoseState state, Skeleton skeleton)
        {
            _pose = state;
            _layerHipsTurn = 0f;

            var layers = Animation.Layers;
            if (state.Layers.Length != layers.Count)
                state.Layers = new LayerPose[layers.Count];

            bool anyModelSpace = false;
            var bones = skeleton.Bones;
            int boneCount = bones.Length;

            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                ref var pose = ref state.Layers[i];

                pose.Samples ??= new StateSamples();
                pose.PreviousSamples ??= new StateSamples();
                pose.Samples.Clear();
                pose.PreviousSamples.Clear();
                pose.Samples.RootBone = state.RootBone;
                pose.PreviousSamples.RootBone = state.RootBone;
                pose.Samples.Skeleton = skeleton;
                pose.PreviousSamples.Skeleton = skeleton;

                pose.Weight = layer.GetWeight(this);

                var mask = pose.Weight > 0f && layer.Mask != null ? layer.Mask.GetWeights(skeleton) : null;
                pose.MaskWeights = mask?.Weight;
                pose.MaskChain = mask?.Chain;
                pose.ParentInfluence = Math.Clamp(layer.ParentInfluence, 0f, 1f);
                pose.ModelSpace = mask != null && pose.ParentInfluence < 1f;
                pose.HipsFollow = pose.ModelSpace ? Math.Clamp(layer.HipsFollow, 0f, 1f) : 0f;

                pose.State = layer.GetCurrent(Playback);
                pose.Fade = layer.GetFade(Playback);

                if (pose.Last == null || pose.Last.Length != boneCount)
                {
                    pose.Last = new BonePose[boneCount];
                    pose.Offset = new BonePose[boneCount];
                    pose.HasLast = false;
                    pose.HasOffset = false;
                }

                // A layer that was switched off has nothing current to blend out of
                if (pose.Weight <= 0f)
                {
                    pose.HasLast = false;
                    pose.HasOffset = false;
                }

                int serial = layer.GetTransitionSerial(Playback);
                pose.Capturing = serial != pose.Serial && pose.Weight > 0f;
                pose.Serial = serial;
                if (pose.Capturing)
                {
                    // A pose that has shown nothing yet (a character's first frame) starts on the new
                    // state as it is: there is nothing on screen to blend out of.
                    pose.FromCoverage = pose.HasLast ? pose.Coverage : 1f;
                    pose.Previous = pose.HasLast ? layer.GetPrevious(Playback) : null;
                    pose.PreviousFade = pose.HasOffset ? layer.GetPreviousFade(Playback) : 1f;
                }

                // Which clips the bones read, at what time and with what share: the same for every
                // bone of the pose
                if (pose.Weight > 0f)
                {
                    pose.State?.Collect(bones, Playback, pose.Samples);
                    if (pose.Capturing && pose.FromCoverage > 0f)
                        pose.Previous?.Collect(bones, Playback, pose.PreviousSamples);
                }

                if (pose.ModelSpace)
                {
                    if (pose.ModelRot == null || pose.ModelRot.Length != boneCount)
                        pose.ModelRot = new Quaternion[boneCount];
                    anyModelSpace = true;
                }
            }

            return anyModelSpace;
        }

        /// <summary>Call when every bone of the pose prepared last has been blended.</summary>
        private void FinishLayers()
        {
            var layers = _pose.Layers;
            for (int i = 0; i < layers.Length; i++)
            {
                ref var layer = ref layers[i];
                if (layer.Weight <= 0f) continue;

                layer.HasLast = true;
                if (layer.Capturing) layer.HasOffset = true;
                layer.Previous = null;
            }
        }

        /// <summary>
        /// Blend every layer into one bone's local pose, which comes in as its bind pose.
        ///
        /// A layer plays one state. When a transition starts, the state changes at once and the
        /// layer carries the difference between the pose it was showing and the new state's pose,
        /// which fades out over the transition. Nothing of the old state is sampled any more, so a
        /// transition can be interrupted at any point: the next one blends out of whatever is on
        /// screen. A state that plays nothing stands for the layers below, so a transition to or
        /// from an empty state fades the layer out or in.
        ///
        /// The layer's pose goes on top of the result so far by layer weight × mask weight × how
        /// much of the layer is playing.
        ///
        /// A masked layer with ParentInfluence below 1 keeps its own model-space orientation: each
        /// masked bone gets the local rotation that puts it where the layer's whole pose would,
        /// under the parent the result actually has. <paramref name="modelRot"/> holds those parent
        /// rotations, filled by the caller as it walks the bones in order.
        /// </summary>
        private void BlendLayers(Bone[] bones, int boneIndex, Quaternion[] modelRot, ref BonePose blendPose)
        {
            var bone = bones[boneIndex];
            int parent = bone.Parent;
            var layers = _pose.Layers;
            BonePose pose = default;

            for (int i = 0; i < layers.Length; i++)
            {
                ref var layer = ref layers[i];
                if (layer.Weight <= 0f) continue;

                float mask = layer.MaskWeights != null ? layer.MaskWeights[boneIndex] : 1f;
                bool chainOnly = mask <= 0f;
                if (chainOnly && !(layer.ModelSpace && layer.MaskChain[boneIndex])) continue;

                bool posed = layer.Samples.Sample(bone, boneIndex, ref pose);
                float coverage = posed ? 1f : 0f;

                if (layer.Fade < 1f && (layer.Capturing || layer.HasOffset))
                {
                    ref var offset = ref layer.Offset[boneIndex];

                    if (layer.Capturing)
                    {
                        // What the layer would show this frame without the new transition: the state
                        // it left, one frame on, plus what remained of a transition it interrupts.
                        // Blending out of that instead of last frame's pose keeps the motion going
                        // through the switch. A layer whose previous state was empty held a pose.
                        bool hasReference = false;
                        BonePose reference = default;

                        if (layer.HasLast && layer.FromCoverage > 0f)
                        {
                            if (layer.PreviousSamples.Sample(bone, boneIndex, ref reference))
                            {
                                if (layer.PreviousFade < 1f)
                                {
                                    float left = 1f - layer.PreviousFade;
                                    reference.Rotation = reference.Rotation * Quaternion.Lerp(Quaternion.Identity, offset.Rotation, left);
                                    reference.Position += offset.Position * left;
                                    reference.Scale += offset.Scale * left;
                                }
                            }
                            else
                                reference = layer.Last[boneIndex];

                            hasReference = true;
                        }

                        if (posed && hasReference)
                        {
                            offset.Rotation = Quaternion.Inverse(pose.Rotation) * reference.Rotation;
                            offset.Position = reference.Position - pose.Position;
                            offset.Scale = reference.Scale - pose.Scale;
                        }
                        else
                        {
                            offset.Rotation = Quaternion.Identity;
                            offset.Position = Vector3.Zero;
                            offset.Scale = Vector3.Zero;

                            // To an empty state: this is the pose held while the layer fades out
                            if (hasReference)
                                layer.Last[boneIndex] = reference;
                        }
                    }

                    if (posed)
                    {
                        float remaining = 1f - layer.Fade;
                        pose.Rotation = pose.Rotation * Quaternion.Lerp(Quaternion.Identity, offset.Rotation, remaining);
                        pose.Position += offset.Position * remaining;
                        pose.Scale += offset.Scale * remaining;
                        coverage = layer.FromCoverage + (1f - layer.FromCoverage) * layer.Fade;
                    }
                    else if (layer.HasLast && layer.FromCoverage > 0f)
                    {
                        // Fading out to an empty state: hold the last pose while its share goes to zero
                        pose = layer.Last[boneIndex];
                        posed = true;
                        coverage = layer.FromCoverage * (1f - layer.Fade);
                    }
                }

                layer.Coverage = coverage;

                if (posed)
                    layer.Last[boneIndex] = pose;
                else
                    pose = bone.BindPose;

                if (layer.ModelSpace)
                {
                    layer.ModelRot[boneIndex] = parent >= 0
                        ? layer.ModelRot[parent] * pose.Rotation
                        : pose.Rotation;

                    // The hips of the layer's own pose against the hips the body below has: how
                    // far the layer's upper body is turned on the legs, about the vertical. The
                    // hips come round by a share of it when the pose goes up the hierarchy.
                    if (boneIndex == _pose.RootBone && posed && layer.HipsFollow > 0f && coverage > 0f)
                    {
                        var have = parent >= 0 ? modelRot[parent] * blendPose.Rotation : blendPose.Rotation;
                        var apart = layer.ModelRot[boneIndex] * Quaternion.Inverse(have);
                        if (apart.W < 0f) apart = -apart;
                        _layerHipsTurn += 2f * MathF.Atan2(apart.Y, apart.W) * layer.HipsFollow * layer.Weight * coverage;
                    }

                    if (!chainOnly && posed)
                    {
                        var modelLocal = parent >= 0
                            ? Quaternion.Inverse(modelRot[parent]) * layer.ModelRot[boneIndex]
                            : layer.ModelRot[boneIndex];

                        pose.Rotation = layer.ParentInfluence > 0f
                            ? Quaternion.Lerp(modelLocal, pose.Rotation, layer.ParentInfluence)
                            : modelLocal;
                    }
                }

                if (chainOnly || !posed) continue;

                float factor = layer.Weight * mask * coverage;
                if (factor >= 1f)
                    blendPose = pose;
                else if (factor > 0f)
                {
                    blendPose.Position = Vector3.Lerp(blendPose.Position, pose.Position, factor);
                    blendPose.Scale = Vector3.Lerp(blendPose.Scale, pose.Scale, factor);
                    blendPose.Rotation = Quaternion.Lerp(blendPose.Rotation, pose.Rotation, factor);
                }
            }
        }

        /// <summary>
        /// Compute retargeted rotations via BoneMap (world-space offset method).
        /// </summary>
        private void ComputeRetargetedRotations(Bone[] skeleton)
        {
            int count = skeleton.Length;

            if (_retargetedLocalRot == null || _retargetedLocalRot.Length != count)
            {
                _retargetedLocalRot = new Quaternion[count];
                _retargetedModelRot = new Quaternion[count];
            }

            _boneMap.Retarget(_srcAnimModelRot, _retargetedLocalRot, _retargetedModelRot);

            // One-time debug: trace arm bone directions
            if (!_debugLogged)
            {
                _debugLogged = true;
                var builder = new System.Text.StringBuilder();
                builder.AppendLine($"[RetargetDiag] {_boneMap.Target.Name}");
                for (int i = 0; i < count; i++)
                {
                    int srcIdx2 = _boneMap.TargetToSource[i];
                    if (srcIdx2 < 0) continue;

                    var srcBoneDir = _boneMap.SrcBoneDir[i];
                    var dstBoneDir = _boneMap.DstBoneDir[i];

                    // Where does the source bone point during animation?
                    var srcDelta2 = _srcAnimModelRot[srcIdx2] *
                                    Quaternion.Inverse(_boneMap.SrcBindModelRot[srcIdx2]);
                    var srcAnimDir = Vector3.Transform(srcBoneDir, srcDelta2);

                    // Where does our retargeted bone point?
                    var retargetedModel2 = _retargetedModelRot[i];
                    var retDelta = retargetedModel2 * Quaternion.Inverse(_boneMap.DstBindModelRot[i]);
                    var retDir = Vector3.Transform(dstBoneDir, retDelta);

                    // How far off is the retargeted direction from the source's direction?
                    float dirError = MathF.Acos(Math.Clamp(Vector3.Dot(
                        Vector3.Normalize(srcAnimDir), Vector3.Normalize(retDir)), -1, 1)) * 180f / MathF.PI;

                    if (dirError > 5f)
                        builder.AppendLine($"  [{i}] {skeleton[i].Name,-22} " +
                            $"srcAnimDir=({srcAnimDir.X,6:F2},{srcAnimDir.Y,6:F2},{srcAnimDir.Z,6:F2}) " +
                            $"retDir=({retDir.X,6:F2},{retDir.Y,6:F2},{retDir.Z,6:F2}) " +
                            $"error={dirError:F1}°");
                }
                Debug.Log(builder.ToString());
            }
        }

        /// <summary>
        /// The skinning matrices of one skeleton for the current animation state.
        /// </summary>
        public void GetPose(Skeleton target, Matrix4x4[] bones)
        {
            Bone[] skeleton = target.Bones;
            bool flipXZ = target.FlipXZ;
            int count = skeleton.Length;

            // For retargeted characters: evaluate source skeleton and compute
            // retargeted rotations via model-space approach with direction correction.
            if (_boneMap != null)
            {
                // The source pose carries transition state from frame to frame: build it once per
                // update, also when this Animator drives more than one target skeleton.
                if (!_sourceEvaluated)
                {
                    EvaluateSourceModelRotations();
                    _sourceEvaluated = true;
                }

                ComputeRetargetedRotations(skeleton);
                _trackPoseModelRot = false;

                // What the graph or the set on it asks of the fingers, blended while a set is swapped
                float current = Playback.AnimationSet?.FingerWeight ?? Animation.FingerWeight;
                if (Playback.SetFade < 1f)
                {
                    float previous = Playback.PreviousAnimationSet?.FingerWeight ?? Animation.FingerWeight;
                    current = previous + (current - previous) * Playback.SetFade;
                }
                _poseFingerWeight = Math.Clamp(current * FingerWeight, 0f, 1f);
            }
            else
            {
                if (!_poseStates.TryGetValue(target, out var state))
                {
                    _poseStates[target] = state = new PoseState { RootBone = HipsOf(target) };
                    if (_poseRootSkeleton == null && state.RootBone >= 0)
                    {
                        _poseRootSkeleton = target;
                        _poseRootBone = state.RootBone;
                    }
                }

                // Without retargeting the pose is blended on this skeleton directly: track its
                // model-space rotations when a masked layer keeps its own orientation.
                _poseBones = skeleton;
                _trackPoseModelRot = PrepareLayers(state, target);
                if (_trackPoseModelRot && (_poseModelRot == null || _poseModelRot.Length != count))
                    _poseModelRot = new Quaternion[count];
            }

            for (int i = 0; i < count; i++)
                BlendBone(i, skeleton[i], out bones[i]);

            if (_boneMap == null)
                FinishLayers();

            if (OnBoneTransform != null)
            {
                for (int i = 0; i < count; i++)
                    OnBoneTransform(skeleton[i], ref bones[i]);
            }

            // Up the hierarchy into model space. Where the upper body or the head is asked to face
            // another way, the bones of the spine and neck are turned as they are reached, each by
            // its share: what hangs on them comes round with them.
            //
            // The hips take a share of a turn to the side: of the aim's (TorsoAimHips) and of what a
            // masked layer is turned by against the legs (AnimationLayer.HipsFollow). The thighs
            // turn back by as much, so the legs stay where they are, and the spine turns by that
            // much less along its length, so the shoulders end up where they would anyway.
            var torsoAim = TorsoAim;
            var headLook = HeadLook;
            float hipsTurn = Math.Clamp(torsoAim.X * Math.Clamp(TorsoAimHips, 0f, 1f) + _layerHipsTurn, -MaxHipsTurn, MaxHipsTurn);
            var aim = torsoAim != Vector2.Zero || headLook != Vector2.Zero || hipsTurn != 0f ? AimChainOf(target) : null;
            if (aim != null && aim.Hips < 0) hipsTurn = 0f;
            float aimedYaw = 0f;

            for (int i = 0; i < count; i++)
            {
                if (skeleton[i].Parent > -1)
                    bones[i] = bones[i] * bones[skeleton[i].Parent];

                if (aim == null) continue;

                if (hipsTurn != 0f)
                {
                    if (i == aim.Hips)
                    {
                        TurnInModelSpace(ref bones[i], hipsTurn, 0f, 0f, flipXZ);
                        aimedYaw += hipsTurn;
                        continue;
                    }
                    if (i == aim.LeftLeg || i == aim.RightLeg)
                    {
                        TurnInModelSpace(ref bones[i], -hipsTurn, 0f, 0f, flipXZ);
                        continue;
                    }
                }

                float torso = aim.Torso[i], head = aim.Head[i];
                if (torso == 0f && head == 0f) continue;

                float yaw = (torsoAim.X - hipsTurn) * torso + headLook.X * head;
                float pitch = torsoAim.Y * torso + headLook.Y * head;
                TurnInModelSpace(ref bones[i], yaw, pitch, aimedYaw, flipXZ);
                aimedYaw += yaw;
            }

            OnPostHierarchy?.Invoke(skeleton, bones);

            // For TryGetBonePose: the skeleton the character itself is posed on (the one it is
            // retargeted to, or the first with hips)
            if (_keepBonePoses && ReferenceEquals(target, _boneMap != null ? _boneMap.Target : _poseRootSkeleton ?? target))
            {
                var kept = _bonePoses;
                if (kept == null || kept.Length != count) kept = new Matrix4x4[count];
                Array.Copy(bones, kept, count);
                _bonePoses = kept;
                _bonePoseSkeleton = target;
            }

            for (int i = 0; i < count; i++)
            {
                var m = skeleton[i].OffsetMatrix * bones[i];
                if (flipXZ) m = ConjugateFlipXZ(m);
                bones[i] = Matrix4x4.Transpose(m);
            }
        }

        /// <summary>
        /// Conjugate a matrix by F = diag(-1,1,-1,1): F * M * F.
        /// Converts bone matrices from Assimp LH space to match -X/-Z flipped mesh vertices.
        /// </summary>
        private static Matrix4x4 ConjugateFlipXZ(Matrix4x4 m)
        {
            return new Matrix4x4(
                 m.M11, -m.M12,  m.M13, -m.M14,
                -m.M21,  m.M22, -m.M23,  m.M24,
                 m.M31, -m.M32,  m.M33, -m.M34,
                -m.M41,  m.M42, -m.M43,  m.M44
            );
        }

        private void BlendBone(int boneIndex, Bone bone, out Matrix4x4 matrix)
        {
            BonePose blendPose;

            if (_boneMap != null)
            {
                // Retargeted: the layers were blended on the source skeleton (the clips name its
                // bones) in EvaluateSourceModelRotations; take that bone's result.
                int sourceBoneIndex = _boneMap.TargetToSource[boneIndex];
                if (sourceBoneIndex < 0)
                {
                    // Unmapped bone — hold bind pose
                    var bp = bone.BindPose;
                    matrix = Matrix4x4.CreateScale(bp.Scale)
                           * Matrix4x4.CreateFromQuaternion(bp.Rotation)
                           * Matrix4x4.CreateTranslation(bp.Position);
                    return;
                }

                blendPose = _srcPose[sourceBoneIndex];

                // Use the precomputed model-space retarget rotation
                blendPose.Rotation = _retargetedLocalRot[boneIndex];

                // Fingers can follow in part: each joint between its bind pose and the clip's
                if (_poseFingerWeight < 1f && _boneMap.TargetIsFinger != null && _boneMap.TargetIsFinger[boneIndex])
                    blendPose.Rotation = Quaternion.Lerp(bone.BindPose.Rotation, blendPose.Rotation, _poseFingerWeight);

                // Scale: the target's own, times what the clip does to the source bone's. A rig's
                // bind scale is its own business (unit conversions sit there), not the source's.
                blendPose.Scale = bone.BindPose.Scale * blendPose.Scale
                                / _boneMap.Source.Bones[sourceBoneIndex].BindPose.Scale;

                // Position: transfer root delta, keep bind for all others
                if (sourceBoneIndex == _boneMap.SourceRoot)
                {
                    // Root delta goes through source-parent space → model space →
                    // height scale → target-parent space (handles unit/axis mismatch)
                    var srcDelta = blendPose.Position - _boneMap.Source.Bones[sourceBoneIndex].BindPose.Position;
                    blendPose.Position = bone.BindPose.Position
                        + Vector3.TransformNormal(srcDelta, _boneMap.RootDeltaTransform);
                }
                else
                {
                    blendPose.Position = bone.BindPose.Position;
                }
            }
            else
            {
                blendPose = bone.BindPose;
                BlendLayers(_poseBones, boneIndex, _poseModelRot, ref blendPose);

                if (_trackPoseModelRot)
                {
                    _poseModelRot[boneIndex] = bone.Parent >= 0
                        ? _poseModelRot[bone.Parent] * blendPose.Rotation
                        : blendPose.Rotation;
                }
            }

            var scale = Matrix4x4.CreateScale(blendPose.Scale);
            var rotation = Matrix4x4.CreateFromQuaternion(blendPose.Rotation);
            var translation = Matrix4x4.CreateTranslation(blendPose.Position);
            matrix = scale * rotation * translation;
        }
    }
}
