using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Animation
{
    /// <summary>
    /// Updates every <see cref="Animator"/>: its state machine every frame, its pose (the bones sampled,
    /// blended and uploaded: most of what an Animator costs) as often as the character is seen. A character
    /// is posed every frame while it is big on screen, and every second, fourth or eighth frame as it gets
    /// smaller. Size is measured the way LODs are: its bounds' diameter as a share of the view's height.
    /// Out of view it is posed half as often (its shadow may still be in view), and not at all once it is
    /// smaller than the GPU culls (<see cref="Mesh.LODCullScreenSize"/>) or none of its renderers is shown.
    /// Characters on the same rate take turns, so each frame poses about as many.
    ///
    /// The state machine runs every frame for every Animator that is drawn, posed that frame or not. One that
    /// is not drawn steps every eighth frame by the time since its last step. In between it reports no root
    /// motion and no events, and the next step reports them all, so whoever moves it by its root motion moves
    /// it as far, in steps nobody sees. The system runs before the scripts (ScriptGroup): what they read (bone
    /// poses for held items, events for footsteps) is of the pose drawn this frame, and the parameters they
    /// set count from the next step.
    ///
    /// Tiers are worked out on the main thread, because transforms are not safe to read from several threads.
    /// Then the Animators are stepped and posed in parallel. Engine.Settings.PoseBySize off poses every
    /// character every frame.
    /// </summary>
    [UpdateInEditor]
    [UpdateBefore(typeof(ScriptGroup))]
    public sealed class AnimatorSystem : EntitySystem
    {
        /// <summary>
        /// Screen sizes (the bounds' diameter as a share of the view's height) down to which a character is
        /// posed every frame, every second frame and every fourth. Below the last it is posed every eighth
        /// frame until it is too small to be drawn. Lowered on 2026-10-23 from 0.12 / 0.05 / 0.02, where
        /// animals a few dozen metres off moved choppily: a human is now posed every frame out to about 60 m,
        /// a chicken to about 18 m.
        /// </summary>
        public static float EveryFrameDown = 0.04f, EverySecondDown = 0.015f, EveryFourthDown = 0.008f;

        // Smaller than the GPU draws, with some to spare: the tiers go by the camera of the last frame
        const float Culled = Mesh.LODCullScreenSize * 0.8f;

        // Out of view: posed this many times less often, at most every MostApart frames
        const int OutOfView = 2, MostApart = 8;

        // Not drawn: the state machine steps every this many frames
        const int UnseenStepEvery = 8;

        private readonly List<Animator> _animators = [];
        private readonly List<Animator> _stepped = [];
        private int _stagger;
        private static int _rendererChanges;

        // The tiers on the main thread (items: the Animators stepped), then stepping and posing (items: those posed)
        private static readonly ProfilerMarker TiersMarker = ProfilerMarker.Get("Animator tiers");
        private static readonly ProfilerMarker PosesMarker = ProfilerMarker.Get("Animator poses");

        /// <summary>
        /// Changes when a skinned renderer was added or removed anywhere, or anything was re-parented: an
        /// Animator then finds its renderers again.
        /// </summary>
        internal static int PartsVersion => _rendererChanges + Transform.HierarchyVersion;

        /// <summary>Animators stepped in the last frame.</summary>
        public int Stepped { get; private set; }

        /// <summary>Of those, posed in the last frame.</summary>
        public int Posed { get; private set; }

        protected internal override void Initialize()
        {
            ComponentCache<Animator>.Added += Add;
            ComponentCache<Animator>.Removed += Remove;
            ComponentCache<SkinnedMeshRenderer>.Added += RendererChanged;
            ComponentCache<SkinnedMeshRenderer>.Removed += RendererChanged;
            foreach (var animator in ComponentCache<Animator>.All)
                Add(animator);
        }

        protected internal override void Destroy()
        {
            ComponentCache<Animator>.Added -= Add;
            ComponentCache<Animator>.Removed -= Remove;
            ComponentCache<SkinnedMeshRenderer>.Added -= RendererChanged;
            ComponentCache<SkinnedMeshRenderer>.Removed -= RendererChanged;
        }

        private void Add(Animator animator)
        {
            if (animator.SystemIndex >= 0) return;
            animator.SystemIndex = _animators.Count;
            animator.Stagger = _stagger++;
            _animators.Add(animator);
        }

        private void Remove(Animator animator)
        {
            int index = animator.SystemIndex;
            if (index < 0) return;
            int last = _animators.Count - 1;
            var moved = _animators[last];
            _animators[index] = moved;
            moved.SystemIndex = index;
            _animators.RemoveAt(last);
            animator.SystemIndex = -1;
        }

        private static void RendererChanged(SkinnedMeshRenderer renderer) => _rendererChanges++;

        public override void Update()
        {
            long frame = Time.FrameCount;
            int posed = 0;
            _stepped.Clear();

            using (TiersMarker.Auto(_animators.Count))
            {
                bool bySize = Engine.Settings.PoseBySize;
                var view = new View(Camera.Main);

                for (int i = 0; i < _animators.Count; i++)
                {
                    var animator = _animators[i];
                    if (!animator.Enabled || animator.IsDestroyed || animator.Entity == null || animator.Animation == null)
                        continue;

                    if (animator.PartsChanged || animator.Unmeasured) Measure(animator);

                    int every = bySize && view.Valid && !animator.Previewing ? Every(animator, view) : 1;
                    animator.PoseEvery = every;

                    // The first pose whatever the size: root motion is read off the skeleton it finds. Then by turns,
                    // or at once when the last pose is older than the rate (it came into view, or nearer)
                    bool due = animator.Parts.Count > 0 && (animator.PosedFrame < 0
                        || every > 0 && ((frame + animator.Stagger) % every == 0 || frame - animator.PosedFrame > every));
                    animator.PoseDue = due;
                    if (due) posed++;

                    // Not drawn: the state machine too only by turns, by the time since its last step
                    animator.Unstepped += Time.SmoothDelta;
                    if (!due && every == 0 && (frame + animator.Stagger) % UnseenStepEvery != 0)
                    {
                        animator.Skip();
                        continue;
                    }
                    animator.StepDelta = animator.Unstepped;
                    animator.Unstepped = 0f;

                    _stepped.Add(animator);
                }
            }

            Stepped = _stepped.Count;
            Posed = posed;

            using (PosesMarker.Auto(posed))
            {
                Parallel.For(0, _stepped.Count, i =>
                {
                    var animator = _stepped[i];
                    try
                    {
                        animator.Step();
                        if (animator.PoseDue) animator.Pose(animator.PoseEvery);
                    }
                    catch (Exception e) { CallbackErrors.Report(animator, nameof(Animator.Update), e); }
                });
            }
        }

        /// <summary>How many frames apart to pose a character now: 0 for not at all.</summary>
        private static int Every(Animator animator, in View view)
        {
            bool shown = false;
            var parts = animator.Parts;
            for (int i = 0; i < parts.Count && !shown; i++)
                shown = parts[i].Enabled;
            if (!shown) return 0;

            float radius = animator.BoundsRadius;
            if (radius <= 0f) return 1;

            var center = Vector3.Transform(animator.BoundsCenter, animator.Transform.Matrix);
            float distanceSq = Vector3.DistanceSquared(center, view.Position);
            float radiusSq = radius * radius;
            if (distanceSq <= radiusSq) return 1;

            // Squared screen size, as the GPU picks LODs by (no root)
            float sizeSq = radiusSq * view.Factor / distanceSq;
            int every = sizeSq >= EveryFrameDown * EveryFrameDown ? 1
                : sizeSq >= EverySecondDown * EverySecondDown ? 2
                : sizeSq >= EveryFourthDown * EveryFourthDown ? 4
                : sizeSq >= Culled * Culled ? 8
                : 0;

            if (every > 0 && !view.Sees(center, radius))
                every = Math.Min(every * OutOfView, MostApart);
            return every;
        }

        /// <summary>
        /// The character's renderers, and a sphere round their meshes (bind pose, as the GPU culls skinned
        /// meshes): its centre in the Animator's entity space, its radius in the world. Measured again while
        /// a renderer has no mesh yet.
        /// </summary>
        private static void Measure(Animator animator)
        {
            animator.FindParts();
            animator.Unmeasured = false;

            var parts = animator.Parts;
            Span<Vector4> spheres = parts.Count <= 32 ? stackalloc Vector4[parts.Count] : new Vector4[parts.Count];
            int count = 0;
            var sum = Vector3.Zero;
            for (int i = 0; i < parts.Count; i++)
            {
                var mesh = parts[i].Mesh;
                var transform = parts[i].Transform;
                if (mesh == null || transform == null)
                {
                    animator.Unmeasured = true;
                    continue;
                }

                var box = mesh.BoundingBox;
                var center = Vector3.Transform((box.Min + box.Max) * 0.5f, transform.Matrix);
                var scale = Vector3.Abs(transform.WorldScale);
                float radius = Vector3.Distance(box.Max, box.Min) * 0.5f * MathF.Max(scale.X, MathF.Max(scale.Y, scale.Z));
                spheres[count++] = new Vector4(center, radius);
                sum += center;
            }

            if (count == 0)
            {
                animator.BoundsCenter = Vector3.Zero;
                animator.BoundsRadius = 0f;
                return;
            }

            var middle = sum / count;
            float reach = 0f;
            for (int i = 0; i < count; i++)
                reach = MathF.Max(reach, Vector3.Distance(new Vector3(spheres[i].X, spheres[i].Y, spheres[i].Z), middle) + spheres[i].W);

            Matrix4x4.Invert(animator.Transform.Matrix, out var toLocal);
            animator.BoundsCenter = Vector3.Transform(middle, toLocal);
            animator.BoundsRadius = reach;
        }

        /// <summary>The main camera as the tiers need it: where it is, its LOD factor and the sides of its view.</summary>
        private readonly struct View
        {
            public readonly bool Valid;
            public readonly Vector3 Position;
            public readonly float Factor;
            private readonly Vector4 _left, _right, _bottom, _top;

            public View(Camera camera)
            {
                this = default;
                if (camera == null) return;

                Valid = true;
                Position = camera.Position;
                float scale = Engine.Settings.LODScale;
                Factor = camera.FoVFactor * scale * scale;

                // The sides of the view from the view-projection (row vectors: clip = v · M), facing in. Near and far
                // are left out: the sides alone meet at the camera, and nothing is too far to cast a shadow.
                var m = camera.View * camera.Projection;
                var x = new Vector4(m.M11, m.M21, m.M31, m.M41);
                var y = new Vector4(m.M12, m.M22, m.M32, m.M42);
                var w = new Vector4(m.M14, m.M24, m.M34, m.M44);
                _left = Normalized(w + x);
                _right = Normalized(w - x);
                _bottom = Normalized(w + y);
                _top = Normalized(w - y);
            }

            private static Vector4 Normalized(Vector4 plane)
            {
                float length = new Vector3(plane.X, plane.Y, plane.Z).Length();
                return length > 0f ? plane / length : plane;
            }

            /// <summary>Whether a sphere is at least partly inside the view's sides.</summary>
            public bool Sees(Vector3 center, float radius)
            {
                var p = new Vector4(center, 1f);
                return Vector4.Dot(_left, p) >= -radius && Vector4.Dot(_right, p) >= -radius
                    && Vector4.Dot(_bottom, p) >= -radius && Vector4.Dot(_top, p) >= -radius;
            }
        }
    }
}
