using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using Freefall.Graphics;

namespace Freefall.Animation
{
    /// <summary>Where on a body something landed.</summary>
    public enum BodyZone : byte { Torso, Head, Arm, Leg }

    /// <summary>What a ray or a stretch met of a character's body (<see cref="Components.Animator.RaycastBody"/>).</summary>
    public struct BodyHit
    {
        /// <summary>The part met: an index into <see cref="BodyShape.Parts"/>.</summary>
        public int Part;

        /// <summary>The bone the part moves with.</summary>
        public HumanoidBone Bone;

        public BodyZone Zone;

        /// <summary>The blow went the way the part faces: it came from behind.</summary>
        public bool FromBehind;

        /// <summary>How far along the ray, from its origin (0 for a stretch that touches).</summary>
        public float Distance;

        /// <summary>On the part's surface, in the world.</summary>
        public Vector3 Point;

        /// <summary>Out of the surface, in the world.</summary>
        public Vector3 Normal;
    }

    /// <summary>
    /// One part of a body: a tapered capsule whose section is an oval, rigid with one bone. Its axis runs from
    /// <see cref="From"/> to <see cref="To"/>, both in the bone's frame without scale (metres); across the body it
    /// reaches <see cref="RadiusFrom"/> at one end and <see cref="RadiusTo"/> at the other, front to back
    /// <see cref="Depth"/> times that.
    /// </summary>
    public sealed class BodyPart
    {
        public string Name;
        public HumanoidBone Bone;

        /// <summary>The bone in the skeleton the shape was fitted on.</summary>
        public int BoneIndex;

        /// <summary>The part it hangs from (an index into <see cref="BodyShape.Parts"/>), -1 for the pelvis.</summary>
        public int Parent;

        public BodyZone Zone;
        public Vector3 From, To;
        public float RadiusFrom, RadiusTo;
        public float Depth;

        /// <summary>The way the part faces (front), and across it: unit, in the bone's frame.</summary>
        public Vector3 Forward, Across;

        /// <summary>Where it turns against its parent (the joint its bone starts at), in the bone's frame.</summary>
        public Vector3 Joint;

        /// <summary>
        /// The bone's frame in the rig's rest pose (the bind pose), in the skeleton's model space without scale: what a
        /// ragdoll's joints measure their bends from.
        /// </summary>
        public Matrix4x4 Bind;

        /// <summary>Cubic metres, for weighing it.</summary>
        public float Volume;
    }

    /// <summary>
    /// The shape of a humanoid character's body, part by part (pelvis, belly, chest, head, upper arms, forearms with
    /// the hands, thighs, shins, feet), each rigid with its bone. Fitted from the character's own meshes in their bind
    /// pose, all of them, so it is what is seen: every vertex goes to the part of the bone that carries most of it, and
    /// a part's ends and radii are where those vertices lie. What a blow meets (hitboxes), what a ragdoll is made of.
    ///
    /// <see cref="Of"/> fits once per skeleton and set of meshes. Poses come from the Animator as model matrices of the
    /// skeleton it was fitted on (<see cref="Frames"/>).
    /// </summary>
    public sealed class BodyShape
    {
        public readonly BodyPart[] Parts;
        public readonly Skeleton Skeleton;

        private BodyShape(Skeleton skeleton, BodyPart[] parts)
        {
            Skeleton = skeleton;
            Parts = parts;
        }

        // Fitted shapes by skeleton, then by the meshes they were fitted from
        private static readonly ConditionalWeakTable<Skeleton, Dictionary<ulong, BodyShape>> s_fitted = new();

        // Of the vertices of a part near one end, the share a radius takes in
        private const float RadiusShare = 0.9f;

        // Of a part's vertices along its axis, the share the tip of a head, a hand or a foot takes in
        private const float TipShare = 0.99f;

        // The same for the ends of the trunk (a buckle or a fold of cloth does not stretch it)
        private const float TrunkShare = 0.98f;

        /// <summary>
        /// The shape of a body on <paramref name="skeleton"/> as <paramref name="meshes"/> make it (their LOD 0), or
        /// null for a skeleton that is not a humanoid's. <paramref name="key"/> names the set of meshes: the same key
        /// gives the same shape without fitting again.
        /// </summary>
        public static BodyShape Of(Skeleton skeleton, IReadOnlyList<Mesh> meshes, ulong key)
        {
            if (skeleton == null) return null;
            var fitted = s_fitted.GetOrCreateValue(skeleton);
            lock (fitted)
            {
                if (fitted.TryGetValue(key, out var shape)) return shape;
                shape = Fit(skeleton, meshes);
                fitted[key] = shape;
                return shape;
            }
        }

        // ---- Fitting ----

        private sealed class Plan
        {
            public string Name;
            public HumanoidBone Bone;
            public int BoneIndex;
            public int Parent;
            public BodyZone Zone;
            public Vector3 Start;       // the joint it starts at, model space, bind
            public Vector3 End;         // the joint it ends at, or a point along its way to a tip
            public bool Tip;            // it ends at the tip of what it holds (head, hand, foot), not at a joint

            // An end of the trunk that nothing as wide carries on from (the bottom of the pelvis, the top of the
            // chest): its round end stops where its vertices do, not a radius past the joint (into the head,
            // between the legs)
            public bool FitFrom, FitTo;
            public readonly List<Vector3> Points = new();
        }

        private static BodyShape Fit(Skeleton skeleton, IReadOnlyList<Mesh> meshes)
        {
            var humanoid = HumanoidTemplate.Detect(skeleton);
            if (humanoid == null) return null;

            var bones = skeleton.Bones;
            int count = bones.Length;
            var bind = new Matrix4x4[count];
            for (int i = 0; i < count; i++)
                if (!Matrix4x4.Invert(bones[i].OffsetMatrix, out bind[i])) bind[i] = Matrix4x4.Identity;

            int Bone(HumanoidBone bone) => humanoid.TryGetValue(bone, out int index) ? index : -1;
            Vector3 Joint(int index) => bind[index].Translation;

            int hips = Bone(HumanoidBone.Hips), spine = Bone(HumanoidBone.Spine), chest = Bone(HumanoidBone.Chest);
            int neck = Bone(HumanoidBone.Neck), head = Bone(HumanoidBone.Head);
            if (hips < 0 || spine < 0 || head < 0) return null;
            if (neck < 0) neck = head;

            var plans = new List<Plan>();
            int Add(string name, HumanoidBone bone, int index, int parent, BodyZone zone, Vector3 start, Vector3 end, bool tip)
            {
                plans.Add(new Plan { Name = name, Bone = bone, BoneIndex = index, Parent = parent, Zone = zone, Start = start, End = end, Tip = tip });
                return plans.Count - 1;
            }

            // The pelvis is wide and short: it lies across, from hip to hip, out to where its vertices end. Up the
            // spine its round lower end reached below the crotch, and fitted up to its vertices it was a blob
            int leftThigh = Bone(HumanoidBone.LeftUpperLeg), rightThigh = Bone(HumanoidBone.RightUpperLeg);
            int pelvis = leftThigh >= 0 && rightThigh >= 0
                ? Add("Pelvis", HumanoidBone.Hips, hips, -1, BodyZone.Torso, Joint(leftThigh), Joint(rightThigh), false)
                : Add("Pelvis", HumanoidBone.Hips, hips, -1, BodyZone.Torso, Joint(hips), Joint(spine), false);
            plans[pelvis].FitFrom = true;
            plans[pelvis].FitTo = leftThigh >= 0 && rightThigh >= 0;
            int top = Add("Belly", HumanoidBone.Spine, spine, pelvis, BodyZone.Torso, Joint(spine), Joint(chest >= 0 ? chest : neck), false);
            if (chest >= 0) top = Add("Chest", HumanoidBone.Chest, chest, top, BodyZone.Torso, Joint(chest), Joint(neck), false);
            plans[top].FitTo = true;
            int skull = Add("Head", HumanoidBone.Head, head, top, BodyZone.Head, Joint(neck), Joint(head), true);

            void Limb(string side, HumanoidBone upper, HumanoidBone lower, HumanoidBone end, int parent, BodyZone zone, string upperName, string lowerName)
            {
                int a = Bone(upper), b = Bone(lower), c = Bone(end);
                if (a < 0 || b < 0 || c < 0) return;
                int first = Add(side + upperName, upper, a, parent, zone, Joint(a), Joint(b), false);
                Add(side + lowerName, lower, b, first, zone, Joint(b), Joint(c), zone == BodyZone.Arm);
            }

            Limb("Left ", HumanoidBone.LeftUpperArm, HumanoidBone.LeftLowerArm, HumanoidBone.LeftHand, top, BodyZone.Arm, "Upper Arm", "Forearm");
            Limb("Right ", HumanoidBone.RightUpperArm, HumanoidBone.RightLowerArm, HumanoidBone.RightHand, top, BodyZone.Arm, "Upper Arm", "Forearm");

            void Leg(string side, HumanoidBone thighBone, HumanoidBone shinBone, HumanoidBone footBone, HumanoidBone toesBone)
            {
                int thigh = Bone(thighBone), shin = Bone(shinBone), foot = Bone(footBone), toes = Bone(toesBone);
                if (thigh < 0 || shin < 0 || foot < 0) return;
                int upper = Add(side + "Thigh", thighBone, thigh, pelvis, BodyZone.Leg, Joint(thigh), Joint(shin), false);
                int lower = Add(side + "Shin", shinBone, shin, upper, BodyZone.Leg, Joint(shin), Joint(foot), false);

                // Toward the toes; without a toe bone, forward and down from the ankle
                var ahead = toes >= 0 ? Joint(toes) : Joint(foot) + Forward(skeleton) * 0.15f - Vector3.UnitY * 0.05f;
                Add(side + "Foot", footBone, foot, lower, BodyZone.Leg, Joint(foot), ahead, true);
            }

            Leg("Left ", HumanoidBone.LeftUpperLeg, HumanoidBone.LeftLowerLeg, HumanoidBone.LeftFoot, HumanoidBone.LeftToes);
            Leg("Right ", HumanoidBone.RightUpperLeg, HumanoidBone.RightLowerLeg, HumanoidBone.RightFoot, HumanoidBone.RightToes);

            // Each bone goes with the part of its parent unless it starts a part of its own (parents come before their
            // children); the neck goes with the head, which starts where the neck does
            var partOf = new int[count];
            for (int i = 0; i < count; i++)
            {
                partOf[i] = -1;
                for (int p = 0; p < plans.Count; p++)
                    if (plans[p].BoneIndex == i) partOf[i] = p;
                if (partOf[i] < 0 && bones[i].Parent >= 0) partOf[i] = partOf[bones[i].Parent];
            }
            if (neck != head) partOf[neck] = skull;
            for (int i = 0; i < count; i++)
                if (partOf[i] < 0 && bones[i].Parent >= 0) partOf[i] = partOf[bones[i].Parent];

            // The vertices, each to the part of the bone with most of its weight
            foreach (var mesh in meshes)
                Gather(mesh, skeleton, partOf, plans);

            var forward = Forward(skeleton);
            var parts = new BodyPart[plans.Count];
            for (int p = 0; p < plans.Count; p++)
                parts[p] = Shape(plans[p], bind[plans[p].BoneIndex], forward);

            // A part whose bone has a mirrored or broken bind cannot be posed: leave the whole body out
            foreach (var part in parts)
                if (part == null) return null;

            return new BodyShape(skeleton, parts);
        }

        /// <summary>The way the character faces, in the model space of the skeleton (+Z of the entity).</summary>
        private static Vector3 Forward(Skeleton skeleton) => skeleton.FlipXZ ? -Vector3.UnitZ : Vector3.UnitZ;

        private static void Gather(Mesh mesh, Skeleton skeleton, int[] partOf, List<Plan> plans)
        {
            var positions = mesh?.Positions;
            var weights = mesh?.BoneWeights;
            var indices = mesh?.CpuIndices;
            var meshSkeleton = mesh?.Skeleton;
            if (positions == null || weights == null || indices == null || meshSkeleton == null) return;

            // The mesh's bones as the fitted skeleton's, by name
            var map = new int[meshSkeleton.Bones.Length];
            for (int j = 0; j < map.Length; j++)
            {
                int own = ReferenceEquals(meshSkeleton, skeleton) ? j : skeleton.FindBone(meshSkeleton.BoneNames[j]);
                map[j] = own >= 0 ? partOf[own] : -1;
            }

            // LOD 0 only: the parts of the first LOD and those in none
            var seen = new bool[positions.Length];
            void Part(int index)
            {
                if ((uint)index >= (uint)mesh.MeshParts.Count) return;
                var part = mesh.MeshParts[index];
                if (!part.Enabled) return;
                int end = Math.Min(part.BaseIndex + part.NumIndices, indices.Length);
                for (int i = part.BaseIndex; i < end; i++)
                {
                    long v = indices[i] + (long)part.BaseVertex;
                    if ((ulong)v >= (ulong)positions.Length || seen[v]) continue;
                    seen[v] = true;

                    var w = weights[v];
                    float best = w.Weights.X;
                    float id = w.BoneIDs.X;
                    if (w.Weights.Y > best) { best = w.Weights.Y; id = w.BoneIDs.Y; }
                    if (w.Weights.Z > best) { best = w.Weights.Z; id = w.BoneIDs.Z; }
                    if (w.Weights.W > best) { best = w.Weights.W; id = w.BoneIDs.W; }
                    int bone = (int)id;
                    if (best <= 0f || (uint)bone >= (uint)map.Length || map[bone] < 0) continue;

                    // Into the skeleton's model space: a flipped mesh's vertices are turned round
                    var position = positions[v];
                    if (meshSkeleton.FlipXZ) position = new Vector3(-position.X, position.Y, -position.Z);
                    plans[map[bone]].Points.Add(position);
                }
            }

            if (mesh.LODs.Count > 0)
            {
                foreach (int index in mesh.LODs[0].MeshPartIndices) Part(index);
                if (mesh.NonLodPartIndices != null)
                    foreach (int index in mesh.NonLodPartIndices) Part(index);
            }
            else
            {
                for (int index = 0; index < mesh.MeshParts.Count; index++) Part(index);
            }
        }

        private static BodyPart Shape(Plan plan, Matrix4x4 bind, Vector3 forward)
        {
            if (!Matrix4x4.Decompose(bind, out _, out var rotation, out var position)) return null;
            var frame = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position);
            if (!Matrix4x4.Invert(frame, out var toBone)) return null;

            var start = plan.Start;
            var axis = plan.End - start;
            float length = axis.Length();
            if (length < 1e-4f) axis = Vector3.UnitY;
            axis = Vector3.Normalize(axis);

            // Front to back and across, square to the axis: the way the character faces, or up for a part that points
            // that way (a foot)
            var depth = forward - axis * Vector3.Dot(forward, axis);
            if (depth.LengthSquared() < 0.09f) depth = Vector3.UnitY - axis * axis.Y;
            depth = Vector3.Normalize(depth);
            var across = Vector3.Cross(depth, axis);

            var points = plan.Points;
            int n = points.Count;

            // Where its vertices lie along its way, first to last
            var along = new float[n];
            for (int i = 0; i < n; i++) along[i] = Vector3.Dot(points[i] - start, axis);
            Array.Sort(along);
            float first = n > 0 ? along[Math.Min(n - 1, (int)(n * (1f - TrunkShare)))] : 0f;
            float last = n > 0 ? along[Math.Min(n - 1, (int)(n * TrunkShare))] : length;

            // A part that ends at a tip reaches as far as nearly all its vertices along its way
            if (plan.Tip && n > 0) length = MathF.Max(length, along[Math.Min(n - 1, (int)(n * TipShare))]);
            length = MathF.Max(length, 0.02f);

            // The section at each end: where its vertices are centred square to the axis (a torso's spine lies near the
            // back) and how far they reach across and front to back
            Section(points, start, axis, across, depth, length, 0f, 0.35f, out var centreFrom, out float acrossFrom, out float depthFrom);
            Section(points, start, axis, across, depth, length, 0.65f, 1f, out var centreTo, out float acrossTo, out float depthTo);
            if (acrossFrom <= 0f) { acrossFrom = acrossTo; depthFrom = depthTo; centreFrom = centreTo; }
            if (acrossTo <= 0f) { acrossTo = acrossFrom; depthTo = depthFrom; centreTo = centreFrom; }
            if (acrossFrom <= 0f) { acrossFrom = acrossTo = depthFrom = depthTo = length * 0.2f; }

            float ratio = Math.Clamp(0.5f * (depthFrom / acrossFrom + depthTo / acrossTo), 0.4f, 1.6f);

            // A tip's round end is the tip itself: its axis stops a radius short of it. So do the trunk's outer ends,
            // at the last of their vertices: the top of the chest not below half its length, the pelvis's sides
            // wherever its vertices end
            float reach = length, begin = 0f;
            if (plan.Tip) reach = MathF.Max(length * 0.5f, length - acrossTo);
            if (plan.FitTo && n > 0) reach = Math.Clamp(last - acrossTo, length * 0.5f, MathF.Max(length, last));
            if (plan.FitFrom && n > 0) begin = Math.Clamp(first + acrossFrom, MathF.Min(0f, first), MathF.Max(0f, reach - 0.02f));

            var from = start + axis * begin + across * centreFrom.X + depth * centreFrom.Y;
            var to = start + axis * reach + across * centreTo.X + depth * centreTo.Y;

            // A round cone needs its ends further apart than their radii differ
            float span = Vector3.Distance(from, to);
            float most = span * 0.9f;
            if (MathF.Abs(acrossFrom - acrossTo) > most)
            {
                if (acrossFrom > acrossTo) acrossTo = acrossFrom - most;
                else acrossFrom = acrossTo - most;
            }

            float mean = 0.5f * (acrossFrom + acrossTo);
            return new BodyPart
            {
                Name = plan.Name,
                Bone = plan.Bone,
                BoneIndex = plan.BoneIndex,
                Parent = plan.Parent,
                Zone = plan.Zone,
                From = Vector3.Transform(from, toBone),
                To = Vector3.Transform(to, toBone),
                RadiusFrom = acrossFrom,
                RadiusTo = acrossTo,
                Depth = ratio,
                Forward = Vector3.Normalize(Vector3.TransformNormal(depth, toBone)),
                Across = Vector3.Normalize(Vector3.TransformNormal(across, toBone)),
                Joint = Vector3.Transform(start, toBone),
                Bind = frame,
                Volume =MathF.PI * mean * mean * ratio * span + 4f / 3f * MathF.PI * mean * mean * mean * ratio,
            };
        }

        private static void Section(List<Vector3> points, Vector3 start, Vector3 axis, Vector3 across, Vector3 depth,
            float length, float low, float high, out Vector2 centre, out float acrossRadius, out float depthRadius)
        {
            centre = Vector2.Zero;
            acrossRadius = depthRadius = 0f;

            var section = new List<Vector2>();
            foreach (var point in points)
            {
                var offset = point - start;
                float t = Vector3.Dot(offset, axis) / length;
                if ((low > 0f && t < low) || (high < 1f && t > high)) continue;
                section.Add(new Vector2(Vector3.Dot(offset, across), Vector3.Dot(offset, depth)));
            }
            if (section.Count < 8) return;

            // The middle of its extent, not the mean: one side's folds of cloth do not pull it over
            var x = new float[section.Count];
            var y = new float[section.Count];
            for (int i = 0; i < section.Count; i++) { x[i] = section[i].X; y[i] = section[i].Y; }
            Array.Sort(x);
            Array.Sort(y);
            int lowIndex = (int)(section.Count * (1f - RadiusShare) * 0.5f);
            int highIndex = section.Count - 1 - lowIndex;
            centre = new Vector2(0.5f * (x[lowIndex] + x[highIndex]), 0.5f * (y[lowIndex] + y[highIndex]));
            acrossRadius = MathF.Max(0.01f, 0.5f * (x[highIndex] - x[lowIndex]));
            depthRadius = MathF.Max(0.01f, 0.5f * (y[highIndex] - y[lowIndex]));
        }

        // ---- Posed ----

        /// <summary>
        /// Each part's frame in the world for a pose (<paramref name="poses"/>: the model matrices of
        /// <see cref="Skeleton"/>, as the Animator keeps them) of a character whose entity is at
        /// <paramref name="world"/>. A part's own points (<see cref="BodyPart.From"/> and the rest) go through it.
        /// </summary>
        public bool Frames(Matrix4x4[] poses, in Matrix4x4 world, Span<Matrix4x4> frames)
        {
            if (poses == null || poses.Length < Skeleton.Bones.Length || frames.Length < Parts.Length) return false;
            var toEntity = Skeleton.FlipXZ ? Matrix4x4.CreateScale(-1f, 1f, -1f) * world : world;
            for (int i = 0; i < Parts.Length; i++)
            {
                if (!Matrix4x4.Decompose(poses[Parts[i].BoneIndex], out _, out var rotation, out var position)) return false;
                frames[i] = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position) * toEntity;
            }
            return true;
        }

        /// <summary>
        /// The first part a ray meets (from <paramref name="origin"/> along <paramref name="direction"/>, a unit vector,
        /// up to <paramref name="distance"/>), the ray being <paramref name="thickness"/> thick, through the parts'
        /// world frames (<see cref="Frames"/>). A ray that starts inside a part meets it at 0.
        /// </summary>
        public bool Raycast(ReadOnlySpan<Matrix4x4> frames, Vector3 origin, Vector3 direction, float distance, float thickness, out BodyHit hit)
        {
            hit = default;
            float nearest = distance;
            bool found = false;

            for (int i = 0; i < Parts.Length; i++)
            {
                var part = Parts[i];
                if (!Matrix4x4.Invert(frames[i], out var toPart)) continue;

                // Into the part's frame, front to back squeezed so that its section is round
                var squeeze = Squeeze(part);
                var o = Squeezed(Vector3.Transform(origin, toPart), part, squeeze);
                var d = Squeezed(Vector3.TransformNormal(direction, toPart), part, squeeze);
                float speed = d.Length();
                if (speed < 1e-8f) continue;
                d /= speed;

                float ra = part.RadiusFrom + thickness, rb = part.RadiusTo + thickness;
                var a = Squeezed(part.From, part, squeeze);
                var b = Squeezed(part.To, part, squeeze);

                float t;
                if (RoundConeDistance(o, a, b, ra, rb) <= 0f) t = 0f;
                else
                {
                    t = RoundConeIntersect(o, d, a, b, ra, rb);
                    if (t < 0f) continue;
                    t /= speed;   // back to the world's distances along the ray
                }
                if (t >= nearest) continue;

                nearest = t;
                found = true;
                hit.Part = i;
                hit.Distance = t;
                hit.Point = origin + direction * t;
            }

            if (found) Finish(frames, direction, ref hit);
            return found;
        }

        /// <summary>
        /// The part a stretch (from <paramref name="a"/> to <paramref name="b"/>, <paramref name="thickness"/> thick)
        /// comes nearest to, if it touches one: a blade between two frames. <paramref name="travel"/> is the way it goes,
        /// for the front and the back.
        /// </summary>
        public bool Touches(ReadOnlySpan<Matrix4x4> frames, Vector3 a, Vector3 b, float thickness, Vector3 travel, out BodyHit hit)
        {
            hit = default;
            float nearest = 0f;
            bool found = false;

            for (int i = 0; i < Parts.Length; i++)
            {
                var part = Parts[i];
                if (!Matrix4x4.Invert(frames[i], out var toPart)) continue;

                var squeeze = Squeeze(part);
                var pa = Squeezed(Vector3.Transform(a, toPart), part, squeeze);
                var pb = Squeezed(Vector3.Transform(b, toPart), part, squeeze);
                var ca = Squeezed(part.From, part, squeeze);
                var cb = Squeezed(part.To, part, squeeze);

                // The place on the stretch nearest the part's axis, and how far it is from the surface
                ClosestOnSegments(pa, pb, ca, cb, out float s, out _);
                float away = RoundConeDistance(Vector3.Lerp(pa, pb, s), ca, cb, part.RadiusFrom, part.RadiusTo) - thickness;
                if (away > 0f || (found && away >= nearest)) continue;

                nearest = away;
                found = true;
                hit.Part = i;
                hit.Point = Vector3.Lerp(a, b, s);
            }

            if (!found) return false;
            var way = travel.LengthSquared() > 1e-10f ? Vector3.Normalize(travel) : Vector3.Normalize(b - a);
            Finish(frames, way, ref hit);
            return true;
        }

        private void Finish(ReadOnlySpan<Matrix4x4> frames, Vector3 way, ref BodyHit hit)
        {
            var part = Parts[hit.Part];
            hit.Bone = part.Bone;
            hit.Zone = part.Zone;

            // The way the struck part faces: its own for the torso and the head, the chest's for a limb
            int facing = hit.Part;
            if (part.Zone == BodyZone.Arm || part.Zone == BodyZone.Leg)
            {
                facing = part.Parent;
                while (facing >= 0 && Parts[facing].Zone != BodyZone.Torso) facing = Parts[facing].Parent;
                if (facing < 0) facing = 0;
            }
            var front = Vector3.TransformNormal(Parts[facing].Forward, frames[facing]);
            hit.FromBehind = Vector3.Dot(way, front) > 0f;

            // Out of the surface: from the nearest place on the axis
            var from = Vector3.Transform(part.From, frames[hit.Part]);
            var to = Vector3.Transform(part.To, frames[hit.Part]);
            var span = to - from;
            float length = span.LengthSquared();
            float t = length > 1e-10f ? Math.Clamp(Vector3.Dot(hit.Point - from, span) / length, 0f, 1f) : 0f;
            var normal = hit.Point - (from + span * t);
            hit.Normal = normal.LengthSquared() > 1e-10f ? Vector3.Normalize(normal) : -way;
        }

        // Front to back, a part's points go in by its depth ratio: (depth axis, how much)
        private static (Vector3 Axis, float By) Squeeze(BodyPart part) => (part.Forward, 1f / part.Depth);

        private static Vector3 Squeezed(Vector3 p, BodyPart part, (Vector3 Axis, float By) squeeze)
        {
            float along = Vector3.Dot(p, squeeze.Axis);
            return p + squeeze.Axis * (along * (squeeze.By - 1f));
        }

        /// <summary>The nearest places on two stretches (a–b and c–d), as shares of each.</summary>
        internal static void ClosestOnSegments(Vector3 a, Vector3 b, Vector3 c, Vector3 d, out float s, out float t)
        {
            var u = b - a;
            var v = d - c;
            var w = a - c;
            float uu = Vector3.Dot(u, u), uv = Vector3.Dot(u, v), vv = Vector3.Dot(v, v);
            float uw = Vector3.Dot(u, w), vw = Vector3.Dot(v, w);
            float denominator = uu * vv - uv * uv;

            s = denominator > 1e-10f ? Math.Clamp((uv * vw - vv * uw) / denominator, 0f, 1f) : 0f;
            t = vv > 1e-10f ? Math.Clamp((uv * s + vw) / vv, 0f, 1f) : 0f;
            s = uu > 1e-10f ? Math.Clamp((uv * t - uw) / uu, 0f, 1f) : 0f;
        }

        /// <summary>
        /// How far a point is outside a round cone (two spheres, of radius <paramref name="ra"/> at
        /// <paramref name="a"/> and <paramref name="rb"/> at <paramref name="b"/>, and the cone that joins them);
        /// below 0 inside. Inigo Quilez's exact distance.
        /// </summary>
        internal static float RoundConeDistance(Vector3 p, Vector3 a, Vector3 b, float ra, float rb)
        {
            var ba = b - a;
            float l2 = Vector3.Dot(ba, ba);
            if (l2 < 1e-12f) return Vector3.Distance(p, a) - MathF.Max(ra, rb);
            float rr = ra - rb;
            float a2 = l2 - rr * rr;
            float il2 = 1f / l2;

            var pa = p - a;
            float y = Vector3.Dot(pa, ba);
            float z = y - l2;
            var q = pa * l2 - ba * y;
            float x2 = Vector3.Dot(q, q);
            float y2 = y * y * l2;
            float z2 = z * z * l2;

            float k = MathF.Sign(rr) * rr * rr * x2;
            if (MathF.Sign(z) * a2 * z2 > k) return MathF.Sqrt(x2 + z2) * il2 - rb;
            if (MathF.Sign(y) * a2 * y2 < k) return MathF.Sqrt(x2 + y2) * il2 - ra;
            return (MathF.Sqrt(x2 * a2 * il2) + y * rr) * il2 - ra;
        }

        /// <summary>
        /// Where a ray (from <paramref name="o"/> along the unit <paramref name="d"/>) first meets a round cone, or -1.
        /// Inigo Quilez's intersection.
        /// </summary>
        internal static float RoundConeIntersect(Vector3 o, Vector3 d, Vector3 a, Vector3 b, float ra, float rb)
        {
            var ba = b - a;
            var oa = o - a;
            var ob = o - b;
            float rr = ra - rb;
            float m0 = Vector3.Dot(ba, ba);
            float m1 = Vector3.Dot(ba, oa);
            float m2 = Vector3.Dot(ba, d);
            float m3 = Vector3.Dot(d, oa);
            float m5 = Vector3.Dot(oa, oa);
            float m6 = Vector3.Dot(ob, d);
            float m7 = Vector3.Dot(ob, ob);

            // The cone between the spheres
            float d2 = m0 - rr * rr;
            float k2 = d2 - m2 * m2;
            float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
            float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2f - m0 * ra * ra;
            float h = k1 * k1 - k0 * k2;
            if (h < 0f) return -1f;
            if (MathF.Abs(k2) > 1e-12f)
            {
                float t = (-MathF.Sqrt(h) - k1) / k2;
                float y = m1 - ra * rr + t * m2;
                if (y > 0f && y < d2) return t >= 0f ? t : -1f;
            }

            // The spheres at the ends
            float h1 = m3 * m3 - m5 + ra * ra;
            float h2 = m6 * m6 - m7 + rb * rb;
            if (MathF.Max(h1, h2) < 0f) return -1f;
            float best = float.MaxValue;
            if (h1 > 0f)
            {
                float t1 = -m3 - MathF.Sqrt(h1);
                if (t1 >= 0f) best = t1;
            }
            if (h2 > 0f)
            {
                float t2 = -m6 - MathF.Sqrt(h2);
                if (t2 >= 0f && t2 < best) best = t2;
            }
            return best < float.MaxValue ? best : -1f;
        }
    }
}
