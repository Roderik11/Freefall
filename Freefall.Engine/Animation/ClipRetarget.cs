using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// A clip made on one rig turned into a clip of another, once: what a graph plays on the second rig's clips can then
    /// play it too, on any layer, blended with them (a pack's rider clips in the Human graph, under its upper-body
    /// swings). The same mapping and math as an Animator that retargets every frame (<see cref="BoneMap.Retarget"/>),
    /// sampled on the clip's frames. Applied at load for a model file whose clips name another rig
    /// (ModelImporter.RetargetTo).
    /// </summary>
    public static class ClipRetarget
    {
        /// <summary>
        /// <paramref name="clip"/>, made on <paramref name="source"/>, as a clip of <paramref name="target"/>: a
        /// rotation channel for every target bone the rigs share, and the hips' travel as position keys, scaled from
        /// the one rig's height to the other's. Null when the rigs share no bones.
        /// </summary>
        public static AnimationClip Bake(AnimationClip clip, Skeleton source, Skeleton target,
            FingerRetargeting fingers = FingerRetargeting.KeepShape)
        {
            if (clip == null || source?.Bones == null || target?.Bones == null) return null;
            var map = HumanoidTemplate.TryMap(source, target, fingers) ?? BoneMap.CreateFromNames(source, target);
            if (map == null) return null;

            var srcBones = source.Bones;
            var dstBones = target.Bones;
            var channels = clip.ChannelsFor(srcBones);

            int srcRoot = map.SourceRoot, dstRoot = map.TargetRoot;
            bool hasRoot = (uint)srcRoot < (uint)srcBones.Length && (uint)dstRoot < (uint)dstBones.Length
                           && map.TargetToSource[dstRoot] == srcRoot;

            // Where the target's hips hang from, for their travel: model space of the target's bind back into the
            // space of the bone above them
            var dstParentWorld = Matrix4x4.Identity;
            if (hasRoot)
                for (int p = dstBones[dstRoot].Parent; p >= 0; p = dstBones[p].Parent)
                    dstParentWorld = dstParentWorld * dstBones[p].BindPoseMatrix;
            if (!Matrix4x4.Invert(dstParentWorld, out var toDstParent)) toDstParent = Matrix4x4.Identity;

            // A key per frame of the clip, evenly from its start to its end
            int frames = Math.Max(1, (int)MathF.Round(clip.Duration));
            float step = clip.Duration / frames;

            var srcModelRot = new Quaternion[srcBones.Length];
            var srcModel = new Matrix4x4[srcBones.Length];
            var localRot = new Quaternion[dstBones.Length];
            var modelRot = new Quaternion[dstBones.Length];
            var rotations = new List<QuaternionKey>[dstBones.Length];
            var travel = hasRoot ? new List<VectorKey>(frames + 1) : null;
            for (int i = 0; i < dstBones.Length; i++)
                if (map.TargetToSource[i] >= 0) rotations[i] = new List<QuaternionKey>(frames + 1);

            for (int f = 0; f <= frames; f++)
            {
                float ticks = f * step;

                // The source's pose up its hierarchy, every bone (those above the hips too, which carry the clip's
                // turn and travel on some rigs)
                for (int j = 0; j < srcBones.Length; j++)
                {
                    var pose = srcBones[j].BindPose;
                    channels[j]?.GetBoneTransform(srcBones[j], ticks, ref pose);
                    var local = Matrix4x4.CreateScale(pose.Scale)
                              * Matrix4x4.CreateFromQuaternion(pose.Rotation)
                              * Matrix4x4.CreateTranslation(pose.Position);
                    int parent = srcBones[j].Parent;
                    srcModelRot[j] = parent >= 0 ? srcModelRot[parent] * pose.Rotation : pose.Rotation;
                    srcModel[j] = parent >= 0 ? local * srcModel[parent] : local;
                }

                // The bones above the hips back at their bind: Retarget takes each bone against its parent, and would
                // drop a turn keyed above the hips (a rider's root turning him from the horse's side to its saddle).
                // So the hips carry it, as the travel does.
                if (hasRoot)
                    for (int p = srcBones[srcRoot].Parent; p >= 0; p = srcBones[p].Parent)
                        srcModelRot[p] = map.SrcBindModelRot[p];

                map.Retarget(srcModelRot, localRot, modelRot);
                for (int i = 0; i < dstBones.Length; i++)
                    rotations[i]?.Add(new QuaternionKey { Time = ticks, Value = localRot[i] });

                // The hips: how far from their bind they are in the source's model space, at the target's height
                if (travel != null)
                {
                    var off = (srcModel[srcRoot].Translation - map.SrcBindModelPos[srcRoot]) * map.PositionScale;
                    travel.Add(new VectorKey
                    {
                        Time = ticks,
                        Value = dstBones[dstRoot].BindPose.Position + Vector3.TransformNormal(off, toDstParent),
                    });
                }
            }

            var baked = new AnimationClip
            {
                Name = clip.Name,
                Duration = clip.Duration,
                TicksPerSecond = clip.TicksPerSecond,
                Skeleton = target,
                Events = clip.Events,
            };
            for (int i = 0; i < dstBones.Length; i++)
            {
                if (rotations[i] == null) continue;
                baked.AddChannel(new AnimationChannel
                {
                    Target = dstBones[i].Name,
                    Rotation = new QuaternionKeys(rotations[i]),
                    Position = i == dstRoot && travel != null ? new VectorKeys(travel) : null!,
                });
            }
            return baked;
        }
    }
}
