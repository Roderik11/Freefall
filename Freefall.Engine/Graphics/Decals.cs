using System.Numerics;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// Spawned decals (DECALS_PLAN.md B3): marks game code leaves where something happens (footprints, blood where a hit
    /// lands, scorch where a spell strikes), kept in a ring of <see cref="DecalBuffer.RingSize"/> on the GPU and faded out
    /// by their age. A spawn writes one record: an event, never per-frame work. Visual only: under the networking design
    /// each client spawns its own. Docs: .agent/knowledge/rendering_engine/features/decals.md.
    /// </summary>
    public static class Decals
    {
        /// <summary>
        /// Leave a mark: <paramref name="material"/> (an effect "decal" .mat) in a box at <paramref name="position"/>,
        /// turned by <paramref name="rotation"/> (its Y the projection axis: identity lies on the ground and projects
        /// down), <paramref name="size"/> metres (X, Z across it, Y how deep it reaches). It lasts
        /// <paramref name="lifetime"/> seconds, fading out over the last <paramref name="fadeOut"/>.
        /// <paramref name="variant"/> picks the tile of the material's atlas (0..255).
        /// A <paramref name="carrier"/> (what the mark is made on, if that moves: a ship's deck) takes it along from then
        /// on, the GPU putting the box under the carrier's matrix each frame with no work here; position and rotation are
        /// still given in the world. Marks go when their carrier's entity does.
        /// </summary>
        public static void Spawn(Material material, Vector3 position, Quaternion rotation, Vector3 size,
            float lifetime = 60f, float fadeOut = 10f, int variant = 0, float opacity = 1f, int priority = 0, Vector3? tint = null,
            Transform? carrier = null)
        {
            var box = Matrix4x4.CreateScale(size) * Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position);
            DecalBuffer.Instance?.Spawn(material, box, variant, tint ?? Vector3.One, opacity, priority, lifetime, fadeOut, carrier);
        }
    }
}
