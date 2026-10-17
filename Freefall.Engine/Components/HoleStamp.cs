using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Cuts the terrain open: inside the stamp the ground is not drawn, casts no shadow, has no collision, carries
    /// no navmesh and grows no grass. A cave mouth, a sinkhole, a mine shaft — whatever mesh fills the opening
    /// brings its own surface and collider, and should overlap the edge of the hole a little.
    ///
    /// The shape is the circle or the closed Spline on the entity, edge noise included; the edge of the hole lies
    /// halfway through the Falloff. Holes are baked on the CPU at heightmap resolution (TerrainBaker.BakeHoles),
    /// saved with the terrain's baked heightmap and cooked into its PhysX height field. A Global hole stamp would
    /// remove the whole terrain and is ignored.
    /// </summary>
    [Icon("icon_holestamp.png")]
    public class HoleStamp : TerrainStamp
    {
        public HoleStamp()
        {
            // A hole has a hard edge: a short falloff only rounds the outline
            Radius = 3f;
            Falloff = 1f;
        }

        protected override void OnEnabledChanged() => MessageDispatcher.Send(EngineMsg.StampChanged, this);

        public override void Destroy()
        {
            base.Destroy();
            // Taken away: whatever it cut is closed again
            MessageDispatcher.Send(EngineMsg.StampChanged, this);
        }

        protected override Color4 GizmoColor => new Color4(0.95f, 0.25f, 0.3f, 1f); // red
    }
}
