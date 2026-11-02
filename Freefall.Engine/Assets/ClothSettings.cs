using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>
    /// How a piece of clothing moves as cloth (Components.Cloth): which of it hangs free, how stiff it is, how the air
    /// takes it. What is free is worked out from the mesh, not painted: a vertex below the anchor line (the hips in
    /// the rig's rest pose, plus <see cref="Anchor"/>) may leave the place its bones would put it by
    /// <see cref="Slope"/> times how far below the line it is, up to <see cref="MostAway"/>; at and above the line it
    /// stays with its bones. A tabard is held at the shoulders and the belt and hangs free below; raise the line for a
    /// cloak. Shared by every piece that names it.
    /// </summary>
    [CreateAsset("Cloth Settings")]
    public class ClothSettings : Asset
    {
        [Category("What hangs free")]
        [Description("Metres above the hips (in the rig's rest pose) of the line below which the cloth hangs free: 0 for a tabard or a coat's skirt, about 0.45 for a cloak held at the shoulders.")]
        [ValueRange(-1f, 1f)]
        public float Anchor = 0f;

        [Description("How far a vertex may leave its bones' place for every metre it hangs below the line.")]
        [ValueRange(0f, 3f)]
        public float Slope = 1f;

        [Description("The farthest any vertex may leave its bones' place, in metres.")]
        [ValueRange(0f, 2f)]
        public float MostAway = 0.6f;

        [Category("Cloth")]
        [Description("How firmly the cloth keeps its length along its edges, 0..1 a pass (1: as stiff as the passes allow).")]
        [ValueRange(0f, 1f)]
        public float Stiffness = 1f;

        [Description("How firmly it keeps from folding (across two triangles), 0..1 a pass: low for linen, higher for felt and leather.")]
        [ValueRange(0f, 1f)]
        public float Bending = 0.25f;

        [Description("Share of its motion it loses each step: more, and it swings less and settles sooner.")]
        [ValueRange(0f, 0.5f)]
        public float Damping = 0.04f;

        [Description("How strongly the air (the weather's wind near the ground, gusts and all) pulls it along, per second. It pulls every particle alike, whichever way the cloth faces: keep it low.")]
        [ValueRange(0f, 5f)]
        public float Drag = 0.3f;

        [Description("How far it keeps off the body's parts, in metres, on top of their size.")]
        [ValueRange(0f, 0.1f)]
        public float Thickness = 0.015f;

        [Description("How far it may sink towards the body from where its bones put it, in metres (from the axis of the body's part nearest that place): it stays over what is worn under it (mail, a coat), which the body's parts, fitted without it, are thinner than. 0: it only ever moves out. With 2 cm the mail under a tabard showed through.")]
        [ValueRange(0f, 0.3f)]
        public float Backstop = 0f;

        [Description("How far a free particle may get from the held particle nearest it, against how far it is at rest (1.05: 5 % further): keeps it from stretching, which passes over its edges alone do not do against a steady pull. 0 = no such limit.")]
        [ValueRange(0f, 2f)]
        public float Tether = 1.05f;

        [Category("Solver")]
        [Description("Steps a frame: each moves it by a share of the frame and solves it again. More for fast motion.")]
        [ValueRange(1, 8)]
        public int Substeps = 3;

        [Description("Passes over its constraints a step.")]
        [ValueRange(1, 32)]
        public int Iterations = 8;

        /// <summary>What a Cloth without settings uses.</summary>
        public static readonly ClothSettings Default = new();
    }
}
