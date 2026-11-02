using System.Numerics;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Assets
{
    /// <summary>What of a cloth is held (Components.Cloth): which line or edge it hangs from.</summary>
    public enum ClothHold
    {
        /// <summary>Clothing: what lies above a line at the hips (in the rig's rest pose) stays with its bones.</summary>
        AboveHips,

        /// <summary>A banner or a flag: the cloth's edge furthest toward a side of the mesh (its bar, its pole) is held.</summary>
        Edge,
    }

    /// <summary>
    /// How a piece of cloth moves (Components.Cloth): which of it hangs free, how stiff it is, how the air takes it.
    /// What is free is worked out from the mesh, not painted. Clothing (<see cref="ClothHold.AboveHips"/>): a vertex
    /// below the line (the hips in the rig's rest pose, plus <see cref="Anchor"/>) may leave the place its bones would
    /// put it by <see cref="Slope"/> times how far below the line it is, up to <see cref="MostAway"/>; a tabard is held
    /// at the shoulders and the belt and hangs free below, a cloak wants the line raised. A banner or a flag
    /// (<see cref="ClothHold.Edge"/>): the band <see cref="HeldWithin"/> wide along the cloth's edge toward
    /// <see cref="HeldToward"/> is held, and the rest is free by its distance from that band. Shared by every piece that
    /// names it.
    /// </summary>
    [CreateAsset("Cloth Settings")]
    public class ClothSettings : Asset
    {
        [Category("What hangs free")]
        [Description("What is held: a line at the hips for clothing, an edge of the mesh for a banner or a flag.")]
        public ClothHold Held = ClothHold.AboveHips;

        [Description("Clothing: metres above the hips (in the rig's rest pose) of the line below which the cloth hangs free: 0 for a tabard or a coat's skirt, about 0.45 for a cloak held at the shoulders.")]
        [ValueRange(-1f, 1f)]
        public float Anchor = 0f;

        [Description("A banner or a flag: the side of the mesh (in its own space) whose edge is held: up (0, 1, 0) for a banner hanging from a bar, toward its pole for a flag.")]
        public Vector3 HeldToward = Vector3.UnitY;

        [Description("A banner or a flag: how wide a band along that edge is held, in metres.")]
        [ValueRange(0f, 1f)]
        public float HeldWithin = 0.05f;

        [Description("How far a vertex may leave its held place for every metre it is from what is held.")]
        [ValueRange(0f, 3f)]
        public float Slope = 1f;

        [Description("The farthest any vertex may leave its held place, in metres.")]
        [ValueRange(0f, 4f)]
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

        [Description("How strongly the air (the weather's wind near the ground, gusts and all) presses on it, per second: on its face, by how square the wind meets it, plus Streams of it whichever way it faces.")]
        [ValueRange(0f, 5f)]
        public float Drag = 0.3f;

        [Description("Share of Drag the wind pulls it along with whichever way it faces: what streams a flag out from its pole even edge-on to the wind. About 0.6 for a flag, 0.1 for clothing.")]
        [ValueRange(0f, 1f)]
        public float Streams = 0.1f;

        [Description("How much the wind wavers over the cloth, place by place and moment by moment, 0..1: what makes a flag ripple.")]
        [ValueRange(0f, 1f)]
        public float Flutter = 0.3f;

        [Description("Clothing: how far it keeps off the body's parts, in metres, on top of their size.")]
        [ValueRange(0f, 0.1f)]
        public float Thickness = 0.015f;

        [Description("Clothing: how far it may sink towards the body from where its bones put it, in metres (from the axis of the body's part nearest that place): it stays over what is worn under it (mail, a coat), which the body's parts, fitted without it, are thinner than. 0: it only ever moves out. With 2 cm the mail under a tabard showed through.")]
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
