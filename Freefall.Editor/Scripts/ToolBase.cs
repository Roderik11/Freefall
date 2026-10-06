using System;
using System.Numerics;

namespace Freefall.Editor
{
    public enum MoveAxis
    {
        Z = 0,
        X = 1,
        Y = 2,
        Free = 3,
        XZ = 4,
        XY = 5,
        ZY = 6,
        SZ = 7,
        SX = 8,
        SY = 9,
    }

    public enum RotateAxis
    {
        Free = 0,
        X = 1,
        Y = 2,
        Z = 3,
    }

    public enum TransformAxis
    {
        None, CameraUp, CameraForward, CameraRight, LocalUp, LocalForward, LocalRight, WorldUp, WorldForward, WorldRight
    }

    public class AxisData
    {
        public TransformAxis Axis1;
        public TransformAxis Axis2;
    }

    public enum TransformMode
    {
        Translate,
        Rotate,
        Scale
    }

    public abstract class ToolBase
    {
        public bool MouseCaptured { get; internal set; }
        public bool IsClicked { get; internal set; }

        public abstract void Initialize();

        public abstract void Update(Components.Camera camera);

        /// <summary>
        /// Called when a gizmo mesh part is picked via GPU picking.
        /// The tool should start its drag operation from this.
        /// </summary>
        public virtual void StartDrag(Components.Camera camera, int meshPart) { }

        public virtual void Render() { }

        public virtual void Disable() { }
    }
}
