using Freefall.Base;

namespace Freefall.Graphics
{
    /// <summary>
    /// A trace for a crash in native code (a driver, vkd3d-proton under Proton), which takes the process down without an
    /// exception: with <see cref="SceneLoad.LogSteps"/> (the player) a line goes to the log before each pipeline is
    /// created, and through the first world frames of each scene load before each pass (<see cref="PixMarker"/>), the
    /// wait for the GPU, the frame's close, submit and present. The log's last line then names what it died in.
    /// </summary>
    public static class FrameTrace
    {
        /// <summary>How many frames are traced from the one a load first draws its world in.</summary>
        public const int WorldFrames = 3;

        private static int _framesLeft, _frame;
        private static string _scene = "";

        /// <summary>A load starts drawing its world (SceneLoad, at its terrain step): trace its next frames.</summary>
        internal static void BeginWorld(string scene)
        {
            if (!SceneLoad.LogSteps) return;
            _scene = scene;
            _frame = 0;
            _framesLeft = WorldFrames;
        }

        /// <summary>A pipeline is about to be created (any thread): logged whenever steps are.</summary>
        public static void Pipeline(string kind, string? name)
        {
            if (SceneLoad.LogSteps) Debug.Trace($"[Trace] creating {kind} pipeline {name ?? "(unnamed)"}");
        }

        /// <summary>Something in a traced frame is about to run: a pass, the wait for the GPU, close, submit, present.</summary>
        public static void Step(string what)
        {
            if (_framesLeft > 0) Debug.Trace($"[Trace] {_scene} world frame {_frame + 1}: {what}");
        }

        /// <summary>The primary view has presented: the traced frame is done.</summary>
        internal static void FrameDone()
        {
            if (_framesLeft <= 0) return;
            Step("presented");
            _frame++;
            _framesLeft--;
        }
    }
}
