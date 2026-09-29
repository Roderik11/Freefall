using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;

namespace Freefall.Editor.Mcp
{
    /// <summary>
    /// Agent screenshots with a visible heads-up for a user watching the editor live:
    ///   Cue    — ScreenshotCueOverlay flashes a border + "hands off the mouse" pill for CueSeconds,
    ///            editor camera input is frozen;
    ///   Settle — overlay hidden, wait until a few overlay-free frames have been presented
    ///            (PrintWindow grabs the last composed frame, which lags the CPU by the frames in flight);
    ///   Capture on the main thread, complete the request, release the camera.
    /// Requests are queued from any thread and never block the main thread; Tick() advances the
    /// state machine once per frame (Program.RenderGui, before the desktop updates and draws).
    /// </summary>
    public static class ScreenshotScheduler
    {
        public const float CueSeconds = 0.75f;
        private const int SettleFrames = 3;
        private const double SettleMinMs = 50;
        private static readonly TimeSpan RequestTimeout = TimeSpan.FromSeconds(60);

        private sealed record Request(CaptureTarget Target, Rectangle? Crop, int MaxSize, bool Png,
            TaskCompletionSource<CaptureResult> Tcs);

        private enum Phase { Cue, Settle }

        private static readonly ConcurrentQueue<Request> _pending = new();
        private static readonly Stopwatch _clock = new();
        private static Request? _current;
        private static Phase _phase;
        private static int _settledFrames;

        /// <summary>Target whose cue should be drawn this frame, or null when no cue is showing. Main thread only.</summary>
        public static CaptureTarget? CueTarget => _current != null && _phase == Phase.Cue ? _current.Target : null;

        /// <summary>0 → 1 over the cue duration.</summary>
        public static float CueProgress => Math.Clamp((float)_clock.Elapsed.TotalSeconds / CueSeconds, 0f, 1f);

        /// <summary>Queue a capture. Safe from any thread; completes after the cue has played and the frame settled.</summary>
        public static Task<CaptureResult> RequestAsync(CaptureTarget target, Rectangle? crop, int maxSize, bool png)
        {
            var tcs = new TaskCompletionSource<CaptureResult>(TaskCreationOptions.RunContinuationsAsynchronously);
            _pending.Enqueue(new Request(target, crop, maxSize, png, tcs));
            return tcs.Task.WaitAsync(RequestTimeout);
        }

        /// <summary>Advance the state machine. Main thread, once per frame.</summary>
        public static void Tick()
        {
            if (_current == null && !Begin())
                return;

            var req = _current!;

            if (_phase == Phase.Cue)
            {
                if (_clock.Elapsed.TotalSeconds < CueSeconds)
                    return;
                _phase = Phase.Settle;
                _settledFrames = 0;
                _clock.Restart();
                return; // this frame is the first one drawn without the overlay
            }

            if (++_settledFrames <= SettleFrames || _clock.Elapsed.TotalMilliseconds < SettleMinMs)
                return;

            try { req.Tcs.TrySetResult(ScreenCapture.Capture(req.Target, req.Crop, req.MaxSize, req.Png)); }
            catch (Exception ex) { req.Tcs.TrySetException(ex); }

            _current = null;
            EditorCamera.InputLocked = false;
        }

        private static bool Begin()
        {
            while (_pending.TryDequeue(out var req))
            {
                // Fail fast instead of playing a cue for a capture that cannot happen.
                if (req.Target == CaptureTarget.Viewport && Program.EditorUI?.SceneViewport == null)
                {
                    req.Tcs.TrySetException(new InvalidOperationException("No scene viewport — is a project open?"));
                    continue;
                }

                _current = req;
                // No editor desktop (landing page) → nothing to draw the cue on, and no camera to freeze.
                _phase = Program.EditorUI != null ? Phase.Cue : Phase.Settle;
                _settledFrames = 0;
                _clock.Restart();
                EditorCamera.InputLocked = true;
                return true;
            }
            return false;
        }
    }
}
