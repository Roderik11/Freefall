using System;
using System.Diagnostics;

namespace Freefall.Base
{
    public static class Time
    {
        /// <summary>
        /// Real time since the last frame in milliseconds (the frame time, whatever <see cref="Scale"/> is)
        /// </summary>
        public static float DeltaMilliseconds { get; private set; }
        
        /// <summary>
        /// Time delta in seconds, of the game clock: times <see cref="Scale"/>
        /// </summary>
        public static float Delta { get; private set; }
        
        /// <summary>
        /// Smoothly interpolated Delta in seconds - use this for movement!
        /// </summary>
        public static float SmoothDelta { get; private set; }
       
        /// <summary>
        /// Total time passed since start in seconds, of the game clock: it stands still while <see cref="Scale"/> is 0.
        /// The shaders' Time is this.
        /// </summary>
        public static float TotalTime { get; private set; }

        /// <summary>
        /// How fast the game clock runs against the real one: 1 = as it is, 0.1 = slow motion, 0 = stood still (everything
        /// on Delta, SmoothDelta and TotalTime: effects, particles, animation, the shaders' time). Editor tools set it (the
        /// project's spell bench pauses and slows its looks with it); play mode starts at 1. What must go on regardless (the
        /// editor's camera, its panels' timers) reads <see cref="RealDelta"/> and <see cref="RealTime"/>.
        /// </summary>
        public static float Scale
        {
            get => _scale;
            set => _scale = Math.Clamp(float.IsFinite(value) ? value : 1f, 0f, 4f);
        }
        private static float _scale = 1f;

        /// <summary>Seconds since the last frame on the real clock, whatever <see cref="Scale"/> is.</summary>
        public static float RealDelta { get; private set; }

        /// <summary>Seconds since start on the real clock.</summary>
        public static float RealTime { get; private set; }

        /// <summary>
        /// Frames per second
        /// </summary>
        public static float FPS { get; private set; }
        
        /// <summary>
        /// Total frames rendered
        /// </summary>
        public static long FrameCount { get; private set; }

        private static Stopwatch _watch = new Stopwatch();
        private static double _gameTime;
        private static double _lastTime = 0;

        private static float _fpsCounter;
        private static float _fpsInterval;
        
        // Rolling average for SmoothDelta
        private const int DeltaHistorySize = 8;
        private static float[] _deltaHistory = new float[DeltaHistorySize];
        private static int _deltaHistoryIndex = 0;
        private static float _deltaSum = 0;

        public static void Initialize()
        {
            _watch.Reset();
            _watch.Start();
            _lastTime = 0;
            _gameTime = 0;
            TotalTime = 0;
            Delta = 1f / 60f; // Initial guess
            SmoothDelta = Delta;
            _deltaSum = Delta * DeltaHistorySize;
            
            // Pre-fill history
            for (int i = 0; i < DeltaHistorySize; i++)
                _deltaHistory[i] = Delta;
        }

        public static void Update()
        {
            double currentTime = _watch.Elapsed.TotalSeconds;
            float real = (float)(currentTime - _lastTime);
            
            // Cap delta to prevent massive jumps (e.g. debugging pauses)
            if (real > 0.2f) real = 0.2f;
            
            _lastTime = currentTime;
            RealDelta = real;
            RealTime = (float)currentTime;
            DeltaMilliseconds = real * 1000.0f;
            Delta = real * _scale;
            _gameTime += (double)Delta;
            TotalTime = (float)_gameTime;

            // Rolling average - conserves total time while smoothing
            _deltaSum -= _deltaHistory[_deltaHistoryIndex];
            _deltaHistory[_deltaHistoryIndex] = Delta;
            _deltaSum += Delta;
            _deltaHistoryIndex = (_deltaHistoryIndex + 1) % DeltaHistorySize;
            
            SmoothDelta = _deltaSum / DeltaHistorySize;
            
            FrameCount++;
            
            UpdateFPS();
        }

        private static void UpdateFPS()
        {
            _fpsCounter++;
            _fpsInterval += DeltaMilliseconds;

            if (_fpsInterval >= 1000)
            {
                FPS = _fpsCounter;
                _fpsInterval = 0;
                _fpsCounter = 0;
            }
        }
    }
}


