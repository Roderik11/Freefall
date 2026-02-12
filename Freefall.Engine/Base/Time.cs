using System;
using System.Diagnostics;

namespace Freefall.Base
{
    public static class Time
    {
        /// <summary>
        /// Time delta in milliseconds
        /// </summary>
        public static float DeltaMilliseconds { get; private set; }
        
        /// <summary>
        /// Time delta in seconds
        /// </summary>
        public static float Delta { get; private set; }
        
        /// <summary>
        /// Smoothly interpolated Delta in seconds - use this for movement!
        /// </summary>
        public static float SmoothDelta { get; private set; }
       
        /// <summary>
        /// Total time passed since start in second
        /// </summary>
        public static float TotalTime { get; private set; }

        /// <summary>
        /// Frames per second
        /// </summary>
        public static float FPS { get; private set; }
        
        /// <summary>
        /// Total frames rendered
        /// </summary>
        public static long FrameCount { get; private set; }

        private static Stopwatch _watch = new Stopwatch();
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
            Delta = (float)(currentTime - _lastTime);
            
            // Cap delta to prevent massive jumps (e.g. debugging pauses)
            if (Delta > 0.2f) Delta = 0.2f;
            
            _lastTime = currentTime;
            TotalTime = (float)currentTime;
            DeltaMilliseconds = Delta * 1000.0f;

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


