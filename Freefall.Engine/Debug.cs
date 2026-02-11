using System;
using System.Diagnostics;

namespace Freefall
{
    /// <summary>
    /// Debug logging utilities. All methods are compiled out in Release builds.
    /// </summary>
    public static class Debug
    {
        [Conditional("DEBUG")]
        public static void Log(string message)
        {
            Console.WriteLine(message);
        }

        [Conditional("DEBUG")]
        public static void Log(string category, string message)
        {
            Console.WriteLine($"[{category}] {message}");
        }

        [Conditional("DEBUG")]
        public static void LogWarning(string message)
        {
            Console.WriteLine($"WARNING: {message}");
        }

        [Conditional("DEBUG")]
        public static void LogWarning(string category, string message)
        {
            Console.WriteLine($"[{category}] WARNING: {message}");
        }

        [Conditional("DEBUG")]
        public static void LogError(string message)
        {
            Console.WriteLine($"ERROR: {message}");
        }

        [Conditional("DEBUG")]
        public static void LogError(string category, string message)
        {
            Console.WriteLine($"[{category}] ERROR: {message}");
        }

        /// <summary>
        /// Always logs regardless of build configuration.
        /// Use sparingly for critical messages.
        /// </summary>
        public static void LogAlways(string message)
        {
            Console.WriteLine(message);
        }

        /// <summary>
        /// Always logs regardless of build configuration.
        /// Use sparingly for critical messages.
        /// </summary>
        public static void LogAlways(string category, string message)
        {
            Console.WriteLine($"[{category}] {message}");
        }
    }
}
