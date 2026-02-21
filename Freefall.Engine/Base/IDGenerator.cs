using System.Security.Cryptography;

namespace Freefall.Base
{
    public static class IDGenerator
    {
        private static int _idCounter;
        private static readonly RandomNumberGenerator _rng = RandomNumberGenerator.Create();

        /// <summary>
        /// Thread-safe auto-incrementing integer ID (for runtime instance tracking).
        /// </summary>
        public static int GetId()
        {
            return Interlocked.Increment(ref _idCounter);
        }

        /// <summary>
        /// Cryptographically random 64-bit unique ID (for persistent serialization).
        /// </summary>
        public static ulong GetUID()
        {
            Span<byte> bytes = stackalloc byte[8];
            _rng.GetBytes(bytes);
            return BitConverter.ToUInt64(bytes);
        }
    }
}
