using System.Security.Cryptography;

namespace Freefall.Base
{
    public static class IDGenerator
    {
        // Starts at a random value: an id kept from an earlier session then almost certainly finds nothing,
        // instead of whatever entity happened to get the same number this time.
        private static int _nextId = RandomNumberGenerator.GetInt32(int.MaxValue);

        /// <summary>
        /// Thread-safe auto-incrementing integer ID (for runtime instance tracking). Unique for the lifetime
        /// of the process, never 0.
        ///
        /// It has to be unique: EntityManager and every ComponentCache key entities by it, and a second
        /// entity with the same id is silently dropped — its components never register, wake up or draw.
        /// Random 32-bit ids did collide: with 80,000 entities in a scene, about every other load.
        /// </summary>
        public static int GetId()
        {
            int id;
            do id = System.Threading.Interlocked.Increment(ref _nextId);
            while (id == 0);
            return id;
        }

        // UIDs drawn 64 at a time per thread: one call to the system's generator per UID was 74 ns, and every entity and
        // component of a prefab copy takes one (ram.md "What a spawn takes"). Still the cryptographic generator.
        [ThreadStatic] private static ulong[]? _uids;
        [ThreadStatic] private static int _uidsUsed;

        /// <summary>
        /// Cryptographically random 64-bit unique ID (for persistent serialization).
        /// </summary>
        public static ulong GetUID()
        {
            var uids = _uids;
            if (uids == null || _uidsUsed == uids.Length)
            {
                uids ??= _uids = new ulong[64];
                RandomNumberGenerator.Fill(System.Runtime.InteropServices.MemoryMarshal.AsBytes(uids.AsSpan()));
                _uidsUsed = 0;
            }
            return uids[_uidsUsed++];
        }
    }
}
