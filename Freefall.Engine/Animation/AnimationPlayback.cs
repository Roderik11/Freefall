using System.Collections.Generic;

namespace Freefall.Animation
{
    /// <summary>
    /// Flat, hash-keyed mutable state container for animation playback.
    /// The Animation asset defines the graph (immutable); this holds the playhead
    /// positions, weights, and state machine state (mutable, per-Animator).
    /// Uses long keys: high 32 bits = object ID, low 32 bits = category + field.
    /// </summary>
    public class AnimationPlayback
    {
        private readonly Dictionary<long, float> _state = new();

        public float Get(long key, float fallback = 0f)
            => _state.TryGetValue(key, out var v) ? v : fallback;

        public void Set(long key, float value)
            => _state[key] = value;

        public bool GetBool(long key)
            => _state.TryGetValue(key, out var v) && v != 0f;

        public void Clear() => _state.Clear();
    }

    /// <summary>
    /// Key generation for AnimationPlayback entries.
    /// Layout: [objectId : 32 bits][category + field : 32 bits]
    /// No overflow, no collisions, works with any ID value.
    /// </summary>
    public static class PK
    {
        // Category offsets in the low 32 bits
        const uint STATE = 0x00000000;
        const uint LAYER = 0x10000000;
        const uint PARAM = 0x20000000;
        const uint BLEND = 0x30000000;
        const uint EVENT = 0x40000000;

        // State fields
        public const int Time   = 0;
        public const int Weight = 1;

        // Layer fields
        public const int Current       = 0;
        public const int InTransition  = 1;
        public const int TransTime     = 2;
        public const int TransDuration = 3;
        public const int TransSource   = 4;
        public const int TransTarget   = 5;
        public const int ReturnTo      = 6;

        static long Key(long id, uint category, uint field)
            => (id << 32) | (category | field);

        public static long State(int stateId, int field) => Key(stateId, STATE, (uint)field);
        public static long Layer(int layerIdx, int field) => Key(layerIdx, LAYER, (uint)field);
        public static long Param(int paramIdx) => Key(paramIdx, PARAM, 0);
        public static long BlendWeight(int treeId, int childIdx) => Key(treeId, BLEND, (uint)childIdx);
        public static long Event(int stateId, int eventIdx) => Key(stateId, EVENT, (uint)eventIdx);
    }
}
