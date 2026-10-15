using System;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Animation
{
    /// <summary>
    /// Replaces the clips of an <see cref="Animation"/> graph, state by state, without copying
    /// the graph: one state machine, one set per weapon type or character style. A state the set
    /// does not list keeps the graph's own clip, so a set only holds what differs.
    /// Assign it to <c>Animator.AnimationSet</c>; swapping it at runtime cross-fades.
    /// </summary>
    [CreateAsset("Animation Set")]
    public class AnimationSet : Asset, IRebuildAfterLoad
    {
        /// <summary>The graph whose states this set fills. A set is ignored on any other graph.</summary>
        public Animation Base;

        public List<ClipOverride> Overrides = new List<ClipOverride>();

        /// <summary>
        /// How much the fingers of a retargeted character follow while this set is on, in place of
        /// the graph's <see cref="Animation.FingerWeight"/>: 1 for a set whose hands hold something.
        /// </summary>
        public float FingerWeight = 1f;

        /// <summary>
        /// For a set whose clips hold something in one hand or on one side (a sword, a bow, a
        /// torch): a clip played left for right (<see cref="AnimationState.Mirror"/>) changes sides
        /// in the hips and legs only. The upper body plays as it is, so the weapon stays where it
        /// is held. Off, the whole body is mirrored, which is right for empty hands.
        /// </summary>
        public bool MirrorLegsOnly;

        [Serializable]
        public class ClipOverride
        {
            /// <summary><see cref="AnimationState.ID"/> in <see cref="Base"/> (blend tree children have their own).</summary>
            public int StateID;
            public AnimationClip Clip;

            /// <summary>Where this clip is taken up, in place of the state's <see cref="AnimationState.Start"/>.</summary>
            public float Start;

            /// <summary>In place of the state's <see cref="AnimationState.Phase"/> and <see cref="AnimationState.MirrorPhase"/>.</summary>
            public float Phase = -1f;
            public float MirrorPhase = -1f;

            /// <summary>Where this clip hands over, in place of the state's <see cref="AnimationState.End"/>.</summary>
            public float End = 1f;
        }

        // Clip and what goes with it per state ID (state IDs are small sequential ints); a null clip = not overridden
        private sealed class Table
        {
            public AnimationClip[] Clips;
            public float[] Starts;
            public float[] Phases;
            public float[] MirrorPhases;
            public float[] Ends;
            public List<ClipOverride> Source;
            public int Count;
        }

        private Table _table;

        /// <summary>The clip this set plays for a state: its override, or the state's own clip.</summary>
        public AnimationClip Resolve(AnimationState state)
        {
            var clips = Current().Clips;
            int id = state.ID;
            if ((uint)id < (uint)clips.Length)
                return clips[id] ?? state.Clip;
            return state.Clip;
        }

        /// <summary>Where the clip of a state is taken up: the override's value for a clip this set replaces, else the state's.</summary>
        public float ResolveStart(AnimationState state)
        {
            var table = Current();
            int id = state.ID;
            if ((uint)id < (uint)table.Clips.Length && table.Clips[id] != null)
                return table.Starts[id];
            return state.Start;
        }

        /// <summary>Where in its loop's cycle the clip of a state takes over, as it is or mirrored: the override's values for a clip this set replaces.</summary>
        public float ResolvePhase(AnimationState state, bool mirrored)
        {
            var table = Current();
            int id = state.ID;
            if ((uint)id < (uint)table.Clips.Length && table.Clips[id] != null)
                return mirrored ? table.MirrorPhases[id] : table.Phases[id];
            return mirrored ? state.MirrorPhase : state.Phase;
        }

        /// <summary>Where the clip of a state hands over: the override's value for a clip this set replaces, else the state's.</summary>
        public float ResolveEnd(AnimationState state)
        {
            var table = Current();
            int id = state.ID;
            if ((uint)id < (uint)table.Clips.Length && table.Clips[id] != null)
                return table.Ends[id];
            return state.End;
        }

        /// <summary>Call after changing an entry of <see cref="Overrides"/> in place.</summary>
        public void Invalidate() => _table = null;

        public void RebuildAfterLoad() => _table = null;

        private Table Current()
        {
            var table = _table;
            if (table == null || table.Count != Overrides.Count || !ReferenceEquals(table.Source, Overrides))
                table = BuildTable();
            return table;
        }

        private Table BuildTable()
        {
            int maxId = 0;
            foreach (var entry in Overrides)
                if (entry != null && entry.StateID > maxId)
                    maxId = entry.StateID;

            // Animators update in parallel and share the set: publish the table only once it is complete.
            var table = new Table
            {
                Clips = new AnimationClip[maxId + 1],
                Starts = new float[maxId + 1],
                Phases = new float[maxId + 1],
                MirrorPhases = new float[maxId + 1],
                Ends = new float[maxId + 1],
                Source = Overrides,
                Count = Overrides.Count,
            };
            foreach (var entry in Overrides)
            {
                if (entry == null || entry.StateID <= 0) continue;
                table.Clips[entry.StateID] = entry.Clip;
                table.Starts[entry.StateID] = entry.Start;
                table.Phases[entry.StateID] = entry.Phase;
                table.MirrorPhases[entry.StateID] = entry.MirrorPhase;
                table.Ends[entry.StateID] = entry.End;
            }

            _table = table;
            return table;
        }
    }
}
