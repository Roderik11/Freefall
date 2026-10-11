using System;
using System.Collections.Generic;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Animation
{
    /// <summary>
    /// A named animation parameter definition.
    /// Runtime value lives in AnimationPlayback, keyed by Index.
    /// </summary>
    [Serializable]
    public class AnimationParameter
    {
        public string Name;
        public float DefaultValue;
        public bool IsTrigger;

        /// <summary>Index in the parent Animation's Parameters list.</summary>
        public int Index;

        public AnimationParameter() { }

        public AnimationParameter(string name, float defaultValue = 0, bool isTrigger = false)
        {
            Name = name;
            DefaultValue = defaultValue;
            IsTrigger = isTrigger;
        }
    }

    /// <summary>
    /// Root animation controller — immutable definition of layers, states, transitions, and parameters.
    /// All runtime state lives in AnimationPlayback on the Animator.
    /// </summary>
    [CreateAsset("Animation")]
    [AssetTypeAlias("AnimationData")]
    public class Animation : Assets.Asset, Assets.IRebuildAfterLoad
    {
        public List<AnimationParameter> Parameters = new List<AnimationParameter>();
        public List<AnimationLayer> Layers = new List<AnimationLayer>();

        /// <summary>
        /// How much the fingers of a retargeted character follow this graph's clips: 1 = in full,
        /// 0 = the hands stay as the character is bound. Clips made with the hands closed around
        /// something look clenched on empty hands: an unarmed graph wants well under 1. An
        /// <see cref="AnimationSet"/> has its own value for its clips.
        /// </summary>
        public float FingerWeight = 1f;

        // Sequential counter for state IDs — must produce small ints
        // because layer state tracking stores IDs as floats in the playback dictionary.
        private int _nextStateId = 1;

        /// <summary>Adds a layer after the existing ones: it is evaluated last and overrides them.</summary>
        public AnimationLayer AddLayer(string name)
        {
            var layer = new AnimationLayer { Name = name };
            Layers.Add(layer);
            ReindexLayers();
            return layer;
        }

        public void RemoveLayer(AnimationLayer layer)
        {
            if (Layers.Remove(layer))
                ReindexLayers();
        }

        /// <summary>Moves a layer earlier (-1) or later (+1) in the evaluation order. False at either end.</summary>
        public bool MoveLayer(AnimationLayer layer, int offset)
        {
            int from = Layers.IndexOf(layer);
            int to = from + offset;
            if (from < 0 || to < 0 || to >= Layers.Count) return false;

            Layers.RemoveAt(from);
            Layers.Insert(to, layer);
            ReindexLayers();
            return true;
        }

        /// <summary>
        /// A layer's Index keys its playback state: it has to equal the layer's position in the list.
        /// Two layers with the same index share one current state, and the second plays nothing.
        /// </summary>
        public void ReindexLayers()
        {
            for (int i = 0; i < Layers.Count; i++)
                Layers[i].Index = i;
        }

        /// <summary>
        /// Assigns a unique ID to a state (and any nested children) and adds it to a layer.
        /// </summary>
        public void AddState(AnimationLayer layer, AnimationState state)
        {
            AssignIds(state);
            layer.States.Add(state);
        }

        private void AssignIds(AnimationState state)
        {
            if (state.ID == 0)
                state.ID = _nextStateId++;

            int count = state.GetStateCount();
            for (int i = 0; i < count; i++)
                AssignIds(state.GetState(i));
        }

        /// <summary>
        /// Creates a transition between two states with IDs synced.
        /// </summary>
        public void AddTransition(AnimationLayer layer, AnimationTransition transition)
        {
            transition.SourceID = transition.Source.ID;
            transition.TargetID = transition.Target.ID;
            layer.Transitions.Add(transition);
        }

        /// <summary>
        /// Rebuild runtime state after deserialization.
        /// Resolves transition SourceID/TargetID to live AnimationState references.
        /// Assigns indices to layers and parameters.
        /// </summary>
        public void RebuildAfterLoad()
        {
            ReindexLayers();

            // Assign parameter indices
            for (int i = 0; i < Parameters.Count; i++)
                Parameters[i].Index = i;

            // First pass: find the highest existing ID so auto-assignment
            // doesn't collide with pre-serialized IDs (e.g. Jump=2, Falling=3).
            int maxId = 0;
            foreach (var layer in Layers)
                foreach (var state in layer.States)
                    FindMaxId(state, ref maxId);

            _nextStateId = maxId + 1;

            // Second pass: assign IDs to uninitialized states and build lookup
            var stateMap = new Dictionary<int, AnimationState>();

            foreach (var layer in Layers)
            {
                foreach (var state in layer.States)
                    CollectStates(state, stateMap);
            }

            // Resolve transition references
            foreach (var layer in Layers)
            {
                foreach (var transition in layer.Transitions)
                {
                    stateMap.TryGetValue(transition.SourceID, out transition.Source);
                    stateMap.TryGetValue(transition.TargetID, out transition.Target);
                }

                // Prune transitions with unresolvable states
                layer.Transitions.RemoveAll(t => t.Source == null || t.Target == null);
            }
        }

        /// <summary>
        /// Reassign parameter indices after editor modifications (add/remove params).
        /// </summary>
        public void InvalidateParamCache()
        {
            for (int i = 0; i < Parameters.Count; i++)
                Parameters[i].Index = i;
        }

        private static void FindMaxId(AnimationState state, ref int maxId)
        {
            if (state.ID > maxId)
                maxId = state.ID;

            int count = state.GetStateCount();
            for (int i = 0; i < count; i++)
                FindMaxId(state.GetState(i), ref maxId);
        }

        private void CollectStates(AnimationState state, Dictionary<int, AnimationState> map)
        {
            if (state.ID == 0)
                state.ID = _nextStateId++;

            map[state.ID] = state;

            int count = state.GetStateCount();
            for (int i = 0; i < count; i++)
                CollectStates(state.GetState(i), map);
        }
    }
}
