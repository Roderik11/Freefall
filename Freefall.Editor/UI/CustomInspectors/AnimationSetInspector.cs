using Freefall.Animation;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Inspects an AnimationSet as named clip slots: one per state of its base graph, grouped by layer,
    /// blend tree children under the tree's name. An empty slot shows the graph's own clip in brackets,
    /// which is what the set plays there.
    /// </summary>
    [GUIInspector(typeof(AnimationSet))]
    public class AnimationSetInspector : GUIInspector
    {
        private readonly AnimationSet _set;
        private readonly Field _clipField = new Field(typeof(Slot).GetProperty(nameof(Slot.Clip)));

        public AnimationSetInspector(GUIObject target) : base(target)
        {
            _set = target.Target as AnimationSet;

            AddCategory("Animation Set");
            AddProperty(target.GetProperty(nameof(AnimationSet.Name)));

            // The slots are those of the base graph: list them again when it changes
            var baseProperty = target.GetProperty(nameof(AnimationSet.Base));
            baseProperty.OnValueChanged += () => MessageDispatcher.Send(Msg.RefreshInspector);
            AddProperty(baseProperty);
            AddProperty(target.GetProperty(nameof(AnimationSet.FingerWeight)));

            if (_set?.Base == null) return;

            foreach (var layer in _set.Base.Layers)
            {
                AddCategory(string.IsNullOrEmpty(layer.Name) ? "Layer" : layer.Name);
                foreach (var state in layer.States)
                    AddSlots(target, state);
            }
        }

        private void AddSlots(GUIObject target, AnimationState state)
        {
            int children = state.GetStateCount();
            if (children > 0)
            {
                AddHeader((state.Name ?? "Blend Tree").ToUpperInvariant());
                for (int i = 0; i < children; i++)
                    AddSlots(target, state.GetState(i));
                return;
            }

            var property = new GUIProperty(target, _clipField, new Slot(_set, state));
            var row = GetInspectorElement(property, state.Name ?? "State");

            if (row is GUIInspectorRow inspectorRow && inspectorRow.Content.Controls.Count > 0
                && inspectorRow.Content.Controls[0] is AssetProperty field)
                field.EmptyText = $"({state.Clip?.Name ?? "no clip"})";

            AddControl(row);
        }

        /// <summary>One state of the base graph as a clip slot of the set (empty = the graph's own clip).</summary>
        private sealed class Slot
        {
            private readonly AnimationSet _set;
            private readonly AnimationState _state;

            public Slot(AnimationSet set, AnimationState state)
            {
                _set = set;
                _state = state;
            }

            public AnimationClip Clip
            {
                get => _set.Overrides.Find(o => o != null && o.StateID == _state.ID)?.Clip;
                set
                {
                    var entry = _set.Overrides.Find(o => o != null && o.StateID == _state.ID);
                    if (value == null)
                    {
                        if (entry != null) _set.Overrides.Remove(entry);
                    }
                    else if (entry != null)
                        entry.Clip = value;
                    else
                        _set.Overrides.Add(new AnimationSet.ClipOverride { StateID = _state.ID, Clip = value });

                    // Characters playing this set pick the change up on their next frame
                    _set.Invalidate();
                }
            }
        }
    }
}
