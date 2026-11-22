using System.Collections.Generic;
using Freefall.Base;
using Freefall.Graphics;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// The weather's patterns on this entity and everything under it follow its frame instead of the world: the snow's
    /// drifts, wind ripples, grain and crystals are anchored to the world, so on a ship or a cart they slid over it as it
    /// moved. Under a WeatherFrame they ride with it, one pattern over all its pieces (each piece takes this entity's frame,
    /// not its own, so there is no seam where two meet). Where the snow lies is still the world's to say, what faces up, but
    /// what keeps the fall off a place on it (its own castles and decks over it) comes from a cover map of its own, drawn
    /// once in its frame as it goes out of the sky captures (a ship leaving its berth), so its cover moves with it too
    /// (<see cref="WeatherFrames"/>, <see cref="ShelterMap"/>). <see cref="Refresh"/> draws that again.
    /// Its own transform and every one under it are entered in <see cref="WeatherFrames"/> as it wakes
    /// or is turned on; one under another WeatherFrame keeps that one's. What is added under it later is taken in by
    /// <see cref="Refresh"/>. Docs: .agent/knowledge/atmospheric_effects/weather.md "Snow on what moves".
    /// </summary>
    [Icon("icon_sky.png")]
    public class WeatherFrame : Component
    {
        private readonly List<int> _slots = new();

        protected override void Awake()
        {
            if (Enabled) Enter();
        }

        protected override void OnEnabledChanged()
        {
            if (!IsAwake) return;
            if (Enabled) Enter();
            else Leave();
        }

        public override void Destroy() => Leave();

        [Button("Refresh")]
        [Description("Take in the pieces added under this entity since it woke (its frame is entered for each) and draw its own cover map again")]
        public void Refresh()
        {
            Leave();
            if (Enabled) Enter();
        }

        private void Enter()
        {
            var root = Transform;
            if (root == null || IsDestroyed) return;
            int frame = root.TransformSlot;
            if (frame < 0) return;
            Walk(root, frame);
            // Its own cover map, drawn once all of it is out of the sky captures (WeatherFrames, ShelterMap)
            WeatherFrames.SetRoot(frame, root);
            _frame = frame;
        }

        private int _frame = -1;

        private void Walk(Transform t, int frame)
        {
            int slot = t.TransformSlot;
            if (slot >= 0)
            {
                WeatherFrames.Set(slot, frame);
                _slots.Add(slot);
            }
            for (int i = 0; i < t.Count; i++)
            {
                var child = t.GetChild(i);
                if (child == null) continue;
                var inner = child.Entity?.GetComponent<WeatherFrame>();
                if (inner != null && inner != this && inner.Enabled) continue;   // it keeps its own
                Walk(child, frame);
            }
        }

        private void Leave()
        {
            foreach (int slot in _slots)
                WeatherFrames.Remove(slot);
            _slots.Clear();
            if (_frame >= 0) WeatherFrames.RemoveRoot(_frame);
            _frame = -1;
        }
    }
}
