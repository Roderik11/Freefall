using System;
using System.Collections.Generic;
using Squid;
using Freefall.Animation;
using Freefall.Assets.Importers;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Inspects an animation clip: what it is, and what is marked on it (AnimationClip.Events).
    ///
    /// Events are found by looking: the clip is shown on a character in the scene (the one selected
    /// last, with what it holds in its hands) at the time of the slider, and "Add event here" marks
    /// that moment. They are saved with the file the clip is imported from (ModelImporter.Clips), at
    /// once, on every change; nothing is reimported.
    /// </summary>
    [GUIInspector(typeof(AnimationClip))]
    public class AnimationClipInspector : GUIInspector
    {
        private readonly AnimationClip _clip;
        private ValueSlider _scrub;
        private Label _at;
        private Label _on;
        private Button _play;

        public AnimationClipInspector(GUIObject target) : base(target)
        {
            _clip = target.Target as AnimationClip;
            if (_clip == null) return;

            ClipPreview.Open(_clip);

            AddCategory("Animation Clip");
            AddText("Name", _clip.Name ?? "(unnamed)");
            AddText("Length", $"{_clip.DurationSeconds:0.00} s, {_clip.Duration:0.#} frames");
            AddText("Frames a second", $"{_clip.TicksPerSecond:0.#}");
            AddText("Bones moved", _clip.Channels.Count.ToString());
            AddText("Rig in its file", _clip.Skeleton?.Name ?? "(none)");
            AddText("Guid", _clip.Guid ?? string.Empty);

            // ── Looking at it ──
            AddCategory("Look at it");

            _on = new Label
            {
                Dock = DockStyle.Fill,
                Tooltip = "The clip is shown on this character in the scene when the slider is moved or it plays: " +
                          "the character selected last. Select another one, then the clip again, to change it.",
            };
            AddRow("Shown on", _on);

            _scrub = new ValueSlider { Minimum = 0f, Maximum = 1f };
            _scrub.SetValue(ClipPreview.Time, notify: false);
            _scrub.ValueChanged += s =>
            {
                ClipPreview.Time = s.Value;
                ClipPreview.Showing = true;
                ClipPreview.Playing = false;
            };
            AddRow("Time", _scrub);

            _at = new Label { Dock = DockStyle.Fill };
            AddRow("That is", _at);

            _play = AddButton("Play", () =>
            {
                ClipPreview.Playing = !ClipPreview.Playing;
                ClipPreview.Showing = true;
            });

            AddButton("Add event here", () =>
            {
                var events = new List<AnimationEvent>(_clip.Events ?? new List<AnimationEvent>())
                {
                    new AnimationEvent { Name = "event", Time = ClipPreview.Time },
                };
                Change(events, rebuild: true);
            });

            // ── What is marked ──
            int count = _clip.Events?.Count ?? 0;
            AddCategory(count == 1 ? "1 Event" : $"{count} Events");

            for (int i = 0; i < count; i++)
                AddEvent(_clip.Events[i], i);

            RefreshLabels();
        }

        private void AddEvent(AnimationEvent evt, int index)
        {
            if (evt == null) return;
            var obj = new GUIObject(evt);

            // The name, with what removes the event at the end of its row
            var name = obj.GetProperty(nameof(AnimationEvent.Name));
            name.OnValueChanged += Save;
            var row = GetInspectorElement(name, $"Event {index + 1}");
            if (row is GUIInspectorRow inspectorRow)
            {
                var remove = new Button
                {
                    Style = "iconclose",
                    Size = new Point(32, 32),
                    Margin = new Margin(1),
                    Dock = DockStyle.Fill,
                    Tooltip = "Remove this event",
                };
                remove.MouseClick += (s, e) =>
                {
                    if (e.Button > 0) return;
                    var events = new List<AnimationEvent>(_clip.Events);
                    events.Remove(evt);
                    Change(events, rebuild: true);
                };
                inspectorRow.ExtraButton.Controls.Add(remove);
            }
            AddControl(row);

            var time = obj.GetProperty(nameof(AnimationEvent.Time));
            time.OnValueChanged += () =>
            {
                // Dragging an event's time shows that moment, the same as the slider above
                ClipPreview.Time = evt.Time;
                ClipPreview.Showing = true;
                ClipPreview.Playing = false;
                _scrub?.SetValue(evt.Time, notify: false);
                ClipPreview.SaveSoon(_clip);
            };
            AddProperty(time, "    when (0..1)");

            var bone = obj.GetProperty(nameof(AnimationEvent.Bone));
            bone.OnValueChanged += Save;
            AddProperty(bone, "    where");
        }

        /// <summary>Put another list of events on the clip (never the same list changed: characters read it while they play) and keep it.</summary>
        private void Change(List<AnimationEvent> events, bool rebuild)
        {
            events.Sort(static (a, b) => a.Time.CompareTo(b.Time));
            _clip.Events = events;
            Save();

            if (rebuild)
                MessageDispatcher.Send(Msg.RefreshInspector, _clip);
        }

        private void Save()
        {
            ClipPreview.SaveSoon(_clip);
            if (!ClipPreview.Flush())
                Debug.LogWarning("AnimationClip", $"The events of '{_clip.Name}' could not be saved: it does not come from a model file");
        }

        private Button AddButton(string text, Action onClick)
        {
            var button = new Button
            {
                Text = text,
                Style = "button",
                Size = new Point(26, 32),
                Margin = new Margin(8, 4, 8, 4),
                Dock = DockStyle.Top,
            };

            button.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                onClick();
            };

            AddControl(button);
            return button;
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
            if (_clip == null || _scrub == null) return;

            float length = _clip.DurationSeconds;

            if (ClipPreview.Playing && length > 0f)
            {
                float time = ClipPreview.Time + Time.Delta / length;
                ClipPreview.Time = time - MathF.Floor(time);
                _scrub.SetValue(ClipPreview.Time, notify: false);
            }

            RefreshLabels();
            ClipPreview.Tick(Time.Delta);

            // Asked for again every frame: the character goes back to what it was doing by itself
            // as soon as this inspector is gone
            if (ClipPreview.Showing)
                ClipPreview.Character?.Preview(_clip, ClipPreview.Time * length);
        }

        private void RefreshLabels()
        {
            float length = _clip.DurationSeconds;
            float time = ClipPreview.Time;

            string at = $"{time * length:0.00} s, frame {time * _clip.Duration:0.#}";
            if (_at.Text != at) _at.Text = at;

            // The character it is (or will be, once the slider is moved or it plays) shown on
            var character = ClipPreview.Character;
            string on = character?.Entity != null ? character.Entity.Name : "(no character)";
            if (_on.Text != on) _on.Text = on;

            string play = ClipPreview.Playing ? "Pause" : "Play";
            if (_play.Text != play) _play.Text = play;
        }
    }

    /// <summary>
    /// Where the clip inspector shows a clip, and what of it: kept outside the inspector, which is made
    /// anew whenever the list of events changes.
    /// </summary>
    internal static class ClipPreview
    {
        private static WeakReference<Entity> s_character;
        private static string s_clip;

        /// <summary>The moment shown, as a fraction of the clip's length.</summary>
        public static float Time;

        /// <summary>The clip is put on the character (after the slider was moved or play was pressed).</summary>
        public static bool Showing;
        public static bool Playing;

        // A clip whose events were changed by dragging and are not written yet: a drag changes the
        // value many times a second, the file is written once it has come to rest
        private static AnimationClip s_unsaved;
        private static float s_saveIn;

        public static void SaveSoon(AnimationClip clip)
        {
            if (s_unsaved != null && !ReferenceEquals(s_unsaved, clip)) Flush();
            s_unsaved = clip;
            s_saveIn = 0.4f;
        }

        public static void Tick(float seconds)
        {
            if (s_unsaved == null) return;
            s_saveIn -= seconds;
            if (s_saveIn <= 0f) Flush();
        }

        /// <summary>Write what is waiting to be written. Returns false when that failed.</summary>
        public static bool Flush()
        {
            var clip = s_unsaved;
            s_unsaved = null;
            return clip == null || ModelImporter.SetClipEvents(clip);
        }

        /// <summary>A selection was made: a character among it is the one clips are looked at on from now on.</summary>
        public static void Note(IEnumerable<Entity> selection)
        {
            // The inspector that was open is about to go
            Flush();

            foreach (var entity in selection)
            {
                if (AnimatorOf(entity) == null) continue;
                s_character = new WeakReference<Entity>(entity);
                return;
            }
        }

        /// <summary>An inspector opens on a clip: another clip than last time starts at its beginning, not shown yet.</summary>
        public static void Open(AnimationClip clip)
        {
            string id = clip.Guid ?? clip.Name;
            if (s_clip == id) return;

            s_clip = id;
            Time = 0f;
            Showing = false;
            Playing = false;
        }

        /// <summary>The character to show clips on: the one selected last, else any in the scene.</summary>
        public static Animator Character
        {
            get
            {
                if (s_character != null && s_character.TryGetTarget(out var entity) && entity != null)
                {
                    var animator = AnimatorOf(entity);
                    if (animator != null && !animator.IsDestroyed) return animator;
                }

                var all = ComponentCache<Animator>.All;
                for (int i = 0; i < all.Count; i++)
                {
                    var animator = all[i];
                    if (animator?.Entity == null || animator.IsDestroyed || animator.Animation == null) continue;
                    s_character = new WeakReference<Entity>(animator.Entity);
                    return animator;
                }
                return null;
            }
        }

        // The Animator of an entity, or of what it is part of (a mesh of a character was picked)
        private static Animator AnimatorOf(Entity entity)
        {
            for (var e = entity; e != null; e = e.Transform?.Parent?.Entity)
            {
                var animator = e.GetComponent<Animator>();
                if (animator != null) return animator;
            }
            return null;
        }
    }
}
