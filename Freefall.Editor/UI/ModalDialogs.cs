using System;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// The landing page's modal popups, on any desktop: the progress popup of a long job (opening a project, publishing
    /// a game), which follows the job by itself, and a message with buttons in the same look. The landing styles go
    /// into the desktop's skin the first time (the editor desktop has a skin of its own).
    /// </summary>
    internal static class ModalDialogs
    {
        /// <summary>What a progress popup shows: the title, the step line, the bar (0..1) and what the step is doing.</summary>
        public sealed record Progress(string Title, string Step, string Detail, float Fraction);

        /// <summary>A modal progress popup that reads <paramref name="follow"/> every frame (null: keep what it shows). Close it with Close().</summary>
        public static Window ShowProgress(Desktop desktop, Func<Progress> follow)
        {
            Prepare(desktop);
            var dialog = new ProgressWindow(follow);
            dialog.Show(desktop);
            return dialog;
        }

        /// <summary>
        /// A modal message: title, wrapped text and buttons, the first one the primary on the right. Every button closes
        /// the popup, then runs its action (null: only closes).
        /// </summary>
        public static Window ShowMessage(Desktop desktop, string title, string message, params (string Text, Action Click)[] buttons)
        {
            Prepare(desktop);

            const int width = 560;
            var window = new Window
            {
                Style = "frame",
                Dock = DockStyle.Center,
                Modal = true,
                Padding = new Margin(24, 18, 24, 18),
            };

            // Wrapped here: Squid wraps only at spaces, and the message is often a long path
            int textWidth = width - window.Padding.Left - window.Padding.Right;
            var text = LandingArt.Wrap(LandingArt.FontBody, message, textWidth);
            int lines = text.Split('\n').Length;
            int lineHeight = LandingArt.Measure(LandingArt.FontBody, "Ag").y + 4;
            window.Size = new Point(width, Math.Clamp(18 + 30 + 16 + lines * lineHeight + 36 + 18, 200, 640));

            window.Controls.Add(new Label
            {
                Text = title,
                Style = "landing.brand",
                Size = new Point(100, 30),
                Dock = DockStyle.Top,
            });

            var row = new Frame
            {
                Size = new Point(100, 36),
                Dock = DockStyle.Bottom,
            };
            for (int i = 0; i < buttons.Length; i++)
            {
                var (label, click) = buttons[i];
                var button = new Button
                {
                    Text = label,
                    Style = i == 0 ? "landing.primary" : "landing.ghost",
                    Size = new Point(Math.Max(100, LandingArt.Measure(LandingArt.FontBodyMedium, label).x + 40), 36),
                    Dock = DockStyle.Right,
                    Margin = new Margin(0, 0, i == 0 ? 0 : 10, 0),
                    Cursor = Cursors.Link,
                };
                button.MouseClick += (s, e) =>
                {
                    window.Close();
                    click?.Invoke();
                };
                row.Controls.Add(button);
            }
            window.Controls.Add(row);

            window.Controls.Add(new Label
            {
                Text = text,
                Style = "landing.sub",
                TextAlign = Alignment.TopLeft,
                Dock = DockStyle.Fill,
                Margin = new Margin(0, 8, 0, 8),
            });

            window.Show(desktop);
            return window;
        }

        private static void Prepare(Desktop desktop)
        {
            LandingArt.Ensure();
            if (desktop.Skin != null && !desktop.Skin.ContainsKey("landing.brand"))
                LandingArt.RegisterStyles(desktop.Skin);
        }

        private sealed class ProgressWindow : Window
        {
            private readonly Func<Progress> _follow;
            private readonly Label _title, _step, _detail;
            private readonly ProgressBar _bar;
            private Progress _shown;

            public ProgressWindow(Func<Progress> follow)
            {
                _follow = follow;
                Style = "frame";
                Size = new Point(560, 150);
                Dock = DockStyle.Center;
                Modal = true;
                Padding = new Margin(24, 18, 24, 18);

                _title = new Label { Style = "landing.brand", Size = new Point(100, 30), Dock = DockStyle.Top };
                _step = new Label { Style = "landing.body", Size = new Point(100, 24), Dock = DockStyle.Top };
                _bar = new ProgressBar { Size = new Point(100, 6), Dock = DockStyle.Top, Margin = new Margin(0, 8, 0, 8) };
                _detail = new Label { Style = "landing.small", Dock = DockStyle.Fill };
                Controls.Add(_title);
                Controls.Add(_step);
                Controls.Add(_bar);
                Controls.Add(_detail);
                Display(_follow());
            }

            protected override void OnUpdate()
            {
                base.OnUpdate();
                Display(_follow());
            }

            /// <summary>The job's latest progress (its workers write a new record per change; equal records draw nothing new).</summary>
            private void Display(Progress progress)
            {
                if (progress == null || progress == _shown) return;
                _shown = progress;

                _title.Text = progress.Title;
                _step.Text = progress.Step;
                _detail.Text = LandingArt.Fit(LandingArt.FontSmall, progress.Detail ?? "", Math.Max(40, _detail.Size.x), keepEnd: true);
                _bar.Value = progress.Fraction;
            }
        }

        /// <summary>A thin bar: the share of the job done.</summary>
        private sealed class ProgressBar : Control
        {
            public float Value;

            protected override void DrawStyle(Style style, float opacity)
            {
                int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
                if (opacity == 0 || w <= 0 || h <= 0) return;

                Gui.Renderer.DrawBox(x, y, w, h, LandingArt.Well);
                int fill = (int)(w * Math.Clamp(Value, 0f, 1f));
                if (fill > 0)
                    Gui.Renderer.DrawBox(x, y, fill, h, LandingArt.Coral);
            }
        }
    }
}
