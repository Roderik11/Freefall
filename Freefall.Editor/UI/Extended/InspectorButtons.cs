using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reflection;
using System.Text;
using Squid;
using Freefall.Reflection;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    /// <summary>
    /// The buttons a type shows in the inspector: its public methods without parameters marked
    /// <see cref="ButtonAttribute"/>, in the order they are declared (base types first). The inspector draws
    /// them (<see cref="Build"/>), the MCP presses them by name (<see cref="Find"/>, <see cref="Press"/>).
    /// Nothing is cached per type: script types are replaced on every hot reload, and a static cache keyed
    /// by them would keep the old script assembly alive.
    /// </summary>
    public static class InspectorButtons
    {
        const int IconSize = 16;

        public sealed class Entry
        {
            public MethodInfo Method;
            public ButtonAttribute Button;
            /// <summary>The method's [Category]; null = after all the fields.</summary>
            public string Category;
            /// <summary>The method's [Description].</summary>
            public string Tooltip;

            /// <summary>What it says: the attribute's text, or the method's name in words.</summary>
            public string Text => Button.Text ?? Words(Method.Name);
        }

        public static List<Entry> Of(Type type)
        {
            var list = new List<Entry>();
            if (type == null) return list;

            foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance))
            {
                var button = method.GetCustomAttribute<ButtonAttribute>();
                if (button == null || method.ContainsGenericParameters || method.GetParameters().Length > 0) continue;

                list.Add(new Entry
                {
                    Method = method,
                    Button = button,
                    Category = method.GetCustomAttribute<CategoryAttribute>()?.Category,
                    Tooltip = method.GetCustomAttribute<DescriptionAttribute>()?.Description,
                });
            }

            list.Sort((a, b) =>
            {
                int da = Depth(a.Method.DeclaringType), db = Depth(b.Method.DeclaringType);
                return da != db ? da.CompareTo(db) : a.Method.MetadataToken.CompareTo(b.Method.MetadataToken);
            });
            return list;
        }

        /// <summary>A type's button by its method name or by what it says (case-insensitive); null if it has none such.</summary>
        public static Entry Find(Type type, string name)
        {
            foreach (var entry in Of(type))
            {
                if (string.Equals(entry.Method.Name, name, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(entry.Text, name, StringComparison.OrdinalIgnoreCase))
                    return entry;
            }
            return null;
        }

        /// <summary>Calls the button's method on each target of its type. A failure is logged and the others still run. Returns how many ran.</summary>
        public static int Press(Entry entry, System.Collections.IEnumerable targets)
        {
            int pressed = 0;
            foreach (var target in targets)
            {
                if (target == null || !entry.Method.DeclaringType.IsInstanceOfType(target)) continue;
                try
                {
                    entry.Method.Invoke(target, null);
                    pressed++;
                }
                catch (TargetInvocationException ex)
                {
                    var inner = ex.InnerException ?? ex;
                    Debug.LogError($"[Button] {entry.Method.DeclaringType.Name}.{entry.Method.Name}: {inner.Message}", inner.StackTrace);
                }
            }
            return pressed;
        }

        /// <summary>
        /// The controls for some of a type's buttons, in their order: a button without a row gets a line of its
        /// own across the inspector; the buttons of a row go into one flow, at the place of the row's first one.
        /// </summary>
        public static List<Control> Build(IEnumerable<Entry> entries, GUIObject target)
        {
            var controls = new List<Control>();
            var rows = new Dictionary<string, FlowLayoutFrame>();

            foreach (var entry in entries)
            {
                string rowName = entry.Button.Row;
                if (string.IsNullOrEmpty(rowName))
                {
                    controls.Add(MakeButton(entry, target, false));
                    continue;
                }

                if (!rows.TryGetValue(rowName, out var row))
                {
                    row = new FlowLayoutFrame
                    {
                        Dock = DockStyle.Top,
                        AutoSize = AutoSize.Vertical,
                        FlowDirection = FlowDirection.LeftToRight,
                        HSpacing = 4,
                        VSpacing = 4,
                        Margin = new Margin(4, 0, 4, 0),   // + the flow's own spacing: in line with full-width buttons
                    };
                    rows.Add(rowName, row);
                    controls.Add(row);
                }

                var button = MakeButton(entry, target, true);
                // A button finds its width from its text once it is laid out, after the flow placed it
                var flow = row;
                button.SizeChanged += sender => flow.ForceFlowLayout();
                row.Controls.Add(button);
            }

            return controls;
        }

        static Button MakeButton(Entry entry, GUIObject target, bool inRow)
        {
            string text = entry.Text;
            bool hasText = !string.IsNullOrEmpty(text);
            bool hasIcon = !string.IsNullOrEmpty(entry.Button.Icon);

            var button = new Button
            {
                Text = hasText ? text : null,
                Style = hasIcon && hasText ? "inspectorIconButton" : "inspectorButton",
                // An icon alone says what it does in its tooltip
                Tooltip = entry.Tooltip ?? (hasText ? null : Words(entry.Method.Name)),
                Size = new Point(28, 28),
            };

            if (!inRow)
            {
                button.Dock = DockStyle.Top;
                button.Size = new Point(28, 30);
                button.Margin = new Margin(8, 4, 8, 4);
            }
            else if (hasText)
                button.AutoSize = AutoSize.Horizontal;

            if (hasIcon)
            {
                // Scaled into a square of IconSize, whatever the image's own size (the editor's icons are
                // 32 and 64 px: drawn as they are, a 64 px one filled the button with its middle)
                // An icon alone in a row fills its square button; otherwise it stands at the left
                int around = (button.Size.y - IconSize) / 2;
                bool alone = !hasText && inRow;
                button.GetElements().Add(new ImageControl
                {
                    Texture = entry.Button.Icon,
                    Tiling = TextureMode.Stretch,
                    Color = ColorInt.ARGB(1f, .85f, .85f, .85f),
                    NoEvents = true,
                    Size = new Point(IconSize, IconSize),
                    Dock = alone ? DockStyle.Fill : DockStyle.Left,
                    Margin = alone
                        ? new Margin((button.Size.x - IconSize) / 2, around, (button.Size.x - IconSize) / 2, around)
                        : new Margin(8, around, 0, around),
                });
            }

            button.MouseClick += (sender, args) =>
            {
                if (args.Button > 0) return;
                Press(entry, target.Targets);
            };

            return button;
        }

        /// <summary>"FireImpulse" → "Fire Impulse", "HTMLExport" → "HTML Export".</summary>
        public static string Words(string name)
        {
            var words = new StringBuilder(name.Length + 4);
            for (int i = 0; i < name.Length; i++)
            {
                char c = name[i];
                if (i > 0 && char.IsUpper(c) &&
                    (char.IsLower(name[i - 1]) || (char.IsUpper(name[i - 1]) && i + 1 < name.Length && char.IsLower(name[i + 1]))))
                    words.Append(' ');
                words.Append(c);
            }
            return words.ToString();
        }

        static int Depth(Type? type)
        {
            int depth = 0;
            for (var t = type; t != null; t = t.BaseType) depth++;
            return depth;
        }
    }
}
