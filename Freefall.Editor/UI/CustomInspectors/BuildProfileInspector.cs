using Squid;
using Freefall.Base;
using Freefall.Editor.Commands;
using Freefall.Editor.Publishing;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// A build profile: its fields and buttons (as the generic inspector shows them), then its textures against the
    /// budget from the profile's latest scan (BuildScanJob): a folder tree in a virtual list with a check box on every
    /// row (a folder's checks or clears everything under it), filters (sharp enough at the budget, too coarse, no use
    /// found, capped) and a search. A scan comes with the suggestion checked: every texture that stays sharp enough at
    /// the budget (until the checks are changed; Check suggested puts it back). Cap and Uncap run on what is checked
    /// (TextureCaps, with the progress popup).
    /// </summary>
    [GUIInspector(typeof(BuildProfile))]
    public class BuildProfileInspector : GenericInspector
    {
        public BuildProfileInspector(GUIObject target) : base(target)
        {
            if (target.Target is not BuildProfile profile) return;
            AddHeader("TEXTURE BUDGET");
            AddControl(new TextureBudgetTree(profile));
        }
    }

    /// <summary>The textures of a profile's latest scan against its budget, as a checkable folder tree.</summary>
    internal sealed class TextureBudgetTree : Frame
    {
        enum Filter { All, SharpEnough, TooCoarse, NoUse, Capped }

        sealed class Node
        {
            public string Name = "", Path = "";
            public Node? Parent;
            public readonly List<Node> Children = new();
            public BudgetTexture? Texture;
            public int Depth;
            // Folders: the textures under them
            public int Count, Coarse, CappedCount;
            public long Bytes, Saving;
            public IEnumerable<BudgetTexture> Textures => Texture != null ? new[] { Texture } : Children.SelectMany(c => c.Textures);
        }

        const int RowHeight = 40;
        static readonly int Accent = ColorInt.ARGB(1f, 1f, .47f, .37f);

        readonly BuildProfile _profile;
        readonly Label _summary1, _summary2, _summary3;
        readonly FlowLayoutFrame _filters, _actions;
        readonly SearchBox _search;
        readonly VirtualList _list;
        readonly Button _cap, _uncap, _suggest, _clear;
        readonly HashSet<string> _checked = new(StringComparer.OrdinalIgnoreCase);
        readonly HashSet<string> _expanded = new(StringComparer.OrdinalIgnoreCase);
        readonly List<Node> _visible = new();
        readonly Dictionary<Filter, Button> _filterButtons = new();

        Filter _filter = Filter.All;
        int _version = -1;
        TextureBudgetResult? _result;
        Node _root = new();
        bool _expandedOnce;
        // The checks were changed since the suggestion was put on them: a new scan keeps them then
        bool _touched;

        public TextureBudgetTree(BuildProfile profile)
        {
            _profile = profile;
            Dock = DockStyle.Top;
            AutoSize = AutoSize.Vertical;
            Padding = new Margin(8, 0, 8, 4);

            // The editor's fonts are ASCII: no arrows or middle dots in what it shows
            _summary1 = new Label { Dock = DockStyle.Top, Size = new Point(100, 20), TextAlign = Alignment.MiddleLeft };
            _summary2 = new Label { Dock = DockStyle.Top, Size = new Point(100, 20), TextAlign = Alignment.MiddleLeft, UseTextColor = true, TextColor = Accent };
            _summary3 = new Label { Dock = DockStyle.Top, Size = new Point(100, 20), Style = "stateSubtitle", TextAlign = Alignment.MiddleLeft };
            Controls.Add(_summary1);
            Controls.Add(_summary2);
            Controls.Add(_summary3);

            _filters = Flow();
            foreach (var f in Enum.GetValues<Filter>())
            {
                var button = new Button { Style = "inspectorButton", Size = new Point(28, 24), AutoSize = AutoSize.Horizontal, Tag = f };
                button.MouseClick += (s, e) => { if (e.Button == 0) { _filter = (Filter)button.Tag; Rebuild(); } };
                var flow = _filters;
                button.SizeChanged += sender => flow.ForceFlowLayout();
                _filterButtons[f] = button;
                _filters.Controls.Add(button);
            }
            Controls.Add(_filters);

            _search = new SearchBox { Dock = DockStyle.Top, Size = new Point(100, 24), Margin = new Margin(0, 4, 0, 4) };
            _search.TextChanged += sender => Rebuild();
            Controls.Add(_search);

            _list = new VirtualList
            {
                Dock = DockStyle.Top,
                Size = new Point(100, 460),
                ItemHeight = RowHeight,
            };
            _list.Scrollbar.ButtonDown.Visible = false;
            _list.Scrollbar.ButtonUp.Visible = false;
            _list.Scrollbar.Slider.Ease = false;
            _list.Scrollbar.Slider.MinHandleSize = 64;
            _list.CreateItem = index => { var row = new Row(this); Bind(row, index); return row; };
            _list.BindItem = (control, index) => Bind((Row)control, index);
            Controls.Add(_list);

            _actions = Flow();
            _cap = ActionButton(() => Apply(cap: true));
            _cap.UseTextColor = true;
            _cap.TextColor = Accent;
            _uncap = ActionButton(() => Apply(cap: false));
            _uncap.Tooltip = "Back to their source size (Max Size: Source) for the checked textures that are capped";
            _suggest = ActionButton(() => { Suggest(); UpdateActions(); _list.Refresh(); });
            _suggest.Text = "Check suggested";
            _suggest.Tooltip = "Check the suggestion again: every texture over the budget that stays sharp enough capped";
            _clear = ActionButton(() => { _checked.Clear(); _touched = true; UpdateActions(); _list.Refresh(); });
            _clear.Text = "Check none";
            _clear.Tooltip = "Clear every check";
            Controls.Add(_actions);

            Rebuild();
        }

        FlowLayoutFrame Flow() => new()
        {
            Dock = DockStyle.Top,
            AutoSize = AutoSize.Vertical,
            FlowDirection = FlowDirection.LeftToRight,
            HSpacing = 4,
            VSpacing = 4,
            Margin = new Margin(0, 4, 0, 0),
        };

        Button ActionButton(Action click)
        {
            var button = new Button { Style = "inspectorButton", Size = new Point(28, 26), AutoSize = AutoSize.Horizontal };
            button.MouseClick += (s, e) => { if (e.Button == 0) click(); };
            var flow = _actions;
            button.SizeChanged += sender => flow.ForceFlowLayout();
            _actions.Controls.Add(button);
            return button;
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
            // A new scan (the Scan button, Publish, a cap) shows as soon as it is kept
            if (BuildScanJob.Version != _version) Rebuild();
        }

        // ── The tree ──

        bool Matches(BudgetTexture t, Filter filter) => filter switch
        {
            Filter.SharpEnough => t.OverBudget(_result!.Budget) && !t.Capped && !double.IsNaN(t.CoarsestAtBudget) && !_result.BelowTarget(t),
            Filter.TooCoarse => t.OverBudget(_result!.Budget) && _result.BelowTarget(t),
            Filter.NoUse => t.OverBudget(_result!.Budget) && double.IsNaN(t.Coarsest),
            Filter.Capped => t.Capped,
            _ => true,
        };

        void Rebuild()
        {
            _version = BuildScanJob.Version;
            _result = BuildScanJob.Latest(_profile)?.Textures;
            var rows = _result?.Rows ?? new List<BudgetTexture>();

            if (_result == null)
            {
                _summary1.Text = _profile.TextureBudget > 0 ? "Scan the profile to list its textures" : "TextureBudget is 0: no list";
                _summary2.Text = _summary3.Text = "";
            }
            else
            {
                var over = _result.Over;
                _summary1.Text = $"{over.Count} over {_result.Budget}: {BuildScan.Size(over.Sum(t => t.Artifact.Size))} -> {BuildScan.Size(over.Sum(t => t.CappedBytes))}";
                var suggested = rows.Where(t => Matches(t, Filter.SharpEnough)).ToList();
                _summary2.Text = $"Suggested: cap {suggested.Count}, saves {BuildScan.Size(suggested.Sum(t => t.Saving))}";
                _summary2.Tooltip = $"They stay above {_result.Target:0} px/m capped at {_result.Budget}. They are checked: press Cap";
                _summary3.Text = $"{over.Count(_result.BelowTarget)} too coarse at {_result.Budget}, {over.Count(t => double.IsNaN(t.Coarsest))} no use found, {rows.Count(t => t.Capped)} capped";
            }

            string Label(Filter f) => f switch
            {
                Filter.SharpEnough => "Suggested", Filter.TooCoarse => "Too coarse", Filter.NoUse => "No use found", Filter.Capped => "Capped", _ => "All",
            };
            foreach (var (f, button) in _filterButtons)
            {
                int count = _result == null ? 0 : rows.Count(t => Matches(t, f));
                button.Text = $"{Label(f)} {count}";
                // The filter shown: its text in the accent colour
                button.UseTextColor = f == _filter;
                button.TextColor = Accent;
                button.Tooltip = f switch
                {
                    Filter.SharpEnough => $"Over the budget and still above {_result?.Target ?? _profile.TexelDensityTarget:0} px/m capped: safe to cap",
                    Filter.TooCoarse => "Would show below the target capped: candidates for kit pieces, tiling materials or trim sheets, not a cap",
                    Filter.NoUse => "On nothing the scan could read (UI, particles, stamps, code): judge them yourself",
                    Filter.Capped => "Capped already: Uncap puts them back",
                    _ => "Everything over the budget, and what is capped",
                };
            }

            // Folders from the source paths, under what the filter and the search let through
            string search = _search.Text?.Trim() ?? "";
            _root = new Node { Depth = -1 };
            foreach (var t in rows.Where(t => _result != null && Matches(t, _filter)))
            {
                var path = t.Artifact.SourcePath.Length > 0 ? t.Artifact.SourcePath : Path.Combine("(no source)", Path.GetFileName(t.Artifact.CachePath));
                if (search.Length > 0 && path.IndexOf(search, StringComparison.OrdinalIgnoreCase) < 0) continue;

                var parts = path.Split('\\', '/');
                var node = _root;
                for (int i = 0; i < parts.Length - 1; i++)
                {
                    var folderPath = string.Join("\\", parts, 0, i + 1);
                    var next = node.Children.Find(c => c.Texture == null && c.Name == parts[i]);
                    if (next == null)
                    {
                        next = new Node { Name = parts[i], Path = folderPath, Parent = node, Depth = node.Depth + 1 };
                        node.Children.Add(next);
                    }
                    node = next;
                }
                node.Children.Add(new Node { Name = parts[^1], Path = path, Parent = node, Depth = node.Depth + 1, Texture = t });
            }
            Aggregate(_root);
            Sort(_root);

            // A new scan: the suggestion, unless the checks were changed by hand since it was last put on
            if (!_touched) Suggest();
            else if (_result != null)
            {
                var listed = new HashSet<string>(rows.Select(t => t.Artifact.Guid), StringComparer.OrdinalIgnoreCase);
                _checked.RemoveWhere(g => !listed.Contains(g));
            }

            // The top folders open the first time; later the open ones stay open
            if (!_expandedOnce && _root.Children.Count > 0)
            {
                _expandedOnce = true;
                foreach (var top in _root.Children) _expanded.Add(top.Path);
            }

            Flatten();
            UpdateActions();
        }

        static void Aggregate(Node node)
        {
            if (node.Texture is { } t)
            {
                node.Count = 1;
                node.Bytes = t.Artifact.Size;
                node.Saving = t.Saving;
                return;
            }
            foreach (var c in node.Children)
            {
                Aggregate(c);
                node.Count += c.Count;
                node.Bytes += c.Bytes;
                node.Saving += c.Saving;
            }
        }

        // Folders first, the biggest saving first; files the same
        static void Sort(Node node)
        {
            node.Children.Sort((a, b) =>
            {
                int folder = (a.Texture == null ? 0 : 1).CompareTo(b.Texture == null ? 0 : 1);
                if (folder != 0) return folder;
                int saving = b.Saving.CompareTo(a.Saving);
                return saving != 0 ? saving : b.Bytes.CompareTo(a.Bytes);
            });
            foreach (var c in node.Children) Sort(c);
        }

        void Flatten()
        {
            _visible.Clear();
            void Add(Node node)
            {
                foreach (var c in node.Children)
                {
                    _visible.Add(c);
                    if (c.Texture == null && _expanded.Contains(c.Path)) Add(c);
                }
            }
            Add(_root);
            _list.DataSource = _visible;
            _list.Refresh();
        }

        // ── Checks and actions ──

        /// <summary>All of a node's textures checked (true), none (false), some (null).</summary>
        bool? CheckState(Node node)
        {
            int on = 0, all = 0;
            foreach (var t in node.Textures)
            {
                all++;
                if (_checked.Contains(t.Artifact.Guid)) on++;
            }
            return on == 0 ? false : on == all ? true : null;
        }

        /// <summary>Check what the scan suggests capping: over the budget, not capped yet, and still sharp enough capped.</summary>
        void Suggest()
        {
            _checked.Clear();
            _touched = false;
            if (_result == null) return;
            foreach (var t in _result.Rows.Where(t => Matches(t, Filter.SharpEnough)))
                _checked.Add(t.Artifact.Guid);
        }

        void ToggleCheck(Node node)
        {
            _touched = true;
            bool check = CheckState(node) != true;
            foreach (var t in node.Textures)
            {
                if (check) _checked.Add(t.Artifact.Guid);
                else _checked.Remove(t.Artifact.Guid);
            }
            UpdateActions();
            _list.Refresh();
        }

        void ToggleExpanded(Node node)
        {
            if (node.Texture != null) return;
            if (!_expanded.Remove(node.Path)) _expanded.Add(node.Path);
            Flatten();
        }

        /// <summary>The checked textures the filter and the search show: what Cap and Uncap act on.</summary>
        List<BudgetTexture> CheckedTextures() =>
            _root.Textures.Where(t => _checked.Contains(t.Artifact.Guid)).ToList();

        void UpdateActions()
        {
            var picked = CheckedTextures();
            int budget = _profile.TextureBudget;
            int toCap = picked.Count(t => t.OverBudget(budget) && !t.Capped);
            int toUncap = picked.Count(t => t.Capped);
            bool isSuggestion = _result != null && _checked.SetEquals(_result.Rows.Where(t => Matches(t, Filter.SharpEnough)).Select(t => t.Artifact.Guid));
            _cap.Text = $"Cap {toCap} at {budget}" + (isSuggestion && toCap > 0 ? " (suggested)" : "");
            _cap.Tooltip = $"Max Size {budget} on the checked textures over it that the list shows; they reimport, then the profile is scanned again. Asks first";
            _cap.Enabled = toCap > 0 && budget > 0;
            _uncap.Text = $"Uncap {toUncap}";
            _uncap.Enabled = toUncap > 0;
            _clear.Enabled = _checked.Count > 0;
            _suggest.Enabled = _result != null;
        }

        void Apply(bool cap)
        {
            int budget = _profile.TextureBudget;
            var picked = CheckedTextures().Where(t => cap ? t.OverBudget(budget) && !t.Capped : t.Capped).ToList();
            if (picked.Count == 0) return;

            var desktop = Program.EditorUI;
            void Go()
            {
                try
                {
                    TextureCaps.Start(_profile, picked.Select(t => t.Artifact.Guid).ToList(), cap ? budget : 0);
                    // The rescan after the job brings a fresh suggestion: what is capped now is no longer in it
                    _checked.Clear();
                    _touched = false;
                }
                catch (InvalidOperationException ex) { ModalDialogs.ShowMessage(desktop, cap ? "Could not cap the textures" : "Could not take the caps off", ex.Message, ("OK", null)); }
            }

            string message = cap
                ? $"Their Max Size becomes {budget}: {BuildScan.Size(picked.Sum(t => t.Artifact.Size))} now, about {BuildScan.Size(picked.Sum(t => t.CappedBytes))} capped. " +
                  $"{picked.Count(_result!.BelowTarget)} of them would show below {_result.Target:0} px/m. They reimport side by side in the background, " +
                  "then the profile is scanned again. The source files are not touched: Uncap puts them back."
                : $"Their Max Size goes back to Source, and they reimport at their source size. The profile is scanned again afterwards.";
            ModalDialogs.ShowMessage(desktop, cap ? $"Cap {picked.Count} textures at {budget}?" : $"Uncap {picked.Count} textures?", message,
                (cap ? "Cap them" : "Uncap them", Go), ("Cancel", null));
        }

        // ── Rows ──

        void Bind(Row row, int index)
        {
            if (index < 0 || index >= _visible.Count) { row.Visible = false; return; }
            row.Visible = true;
            row.Show(_visible[index], CheckState(_visible[index]), _expanded.Contains(_visible[index].Path), _result!);
        }

        /// <summary>One row: indent, foldout (folders), check box, the name over its numbers.</summary>
        sealed class Row : Button
        {
            readonly TextureBudgetTree _tree;
            readonly Control _indent;
            readonly ImageControl _foldout, _checkmark;
            readonly Button _check;
            readonly Label _name, _numbers;
            Node? _node;

            public Row(TextureBudgetTree tree)
            {
                _tree = tree;
                Size = new Point(100, RowHeight);
                Dock = DockStyle.Top;
                Style = "";
                AllowFocus = false;

                _indent = new Control { Style = "indent18", NoEvents = true, Dock = DockStyle.Left };
                _foldout = new ImageControl
                {
                    Style = "item", Size = new Point(22, RowHeight), Dock = DockStyle.Left, Tiling = TextureMode.Center,
                    Color = ColorInt.ARGB(1f, .5f, .5f, .5f),
                };
                _check = new Button { Style = "checkbox", Size = new Point(18, 18), Dock = DockStyle.Left, Margin = new Margin(2, 11, 6, 11), TextAlign = Alignment.MiddleCenter };
                _checkmark = new ImageControl { Style = "checkmark", Dock = DockStyle.Fill, Margin = new Margin(4), NoEvents = true };
                _check.GetElements().Add(_checkmark);

                var text = new Frame { Dock = DockStyle.Fill, NoEvents = true, Padding = new Margin(0, 3, 0, 3) };
                _name = new Label { Dock = DockStyle.Top, Size = new Point(100, 18), NoEvents = true, TextAlign = Alignment.MiddleLeft };
                _numbers = new Label { Dock = DockStyle.Fill, Style = "stateSubtitle", NoEvents = true, TextAlign = Alignment.MiddleLeft };
                text.Controls.Add(_name);
                text.Controls.Add(_numbers);

                Elements.Add(_indent);
                Elements.Add(_foldout);
                Elements.Add(_check);
                Elements.Add(text);

                _foldout.MouseClick += (s, e) => { if (e.Button == 0 && _node != null) _tree.ToggleExpanded(_node); };
                _check.MouseClick += (s, e) => { if (e.Button == 0 && _node != null) _tree.ToggleCheck(_node); };
                MouseClick += (s, e) => { if (e.Button == 0 && _node?.Texture == null && _node != null) _tree.ToggleExpanded(_node); };
                MouseDoubleClick += (s, e) =>
                {
                    // A texture's own inspector (select the profile again to come back)
                    if (_node?.Texture is { } t && CommandHelpers.FindOrLoadAsset(t.Artifact.Guid) is { } asset)
                        Selector.SelectedObject = asset;
                };
            }

            public void Show(Node node, bool? check, bool expanded, TextureBudgetResult result)
            {
                _node = node;
                _indent.Size = new Point(14 * node.Depth, RowHeight);
                bool folder = node.Texture == null;
                _foldout.Texture = folder ? (expanded ? "nav_down.dds" : "nav_right.dds") : "";
                _foldout.NoEvents = !folder;
                // All checked: the mark; some: a dash (the mark's style keeps its own colour, so a dimmer mark did not show)
                _checkmark.Visible = check == true;
                _check.Text = check == null ? "-" : "";

                _name.Text = node.Name;
                _numbers.UseTextColor = false;
                if (folder)
                {
                    int coarse = node.Textures.Count(result.BelowTarget), capped = node.Textures.Count(t => t.Capped);
                    _numbers.Text = $"{node.Count}: {BuildScan.Size(node.Bytes)}" +
                                    (node.Saving > 0 ? $", saves {BuildScan.Size(node.Saving)}" : "") +
                                    (coarse > 0 ? $", {coarse} coarse" : "") + (capped > 0 ? $", {capped} capped" : "");
                    Tooltip = node.Path;
                    return;
                }

                var t = node.Texture!;
                string sharp = double.IsNaN(t.Coarsest) ? "no use found"
                    : t.Capped ? $"{t.Coarsest:0} px/m" : $"{t.Coarsest:0} -> {t.CoarsestAtBudget:0} px/m";
                _numbers.Text = t.Capped
                    ? $"capped at {t.MaxSize}: {t.Width}x{t.Height} {t.Format} {BuildScan.Size(t.Artifact.Size)} | {sharp}"
                    : $"{t.Width}x{t.Height} {t.Format} {BuildScan.Size(t.Artifact.Size)} -> {BuildScan.Size(t.CappedBytes)} | {sharp}";
                if (result.BelowTarget(t))
                {
                    _numbers.UseTextColor = true;
                    _numbers.TextColor = ColorInt.ARGB(1f, .95f, .45f, .4f);
                }
                else if (t.Capped)
                {
                    _numbers.UseTextColor = true;
                    _numbers.TextColor = ColorInt.ARGB(1f, .5f, .8f, .5f);
                }
                Tooltip = $"{t.Artifact.SourcePath}\n{t.CoarsestUse.Replace('\u203A', '>')}\nDouble-click: its inspector";
            }
        }
    }
}
