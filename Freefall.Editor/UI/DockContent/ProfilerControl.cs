using System;
using System.Collections.Generic;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// The <see cref="Profiler"/>: a frame-time graph over the recorded history and the CPU and GPU trees,
    /// averaged over the last <see cref="WindowFrames"/> frames, or one frame picked in the graph (which
    /// pauses recording). Children are sorted by time; numbers refresh a few times per second.
    /// </summary>
    public class ProfilerControl : Frame
    {
        private const int WindowFrames = 120;
        private const long RefreshMs = 250;

        // Value columns, right to left: count, alloc, max, ms
        private const int CountWidth = 56;
        private const int AllocWidth = 72;
        private const int MsWidth = 60;
        private const int ValueGap = 8;
        private const int ScrollbarWidth = 12;

        private sealed class Row
        {
            public ProfilerNode Node = null!;
            public bool HasChildren;
            public bool Expanded;
            public string Ms = "", Max = "", Alloc = "", Count = "";
        }

        private readonly Button pauseButton;
        private readonly Label summary;
        private readonly FrameGraph graph;
        private readonly VirtualList list;
        private readonly List<Row> rows = [];
        private readonly List<Row> rowPool = [];

        // Top two levels start open, deeper ones closed; a node in here is the other way round
        private readonly HashSet<ProfilerNode> toggled = [];

        private long selectedFrame = -1;
        private long lastRefresh;

        public ProfilerControl()
        {
            Size = new Point(100, 100);

            var toolbar = new Frame
            {
                Style = "frame",
                Size = new Point(16, 28),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };

            pauseButton = new Button
            {
                Style = "button",
                Text = "Pause",
                Size = new Point(64, 20),
                Dock = DockStyle.Left,
                Margin = new Margin(4, 4, 4, 4),
                CheckOnClick = true,
                Tooltip = "Stop recording frames; click a bar in the graph to look at that frame"
            };
            pauseButton.CheckedChanged += _ =>
            {
                Profiler.Paused = pauseButton.Checked;
                if (!pauseButton.Checked) SelectFrame(-1);
                else RefreshNow();
            };

            summary = new Label
            {
                Style = "label",
                Dock = DockStyle.Fill,
                NoEvents = true,
                AutoEllipsis = true
            };

            toolbar.Controls.Add(pauseButton);
            toolbar.Controls.Add(summary);

            graph = new FrameGraph
            {
                Size = new Point(16, 72),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };
            graph.FramePicked += frame =>
            {
                if (!pauseButton.Checked) pauseButton.Checked = true; // pauses via CheckedChanged
                SelectFrame(frame);
            };

            var header = new Frame
            {
                Style = "header",
                Size = new Point(16, 26),
                Dock = DockStyle.Top,
                Padding = new Margin(0, 0, ScrollbarWidth, 0),
                Margin = new Margin(0, 0, 0, 1)
            };
            header.Controls.Add(HeaderLabel("Count", CountWidth));
            header.Controls.Add(HeaderLabel("Alloc", AllocWidth));
            header.Controls.Add(HeaderLabel("Max ms", MsWidth));
            header.Controls.Add(HeaderLabel("ms", MsWidth));
            header.Controls.Add(new Label { Style = "header", Text = "Section", Dock = DockStyle.Fill, NoEvents = true });

            list = new VirtualList
            {
                Dock = DockStyle.Fill,
                ItemHeight = 24,
                CreateItem = CreateRow,
                BindItem = BindRow
            };
            list.Scrollbar.ButtonDown.Visible = false;
            list.Scrollbar.ButtonUp.Visible = false;
            list.Scrollbar.Slider.Ease = false;
            list.Scrollbar.Slider.MinHandleSize = 64;
            list.Scrollbar.ShowAlways = true; // keeps the value columns under the header
            list.DataSource = rows;

            Controls.Add(toolbar);
            Controls.Add(graph);
            Controls.Add(header);
            Controls.Add(list);
        }

        private static Label HeaderLabel(string text, int width) => new()
        {
            Style = "header",
            Text = text,
            Size = new Point(width - ValueGap, 26),
            Margin = new Margin(0, 0, ValueGap, 0),
            Dock = DockStyle.Right,
            TextAlign = Alignment.MiddleRight,
            NoEvents = true
        };

        protected override void OnUpdate()
        {
            base.OnUpdate();

            // Paused, the numbers only change when something is picked or opened (RefreshNow)
            long now = Environment.TickCount64;
            if (!Profiler.Paused && now - lastRefresh >= RefreshMs)
                RefreshNow();
        }

        private void SelectFrame(long frame)
        {
            selectedFrame = frame;
            graph.Selected = frame;
            RefreshNow();
        }

        private void RefreshNow()
        {
            lastRefresh = Environment.TickCount64;
            if (selectedFrame >= 0 && !Profiler.IsRecorded(selectedFrame))
                selectedFrame = graph.Selected = -1;

            UpdateSummary();

            int count = 0;
            AddRow(Profiler.Root, ref count);
            AddRow(Profiler.GpuRoot, ref count);
            for (int i = rows.Count - 1; i >= count; i--)
            {
                rowPool.Add(rows[i]);
                rows.RemoveAt(i);
            }

            list.Refresh();
        }

        private void UpdateSummary()
        {
            if (selectedFrame >= 0)
            {
                string gpuMs = Profiler.IsGpuRecorded(selectedFrame)
                    ? $"{Profiler.GpuRoot.Sample(selectedFrame).Ms:F2} ms" : "not timed";
                summary.Text = $"Frame {selectedFrame}: {Profiler.FrameMs(selectedFrame):F2} ms, GPU {gpuMs}" +
                               GcText(Profiler.FrameCollections(selectedFrame));
                return;
            }

            long end = Profiler.FrameCount;
            long start = Math.Max(end - WindowFrames, end - Profiler.RecordedFrames);
            int n = (int)(end - start);
            if (n <= 0)
            {
                summary.Text = "";
                return;
            }

            float sum = 0, max = 0;
            long alloc = 0;
            int gc = 0;
            for (long f = start; f < end; f++)
            {
                float ms = Profiler.FrameMs(f);
                sum += ms;
                max = Math.Max(max, ms);
                alloc += Profiler.FrameAllocatedBytes(f);
                gc += Profiler.FrameCollections(f) != 0 ? 1 : 0;
            }

            float avg = sum / n;
            var gpu = Profiler.GpuRoot.Stats(WindowFrames);
            summary.Text = $"{avg:F1} ms ({1000f / Math.Max(avg, 0.001f):F0} fps), max {max:F1}, GPU {gpu.AvgMs:F1} ms, " +
                           $"{FormatBytes(alloc / n)}/frame on all threads, GC in {gc} of {n} frames";
        }

        private static string GcText(int gc)
        {
            if (gc == 0) return "";
            int gen = (gc & 4) != 0 ? 2 : (gc & 2) != 0 ? 1 : 0;
            return $", gen{gen} GC";
        }

        // The number a node is sorted and shown by: the picked frame or the window average
        private float Value(ProfilerNode node) =>
            selectedFrame >= 0 ? node.Sample(selectedFrame).Ms : node.Stats(WindowFrames).AvgMs;

        private bool IsExpanded(ProfilerNode node) => (node.Depth <= 1) != toggled.Contains(node);

        private void AddRow(ProfilerNode node, ref int count)
        {
            var row = count < rows.Count ? rows[count] : NewRow();
            count++;

            row.Node = node;
            row.HasChildren = node.Children.Count > 0;
            row.Expanded = row.HasChildren && IsExpanded(node);

            if (selectedFrame >= 0)
            {
                var sample = node.Sample(selectedFrame);
                row.Ms = sample.Calls > 0 ? sample.Ms.ToString("F2") : "";
                row.Max = "";
                row.Alloc = FormatBytes(sample.AllocBytes);
                row.Count = FormatCount(sample.Items, sample.Calls);
            }
            else
            {
                var stats = node.Stats(WindowFrames);
                row.Ms = stats.AvgCalls > 0 ? stats.AvgMs.ToString("F2") : "";
                row.Max = stats.AvgCalls > 0 ? stats.MaxMs.ToString("F2") : "";
                row.Alloc = FormatBytes((long)stats.AvgAllocBytes);
                row.Count = FormatCount(stats.AvgItems, stats.AvgCalls);
            }

            if (!row.Expanded) return;

            var children = new List<(ProfilerNode Node, float Value)>(node.Children.Count);
            foreach (var child in node.Children)
                children.Add((child, Value(child)));
            children.Sort((a, b) => b.Value.CompareTo(a.Value));

            foreach (var (child, _) in children)
                AddRow(child, ref count);
        }

        private Row NewRow()
        {
            Row row;
            if (rowPool.Count > 0)
            {
                row = rowPool[^1];
                rowPool.RemoveAt(rowPool.Count - 1);
            }
            else row = new Row();

            rows.Add(row);
            return row;
        }

        private static string FormatBytes(long bytes)
        {
            if (bytes <= 0) return "";
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024f:F1} KB";
            return $"{bytes / (1024f * 1024f):F2} MB";
        }

        // Items when the section reports what it worked on (components), otherwise calls when more than one
        private static string FormatCount(float items, float calls)
        {
            if (items > 0) return MathF.Round(items).ToString("N0");
            if (calls > 1.01f) return "x" + calls.ToString(calls % 1 == 0 ? "F0" : "F1");
            return "";
        }

        private Control CreateRow(int index)
        {
            var control = new RowControl();
            control.MouseClick += (s, e) =>
            {
                if (e.Button > 0 || s is not RowControl { Row: { HasChildren: true } row }) return;
                if (!toggled.Remove(row.Node)) toggled.Add(row.Node);
                RefreshNow();
            };
            BindRow(control, index);
            return control;
        }

        private void BindRow(Control control, int index)
        {
            if (index < rows.Count) ((RowControl)control).Bind(rows[index]);
        }

        private sealed class RowControl : Button
        {
            private readonly Control indent;
            private readonly ImageControl foldout;
            private readonly Label name, ms, max, alloc, count;

            public Row? Row { get; private set; }

            public RowControl()
            {
                Size = new Point(100, 24);
                Dock = DockStyle.Top;
                Style = "item"; // hot across the whole row, gaps between the columns included
                AllowFocus = false;

                indent = new Control { Style = "indent18", NoEvents = true, Dock = DockStyle.Left };
                foldout = new ImageControl
                {
                    Style = "item",
                    NoEvents = true,
                    Size = new Point(20, 24),
                    Dock = DockStyle.Left,
                    Tiling = TextureMode.Center,
                    Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
                };

                count = Value(CountWidth);
                alloc = Value(AllocWidth);
                max = Value(MsWidth);
                ms = Value(MsWidth);
                name = new Label { Style = "item", NoEvents = true, Dock = DockStyle.Fill, AutoEllipsis = true };

                Elements.Add(indent);
                Elements.Add(foldout);
                Elements.Add(count);
                Elements.Add(alloc);
                Elements.Add(max);
                Elements.Add(ms);
                Elements.Add(name);
            }

            private static Label Value(int width) => new()
            {
                Style = "item",
                NoEvents = true,
                Size = new Point(width - ValueGap, 24),
                Dock = DockStyle.Right,
                Margin = new Margin(0, 0, ValueGap, 0),
                TextAlign = Alignment.MiddleRight
            };

            public void Bind(Row row)
            {
                Row = row;
                indent.Size = new Point(12 * row.Node.Depth, indent.Size.y);
                foldout.Texture = !row.HasChildren ? "" : row.Expanded ? "nav_down.dds" : "nav_right.dds";
                name.Text = row.Node.Name;
                ms.Text = row.Ms;
                max.Text = row.Max;
                alloc.Text = row.Alloc;
                count.Text = row.Count;
            }

            protected override void OnStateChanged()
            {
                indent.State = State;
                foldout.State = State;
                name.State = State;
                ms.State = State;
                max.State = State;
                alloc.State = State;
                count.State = State;
            }
        }

        /// <summary>
        /// One bar per recorded frame (CPU wall time), newest on the right, scaled to at least 33 ms with lines
        /// at 60 and 30 fps; a blue mark at the frame's GPU time, a tick on top for a garbage collection.
        /// Click or drag to pick a frame.
        /// </summary>
        private sealed class FrameGraph : Control
        {
            public event Action<long>? FramePicked;
            public long Selected = -1;

            private bool dragging;

            private static readonly int Background = ColorInt.ARGB(1f, .11f, .11f, .12f);
            private static readonly int Guide = ColorInt.ARGB(.35f, .6f, .6f, .6f);
            private static readonly int Fast = ColorInt.ARGB(1f, .30f, .62f, .38f);
            private static readonly int Slow = ColorInt.ARGB(1f, .85f, .62f, .20f);
            private static readonly int VerySlow = ColorInt.ARGB(1f, .85f, .28f, .24f);
            private static readonly int Picked = ColorInt.ARGB(1f, 1f, 1f, 1f);
            private static readonly int Gpu = ColorInt.ARGB(1f, .40f, .70f, 1f);
            private static readonly int Gc0 = ColorInt.ARGB(1f, .55f, .55f, .55f);
            private static readonly int Gc1 = ColorInt.ARGB(1f, .90f, .60f, .20f);
            private static readonly int Gc2 = ColorInt.ARGB(1f, .95f, .25f, .25f);

            public FrameGraph()
            {
                NoEvents = false;
                MouseDown += (s, e) => { if (e.Button == 0) { dragging = true; Pick(); } };
                MouseUp += (s, e) => dragging = false;
                MousePress += (s, e) => { if (dragging) Pick(); };
            }

            private void Pick()
            {
                int w = Size.x;
                if (w <= 0 || Profiler.RecordedFrames == 0) return;

                int slot = Math.Clamp((Gui.MousePosition.x - Location.x) * Profiler.HistoryLength / w, 0, Profiler.HistoryLength - 1);
                long frame = Profiler.FrameCount - Profiler.HistoryLength + slot;
                if (Profiler.IsRecorded(frame) && frame != Selected) FramePicked?.Invoke(frame);
            }

            protected override void DrawCustom()
            {
                var r = Gui.Renderer;
                int x0 = Location.x, y0 = Location.y, w = Size.x, h = Size.y;
                r.DrawBox(x0, y0, w, h, Background);

                long end = Profiler.FrameCount;
                long first = end - Profiler.HistoryLength;

                float scale = 33.4f;
                for (long f = Math.Max(first, 0); f < end; f++)
                    scale = Math.Max(scale, Profiler.FrameMs(f));

                const int top = 4; // room for the GC ticks
                int plot = h - top;

                r.DrawBox(x0, y0 + h - (int)(plot * 16.67f / scale), w, 1, Guide);
                r.DrawBox(x0, y0 + h - (int)(plot * 33.33f / scale), w, 1, Guide);

                for (int slot = 0; slot < Profiler.HistoryLength; slot++)
                {
                    long frame = first + slot;
                    if (!Profiler.IsRecorded(frame)) continue;

                    int x = x0 + slot * w / Profiler.HistoryLength;
                    int bw = Math.Max(1, x0 + (slot + 1) * w / Profiler.HistoryLength - x);

                    float ms = Profiler.FrameMs(frame);
                    int bh = Math.Max(1, (int)(plot * ms / scale));
                    int color = frame == Selected ? Picked : ms <= 16.67f ? Fast : ms <= 33.33f ? Slow : VerySlow;
                    r.DrawBox(x, y0 + h - bh, bw, bh, color);

                    if (Profiler.IsGpuRecorded(frame))
                    {
                        int gh = Math.Min(plot, (int)(plot * Profiler.GpuRoot.Sample(frame).Ms / scale));
                        r.DrawBox(x, y0 + h - gh - 1, bw, 2, Gpu);
                    }

                    int gc = Profiler.FrameCollections(frame);
                    if (gc != 0)
                        r.DrawBox(x, y0, bw, 3, (gc & 4) != 0 ? Gc2 : (gc & 2) != 0 ? Gc1 : Gc0);
                }
            }
        }
    }
}
