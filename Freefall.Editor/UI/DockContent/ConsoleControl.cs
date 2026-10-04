using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Squid;
using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using Freefall.Base;

namespace Freefall.Editor
{
    public class ConsoleControl : Frame
    {
        private readonly VirtualList VirtualList;
        private readonly Label lblDetails;
        private Frame toolbar;
        private SplitContainer split;
        private SearchBox searchbox;
        private Frame bottombar;
        private IList<Debug.LogEntry> bindList;

        public ConsoleControl()
        {
            Size = new Point(100, 100);
            bindList = Debug.Lines;

            toolbar = new Frame
            {
                Style = "frame",
                Size = new Point(16, 28),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };

            bottombar = new Frame
            {
                Style = "frame",
                Size = new Point(16, 28),
                Dock = DockStyle.Bottom,
                Margin = new Margin(0, 1, 0, 0)
            };

            searchbox = new SearchBox
            {
                Size = new Point(200, 16),
                Dock = DockStyle.Left,
                Margin = new Margin(28, 4, 4, 4)
            };

            lblDetails = new Label
            {
                Size = new Point(200, 100),
                Dock = DockStyle.Fill,
                Style = "multiline",
                TextWrap = true,
                BBCodeEnabled = true,
                LinkColor = ColorInt.ARGB(1, .2f, .6f, 1f),
                Leading = 4
            };
            lblDetails.LinkClicked += LblDetails_LinkClicked;

            split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.RetainAspect = false;
            split.Orientation = Orientation.Vertical;
            split.SplitFrame1.Size = new Point(200, 200);
            split.SplitButton.Size = new Point(4, 4);
            split.SplitButton.Style = "frame";
            split.SplitButton.Margin = new Margin(1, 0, 1, 0);

            searchbox.TextChanged += Searchbox_TextChanged;

            VirtualList = new VirtualList();
            VirtualList.Dock = DockStyle.Fill;
            VirtualList.Scrollbar.ButtonDown.Visible = false;
            VirtualList.Scrollbar.ButtonUp.Visible = false;
            VirtualList.Scrollbar.Slider.Ease = false;
            VirtualList.Scrollbar.Slider.MinHandleSize = 64;
            VirtualList.CreateItem = CreateNode;
            VirtualList.BindItem = BindNode;
            VirtualList.ItemHeight = 28;
            VirtualList.DataSource = bindList as IList;


            toolbar.Controls.Add(searchbox);

            split.SplitFrame1.Controls.Add(VirtualList);
            split.SplitFrame2.Controls.Add(lblDetails);

            Controls.Add(toolbar);
            Controls.Add(bottombar);
            Controls.Add(split);
        }

        private void Searchbox_TextChanged(Control sender)
        {
            var str = searchbox.Text;
            bool isempty = string.IsNullOrEmpty(str);
            str = str.ToLower();

            bindList = Debug.Lines;

            if (!isempty)
            {
                var list = new List<Debug.LogEntry>();

                foreach (Debug.LogEntry e in Debug.Lines)
                {
                    if (e.Message.ToLower().Contains(str))
                        list.Add(e);
                }
                bindList = list;
            }

            VirtualList.DataSource = bindList as IList;
            VirtualList.Refresh();
        }

        private void LblDetails_LinkClicked(string href)
        {
            var parts = href.Split(':');
            string name = parts[0] + ":" + parts[1];
            int line = Convert.ToInt32(parts[2]);
            
            if(OpenFileAtLine(name, line))
                return;

            Process process = new Process();
            ProcessStartInfo startInfo = new ProcessStartInfo("devenv.exe", $"/edit {name}");
            process.StartInfo = startInfo;
            process.Start();
        }

        [DllImport("oleaut32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void GetActiveObject(ref Guid rclsid, IntPtr reserved, [MarshalAs(UnmanagedType.IUnknown)] out object ppunk);

        private static object GetActiveObject(string progID)
        {
            Type? type = Type.GetTypeFromProgID(progID);
            if (type == null)
                throw new InvalidOperationException($"ProgID not found: {progID}");

            Guid clsid = type.GUID;
            GetActiveObject(ref clsid, IntPtr.Zero, out object obj);
            return obj;
        }

        
        private bool OpenFileAtLine(string file, int line)
        {
            try
            {
                object vs = GetActiveObject("VisualStudio.DTE");
                object ops = vs.GetType().InvokeMember("ItemOperations", BindingFlags.GetProperty, null, vs, null);
                object window = ops.GetType().InvokeMember("OpenFile", BindingFlags.InvokeMethod, null, ops, new object[] { file });
                object selection = window.GetType().InvokeMember("Selection", BindingFlags.GetProperty, null, window, null);
                selection.GetType().InvokeMember("GotoLine", BindingFlags.InvokeMethod, null, selection, new object[] { line, true });
                return true;
            }
            catch
            {
                return false;
            }
        }

        private void Result_MouseClick(Control sender, MouseEventArgs args)
        {
            var logentry = (Debug.LogEntry)sender.Tag;
            var text = MakePretty(logentry);
            lblDetails.Text = text;
        }

        private Control CreateNode(int index)
        {
            var result = new Button
            {
                Style = "item",
                Size = new Point(100, VirtualList.ItemHeight),
                Dock = DockStyle.Top,
                Text = bindList[index].Message,
                Tag = bindList[index],
            };

            result.MouseClick += Result_MouseClick;

            return result;
        }
        
        private string MakePretty(Debug.LogEntry entry)
        {
            // Script logs carry a pre-rendered stack (a live StackTrace would pin the script assembly)
            string text;
            (string File, int Line)[] locations;
            if (entry.StackTrace != null)
            {
                text = entry.StackTrace.GetFrames().StackFramesToString();
                locations = Debug.StackLocationsOf(entry.StackTrace);
            }
            else if (entry.StackText != null)
            {
                text = entry.StackText;
                locations = entry.StackLocations ?? [];
            }
            else
            {
                // Release builds log warnings/errors without a captured stack
                return "(no stack trace captured in Release builds)";
            }

            string reg = "{0}:line {1}";

            foreach (var (filename, line) in locations)
            {
                string find = string.Format(CultureInfo.InstalledUICulture, reg, filename, line);
                text = text.Replace(find, $"[url={filename}:{line}]{find}[/url]");
            }

            return text;
        }

        private void BindNode(Control control, int index)
        {
            var label = control as Button;
            label.Text = bindList[index].Message;
            label.Tag = bindList[index];
        }
    }
}
