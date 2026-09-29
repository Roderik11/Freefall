using System;
using Squid;
using Vortice.Mathematics;
using Freefall.Graphics;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    /// <summary>
    /// Dropdown-style HSV color picker.
    /// Opens as a popup Window (like DropDownList) that closes on outside click.
    /// </summary>
    public class ColorPicker : Window
    {
        // ── Singleton textures ──
        private static Texture _hueTexture;
        private static Texture _svTexture;
        private static bool _texturesRegistered;
        private static float _lastGeneratedHue = -1f;

        private const int SvSize = 64;   // CPU texture resolution (Squid stretches)
        private const int HueSize = 64;

        // ── HSV state ──
        private float _hue;        // 0–360
        private float _saturation; // 0–1
        private float _value;      // 0–1
        private float _alpha = 1f;

        // ── UI ──
        private readonly Frame _svArea;
        private readonly ImageControl _svImage;
        private readonly Frame _svMarker;
        private readonly Frame _hueStrip;
        private readonly ImageControl _hueImage;
        private readonly Frame _hueMarker;
        private readonly Frame _previewSwatch;
        private readonly Frame _oldSwatch;
        private readonly TextBox _tbR, _tbG, _tbB;
        private readonly TextBox _tbHex;

        private bool _draggingSV;
        private bool _draggingHue;

        private Color4 _originalColor;

        public event Action<Color4> ColorChanged;

        public static Control Opener {get; private set;}

        public Color4 Color
        {
            get => HsvToRgb(_hue, _saturation, _value, _alpha);
            set => SetFromRgba(value);
        }

        public ColorPicker()
        {
            EnsureTextures();

            int popupW = 260;
            int svW = 200;
            int svH = 200;
            int hueW = 20;
            int gap = 8;
            int fieldH = 22;
            int previewH = 24;

            Style = "frame";
            Size = new Point(popupW, svH + previewH + fieldH * 2 + gap * 5 + 4);
            Padding = new Margin(gap);
            AutoSize = AutoSize.Vertical;

            // ── SV area container ──
            var svRow = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(popupW, svH),
                Margin = new Margin(0, 0, 0, gap),
            };

            _svArea = new Frame
            {
                Dock = DockStyle.Fill,
                Style = "color",
                NoEvents = false,
            };
            _svArea.MouseDown += SvArea_MouseDown;
            _svArea.MouseUp += SvArea_MouseUp;
            _svArea.MousePress += SvArea_MouseDrag;

            _svImage = new ImageControl
            {
                Dock = DockStyle.Fill,
                Texture = "colorpicker_sv",
                Tiling = TextureMode.Stretch,
                NoEvents = true,
            };
            _svArea.Controls.Add(_svImage);

            _svMarker = new Frame
            {
                Size = new Point(10, 10),
                Style = "border",
                NoEvents = true,
                Position = new Point(0, 0),
            };
            _svArea.Controls.Add(_svMarker);

            // ── Hue strip ──
            _hueStrip = new Frame
            {
                Dock = DockStyle.Right,
                Size = new Point(hueW, svH),
                Style = "color",
                Margin = new Margin(gap, 0, 0, 0),
                NoEvents = false,
            };
            _hueStrip.MouseDown += HueStrip_MouseDown;
            _hueStrip.MouseUp += HueStrip_MouseUp;
            _hueStrip.MousePress += HueStrip_MouseDrag;

            _hueImage = new ImageControl
            {
                Dock = DockStyle.Fill,
                Texture = "colorpicker_hue",
                Tiling = TextureMode.Stretch,
                NoEvents = true,
            };
            _hueStrip.Controls.Add(_hueImage);

            _hueMarker = new Frame
            {
                Size = new Point(hueW, 3),
                Style = "border",
                NoEvents = true,
                Position = new Point(0, 0),
            };
            _hueStrip.Controls.Add(_hueMarker);

            svRow.Controls.Add(_hueStrip);
            svRow.Controls.Add(_svArea);
            Controls.Add(svRow);

            // ── Preview row (old | new) ──
            var previewRow = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(popupW, previewH),
                Margin = new Margin(0, 0, 0, gap),
            };
            _oldSwatch = new Frame
            {
                Dock = DockStyle.Left,
                Size = new Point(popupW / 2 - gap, previewH),
                Style = "color",
            };
            _previewSwatch = new Frame
            {
                Dock = DockStyle.Fill,
                Style = "color",
            };
            previewRow.Controls.Add(_oldSwatch);
            previewRow.Controls.Add(_previewSwatch);
            Controls.Add(previewRow);

            // ── RGB row ──
            var rgbRow = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(popupW, fieldH),
                Margin = new Margin(0, 0, 0, gap / 2),
            };

            var lblR = new Label { Text = "R", Style = "label", Size = new Point(20, fieldH), Dock = DockStyle.Left };
            _tbR = CreateNumericField(fieldH);
            var lblG = new Label { Text = "G", Style = "label", Size = new Point(20, fieldH), Dock = DockStyle.Left, Margin = new Margin(4, 0, 0, 0) };
            _tbG = CreateNumericField(fieldH);
            var lblB = new Label { Text = "B", Style = "label", Size = new Point(20, fieldH), Dock = DockStyle.Left, Margin = new Margin(4, 0, 0, 0) };
            _tbB = CreateNumericField(fieldH);

            rgbRow.Controls.Add(lblR);
            rgbRow.Controls.Add(_tbR);
            rgbRow.Controls.Add(lblG);
            rgbRow.Controls.Add(_tbG);
            rgbRow.Controls.Add(lblB);
            rgbRow.Controls.Add(_tbB);
            Controls.Add(rgbRow);

            // ── Hex row ──
            var hexRow = new Frame
            {
                Dock = DockStyle.Top,
                Size = new Point(popupW, fieldH),
            };
            var lblHex = new Label { Text = "#", Style = "label", Size = new Point(20, fieldH), Dock = DockStyle.Left };
            _tbHex = new TextBox
            {
                Style = "textbox",
                Size = new Point(80, fieldH),
                Dock = DockStyle.Fill,
            };
            _tbHex.TextCommit += HexField_OnCommit;

            hexRow.Controls.Add(lblHex);
            hexRow.Controls.Add(_tbHex);
            Controls.Add(hexRow);

            // ── RGB text commit handlers ──
            _tbR.TextCommit += RgbField_OnCommit;
            _tbG.TextCommit += RgbField_OnCommit;
            _tbB.TextCommit += RgbField_OnCommit;
        
            this.OnParentChanged += () => 
            { 
                if(Parent == null)
                 {
                    ColorChanged = null;
                 }
            };
        }

        // ── Public API ──

        public void Open(Control anchor)
        {
            var desktop = anchor.Desktop;
            if (desktop == null) return;

            // Position below the anchor
            var loc = anchor.Location;
            Position = new Point(loc.x, loc.y + anchor.Size.y + 2);

            // Ensure it stays on screen
            int dw = desktop.Size.x;
            int dh = desktop.Size.y;
            if (Position.x + Size.x > dw)
                Position = new Point(dw - Size.x, Position.y);
            if (Position.y + Size.y > dh)
                Position = new Point(Position.x, loc.y - Size.y - 2);

            desktop.ShowDropdown(this, false);

            _originalColor = Color;
            _oldSwatch.Tint = (int)_originalColor.ToRgba();

            _lastGeneratedHue = -1f; // force SV texture rebuild
            RebuildSvTexture();
            UpdateUI();
        }

        public void Close()
        {
            _draggingSV = false;
            _draggingHue = false;
        }

        // ── SV area interaction ──

        private void SvArea_MouseDown(Control sender, MouseEventArgs e)
        {
            _draggingSV = true;
            UpdateSvFromMouse();
        }

        private void SvArea_MouseUp(Control sender, MouseEventArgs e)
        {
            _draggingSV = false;
        }

        private void SvArea_MouseDrag(Control sender, MouseEventArgs e)
        {
            if (_draggingSV) UpdateSvFromMouse();
        }

        private void UpdateSvFromMouse()
        {
            var mp = Gui.MousePosition;
            var loc = _svArea.Location;
            var size = _svArea.Size;

            float x = Math.Clamp((mp.x - loc.x) / (float)size.x, 0f, 1f);
            float y = Math.Clamp((mp.y - loc.y) / (float)size.y, 0f, 1f);

            _saturation = x;
            _value = 1f - y;

            UpdateUI();
            NotifyColorChanged();
        }

        // ── Hue strip interaction ──

        private void HueStrip_MouseDown(Control sender, MouseEventArgs e)
        {
            _draggingHue = true;
            UpdateHueFromMouse();
        }

        private void HueStrip_MouseUp(Control sender, MouseEventArgs e)
        {
            _draggingHue = false;
        }

        private void HueStrip_MouseDrag(Control sender, MouseEventArgs e)
        {
            if (_draggingHue) UpdateHueFromMouse();
        }

        private void UpdateHueFromMouse()
        {
            var mp = Gui.MousePosition;
            var loc = _hueStrip.Location;
            var size = _hueStrip.Size;

            float y = Math.Clamp((mp.y - loc.y) / (float)size.y, 0f, 1f);
            _hue = y * 360f;

            RebuildSvTexture();
            UpdateUI();
            NotifyColorChanged();
        }

        // ── RGB text fields ──

        private void RgbField_OnCommit(object sender, EventArgs e)
        {
            if (!int.TryParse(_tbR.Text, out int r)) r = 0;
            if (!int.TryParse(_tbG.Text, out int g)) g = 0;
            if (!int.TryParse(_tbB.Text, out int b)) b = 0;

            r = Math.Clamp(r, 0, 255);
            g = Math.Clamp(g, 0, 255);
            b = Math.Clamp(b, 0, 255);

            var c = new Color4(r / 255f, g / 255f, b / 255f, _alpha);
            SetFromRgba(c);
            RebuildSvTexture();
            UpdateUI();
            NotifyColorChanged();
        }

        private void HexField_OnCommit(object sender, EventArgs e)
        {
            var hex = _tbHex.Text.Trim().TrimStart('#');
            if (hex.Length == 6 &&
                int.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out int val))
            {
                int r = (val >> 16) & 0xFF;
                int g = (val >> 8) & 0xFF;
                int b = val & 0xFF;

                var c = new Color4(r / 255f, g / 255f, b / 255f, _alpha);
                SetFromRgba(c);
                RebuildSvTexture();
                UpdateUI();
                NotifyColorChanged();
            }
        }

        // ── Internal update ──

        private void NotifyColorChanged()
        {
            ColorChanged?.Invoke(Color);
        }

        private void SetFromRgba(Color4 c)
        {
            RgbToHsv(c.R, c.G, c.B, out _hue, out _saturation, out _value);
            _alpha = c.A;
        }

        private void UpdateUI()
        {
            var c = Color;
            int tint = (int)c.ToRgba();

            _previewSwatch.Tint = tint;

            // SV marker position
            if (_svArea.Size.x > 0 && _svArea.Size.y > 0)
            {
                int mx = (int)(_saturation * _svArea.Size.x) - _svMarker.Size.x / 2;
                int my = (int)((1f - _value) * _svArea.Size.y) - _svMarker.Size.y / 2;
                _svMarker.Position = new Point(mx, my);
            }

            // Hue marker position
            if (_hueStrip.Size.y > 0)
            {
                int hy = (int)(_hue / 360f * _hueStrip.Size.y) - _hueMarker.Size.y / 2;
                _hueMarker.Position = new Point(0, hy);
            }

            // RGB text (only update if not focused)
            int ri = (int)(c.R * 255f + 0.5f);
            int gi = (int)(c.G * 255f + 0.5f);
            int bi = (int)(c.B * 255f + 0.5f);

            if (_tbR.Desktop == null || _tbR.Desktop.FocusedControl != _tbR)
                _tbR.Text = ri.ToString();
            if (_tbG.Desktop == null || _tbG.Desktop.FocusedControl != _tbG)
                _tbG.Text = gi.ToString();
            if (_tbB.Desktop == null || _tbB.Desktop.FocusedControl != _tbB)
                _tbB.Text = bi.ToString();

            if (_tbHex.Desktop == null || _tbHex.Desktop.FocusedControl != _tbHex)
                _tbHex.Text = $"{ri:X2}{gi:X2}{bi:X2}";
        }

        // ── Texture generation ──

        private static void EnsureTextures()
        {
            if (_texturesRegistered) return;
            _texturesRegistered = true;

            // ── Hue strip (1 × HueSize) ──
            byte[] hueData = new byte[1 * HueSize * 4];
            for (int y = 0; y < HueSize; y++)
            {
                float h = y / (float)HueSize * 360f;
                var c = HsvToRgb(h, 1f, 1f, 1f);
                int i = y * 4;
                hueData[i + 0] = (byte)(c.R * 255f + 0.5f);
                hueData[i + 1] = (byte)(c.G * 255f + 0.5f);
                hueData[i + 2] = (byte)(c.B * 255f + 0.5f);
                hueData[i + 3] = 255;
            }
            _hueTexture = Texture.CreateFromData(Engine.Device, 1, HueSize, hueData, Vortice.DXGI.Format.R8G8B8A8_UNorm);
            _hueTexture.Name = "colorpicker_hue";

            var renderer = Gui.Renderer as SquidRenderer;
            renderer?.InsertTexture("colorpicker_hue", _hueTexture);

            // ── SV gradient (SvSize × SvSize) — initial with hue=0 ──
            GenerateSvTexture(0f);
        }

        private void RebuildSvTexture()
        {
            // Avoid rebuilding if hue hasn't changed enough
            if (MathF.Abs(_hue - _lastGeneratedHue) < 0.5f) return;

            GenerateSvTexture(_hue);
        }

        private static void GenerateSvTexture(float hue)
        {
            _lastGeneratedHue = hue;

            byte[] svData = new byte[SvSize * SvSize * 4];
            for (int y = 0; y < SvSize; y++)
            {
                float v = 1f - y / (float)(SvSize - 1);
                for (int x = 0; x < SvSize; x++)
                {
                    float s = x / (float)(SvSize - 1);
                    var c = HsvToRgb(hue, s, v, 1f);
                    int i = (y * SvSize + x) * 4;
                    svData[i + 0] = (byte)(c.R * 255f + 0.5f);
                    svData[i + 1] = (byte)(c.G * 255f + 0.5f);
                    svData[i + 2] = (byte)(c.B * 255f + 0.5f);
                    svData[i + 3] = 255;
                }
            }

            _svTexture = Texture.CreateFromData(Engine.Device, SvSize, SvSize, svData, Vortice.DXGI.Format.R8G8B8A8_UNorm);
            _svTexture.Name = "colorpicker_sv";
            var renderer = Gui.Renderer as SquidRenderer;
            renderer?.InsertTexture("colorpicker_sv", _svTexture);
        }

        // ── Helpers ──

        private static TextBox CreateNumericField(int height)
        {
            return new TextBox
            {
                Style = "textbox",
                Size = new Point(36, height),
                Dock = DockStyle.Left,
                Mode = TextBoxMode.Numeric,
            };
        }

        // ── HSV ↔ RGB ──

        public static Color4 HsvToRgb(float h, float s, float v, float a)
        {
            h = ((h % 360f) + 360f) % 360f;
            float c = v * s;
            float x = c * (1f - MathF.Abs((h / 60f) % 2f - 1f));
            float m = v - c;

            float r, g, b;
            if (h < 60f) { r = c; g = x; b = 0; }
            else if (h < 120f) { r = x; g = c; b = 0; }
            else if (h < 180f) { r = 0; g = c; b = x; }
            else if (h < 240f) { r = 0; g = x; b = c; }
            else if (h < 300f) { r = x; g = 0; b = c; }
            else { r = c; g = 0; b = x; }

            return new Color4(r + m, g + m, b + m, a);
        }

        public static void RgbToHsv(float r, float g, float b, out float h, out float s, out float v)
        {
            float max = MathF.Max(r, MathF.Max(g, b));
            float min = MathF.Min(r, MathF.Min(g, b));
            float delta = max - min;

            v = max;
            s = max > 0f ? delta / max : 0f;

            if (delta < 1e-6f)
            {
                h = 0f;
            }
            else if (max == r)
            {
                h = 60f * (((g - b) / delta) % 6f);
            }
            else if (max == g)
            {
                h = 60f * (((b - r) / delta) + 2f);
            }
            else
            {
                h = 60f * (((r - g) / delta) + 4f);
            }

            if (h < 0f) h += 360f;
        }
    }
}
