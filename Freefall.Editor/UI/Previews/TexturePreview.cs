using System;
using System.Numerics;
using Squid;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Texture preview: renders a fullscreen quad with channel masking.
    /// Returns a toolbar (RGBA buttons + info label) as the preview control.
    /// Sets InspectorControl.PreviewView.OnRender to its render callback.
    /// </summary>
    public class TexturePreview : Frame, IPreview
    {
        private readonly Frame header;
        private readonly Label lblInfo;
        private readonly Button btnR, btnG, btnB, btnA;

        private Material _material;
        private Texture _texture;
        private InspectorControl _inspector;

        // Channel mask state
        private bool _showR = true, _showG = true, _showB = true, _showA = false;

        public TexturePreview()
        {
            Dock = DockStyle.Top;
            Size = new Point(100, 48);

            header = new Frame
            {
                Style = "category",
                Size = new Point(100, 24),
                Dock = DockStyle.Top,
            };

            btnA = AddButton("A");
            btnB = AddButton("B");
            btnG = AddButton("G");
            btnR = AddButton("R");

            btnR.MouseClick += (s, e) => { _showR = !_showR; UpdateButtonStyles(); };
            btnG.MouseClick += (s, e) => { _showG = !_showG; UpdateButtonStyles(); };
            btnB.MouseClick += (s, e) => { _showB = !_showB; UpdateButtonStyles(); };
            btnA.MouseClick += (s, e) =>
            {
                _showA = !_showA;
                if (_showA) { _showR = false; _showG = false; _showB = false; }
                else { _showR = true; _showG = true; _showB = true; }
                UpdateButtonStyles();
            };

            lblInfo = new Label
            {
                Style = "",
                Size = new Point(100, 24),
                Dock = DockStyle.Top,
                TextAlign = Alignment.MiddleCenter
            };

            Controls.Add(header);
            Controls.Add(lblInfo);
        }

        public void Bind(Texture texture, InspectorControl inspector)
        {
            _texture = texture;
            _inspector = inspector;

            if (texture?.Native != null)
            {
                var desc = texture.Native.Description;
                lblInfo.Text = $"{desc.Format}  {desc.Width}x{desc.Height}";
            }
            else
            {
                lblInfo.Text = "";
            }

            // Reset channel mask
            _showR = true; _showG = true; _showB = true; _showA = false;
            UpdateButtonStyles();
        }

        public void OnEnable()
        {
            if (_inspector == null) return;
            _inspector.PreviewView.OnRender = OnRender;
        }

        public void OnDisable()
        {
            if (_inspector == null) return;
            _inspector.PreviewView.OnRender = null;
        }

        private bool _loggedOnce;

        private void OnRender(RenderView view)
        {
            if (_texture == null) return;
            
            var cmd = RenderView.Primary?.CommandList?.Native;
            if (cmd == null) return;

            // Prepare headless RT (transitions, viewport, clear)
            view.PrepareHeadless(cmd);

            if (_material == null)
            {
                _material = new Material(new Effect("texture_preview"));
            }

            // Set channel mask
            var mask = new Vector4(
                _showR ? 1f : 0f,
                _showG ? 1f : 0f,
                _showB ? 1f : 0f,
                0f);

            _material.SetParameter("ChannelMask", mask);
            _material.SetParameter("ShowAlpha", _showA ? 1f : 0f);
            _material.SetParameter("MipLevel", 0f);

            // Bind the texture
            _material.SetTextureIndex("Texture", _texture.BindlessIndex);

            _material.Apply(cmd, Engine.Device);

            // Fullscreen triangle (3 vertices, no index buffer)
            cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
            cmd.DrawInstanced(3, 1, 0, 0);

            if (!_loggedOnce)
            {
                _loggedOnce = true;
                var bindings = _material.Effect?.ResourceBindings?.Count ?? -1;
                var cbs = _material.ConstantBuffers?.Count() ?? -1;
                Debug.Log($"[TexturePreview] Rendered: tex={_texture.Name} idx={_texture.BindlessIndex} RT={view.Width}x{view.Height} PSO={_material.PipelineState?.Native != null} bindings={bindings} cbs={cbs}");
            }

            // Finish headless RT (transition back)
            view.FinishHeadless(cmd);
        }

        private void UpdateButtonStyles()
        {
            btnR.Style = _showR ? "button" : "colorGrey170";
            btnG.Style = _showG ? "button" : "colorGrey170";
            btnB.Style = _showB ? "button" : "colorGrey170";
            btnA.Style = _showA ? "button" : "colorGrey170";
        }

        Button AddButton(string label)
        {
            var btn = new Button
            {
                Dock = DockStyle.Right,
                Size = new Point(24, 24),
                Margin = new Margin(1, 0, 0, 0),
                Text = label,
            };
            header.Controls.Add(btn);
            return btn;
        }
    }
}
