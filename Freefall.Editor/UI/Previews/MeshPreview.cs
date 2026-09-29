using System;
using System.Numerics;
using Squid;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Mesh preview: renders a mesh with basic lighting and orbit controls.
    /// Returns a toolbar (wireframe button + info label) as the preview control.
    /// Sets InspectorControl.PreviewView.OnRender to its render callback.
    /// </summary>
    public class MeshPreview : Frame, IPreview
    {
        private readonly Frame header;
        private readonly Label lblInfo;
        private readonly Button btnWireframe;

        private Material _material;
        private Mesh _mesh;
        private InspectorControl _inspector;
        private bool _isWireframe;

        // Orbit state
        private bool _isRotating;
        private Quaternion _meshRotation = Quaternion.Identity;
        private Quaternion _finalRotation = Quaternion.Identity;

        public MeshPreview()
        {
            Dock = DockStyle.Top;
            Size = new Point(100, 48);

            header = new Frame
            {
                Style = "category",
                Size = new Point(100, 24),
                Dock = DockStyle.Top,
            };

            btnWireframe = AddButton("Wire");
            btnWireframe.MouseClick += (s, e) => { _isWireframe = !_isWireframe; };

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

        public void Bind(Mesh mesh, InspectorControl inspector)
        {
            _mesh = mesh;
            _inspector = inspector;

            // Reset orbit
            _meshRotation = Quaternion.Identity;
            _finalRotation = Quaternion.Identity;

            if (mesh != null)
            {
                lblInfo.Text = $"Verts: {mesh.VertexCount}  Idx: {mesh.IndexCount}  Parts: {mesh.MeshParts.Count}";
            }
            else
            {
                lblInfo.Text = "";
            }
        }

        public void OnEnable()
        {
            if (_inspector == null) return;
            _inspector.PreviewView.OnRender = OnRender;

            // Wire mouse events on the preview image for orbit
            _inspector.PreviewImage.MouseDown += OnMouseDown;
            _inspector.PreviewImage.MouseUp += OnMouseUp;
        }

        public void OnDisable()
        {
            if (_inspector == null) return;
            _inspector.PreviewView.OnRender = null;

            _inspector.PreviewImage.MouseDown -= OnMouseDown;
            _inspector.PreviewImage.MouseUp -= OnMouseUp;
            _isRotating = false;
        }

        private void OnMouseDown(Control sender, MouseEventArgs args) => _isRotating = true;
        private void OnMouseUp(Control sender, MouseEventArgs args) => _isRotating = false;



        private void OnRender(RenderView view)
        {
            if (_mesh == null) return;

            var cmd = RenderView.Primary?.CommandList?.Native;
            if (cmd == null) return;

            // Prepare headless RT (transitions, viewport, clear)
            view.PrepareHeadless(cmd);

            if (_material == null)
            {
                _material = new Material(new Effect("mesh_preview"));
            }

            // Orbit rotation from mouse drag
            if (_isRotating)
            {
                var mouse = new Vector2(-Input.MouseDelta.X, Input.MouseDelta.Y) * 0.01f;
                _finalRotation = Quaternion.CreateFromAxisAngle(-Vector3.UnitX, mouse.Y) * _finalRotation;
                _finalRotation = _finalRotation * Quaternion.CreateFromAxisAngle(Vector3.UnitY, mouse.X);
            }

            _meshRotation = Quaternion.Slerp(_meshRotation, _finalRotation, Base.Time.SmoothDelta * 8);

            // Camera setup from bounding box
            var bounds = _mesh.BoundingBox;
            var center = bounds.Center;
            var extents = bounds.Max - bounds.Min;
            float diagonal = extents.Length();

            // Normalize mesh to a standard size so tiny/huge objects look right
            float targetSize = 2.0f;
            float meshScale = diagonal > 0.0001f ? targetSize / diagonal : 1f;
            var distance = targetSize * 1.5f;

            var cameraPos = center * meshScale - Vector3.UnitZ * distance;
            var viewMatrix = Matrix4x4.CreateLookAtLeftHanded(cameraPos, center * meshScale, Vector3.UnitY);
            var projMatrix = Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(
                MathF.PI / 4f, (float)view.Width / view.Height,
                0.01f, distance * 4f);

            // World matrix: scale + rotate around mesh center
            var meshOffset = -center;
            var rotatedOrigin = Vector3.Transform(meshOffset, Matrix4x4.CreateFromQuaternion(_meshRotation));
            var worldMatrix = Matrix4x4.CreateFromQuaternion(_meshRotation)
                            * Matrix4x4.CreateTranslation(rotatedOrigin + center)
                            * Matrix4x4.CreateScale(meshScale);

            // Set SceneConstants (View, Projection, CamPos) via the effect's master block
            var effect = _material.Effect;
            effect.SetParameter("View", viewMatrix);
            effect.SetParameter("Projection", projMatrix);
            effect.SetParameter("CamPos", cameraPos);

            // Set PreviewConstants (World, lighting)
            _material.SetParameter("World", worldMatrix);
            _material.SetParameter("LightDir", Vector3.Normalize(new Vector3(1, -1, 1)));
            _material.SetParameter("LightColor", new Vector3(1f, 0.95f, 0.9f));
            _material.SetParameter("MaterialColor", new Vector3(0.8f, 0.8f, 0.8f));

            // Push constants for mesh buffer indices
            _material.SetTextureIndex("PosBuffer", _mesh.PosBufferIndex);
            _material.SetTextureIndex("NormBuffer", _mesh.NormBufferIndex);
            _material.SetTextureIndex("UVBuffer", _mesh.UVBufferIndex);
            _material.SetTextureIndex("IndexBuffer", _mesh.IndexBufferIndex);
            _material.Apply(cmd, Engine.Device);

            cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);

            // Draw LOD0 if available, otherwise all parts
            if (_mesh.LODs.Count > 0 && _mesh.LODs[0].MeshPartIndices != null)
            {
                foreach (var partIdx in _mesh.LODs[0].MeshPartIndices)
                {
                    if (partIdx >= _mesh.MeshParts.Count) continue;
                    var part = _mesh.MeshParts[partIdx];
                    if (!part.Enabled) continue;
                    _material.SetTextureIndex("BaseIndex", (uint)part.BaseIndex);
                    _material.Apply(cmd, Engine.Device);
                    cmd.DrawInstanced((uint)part.NumIndices, 1, 0, 0);
                }
            }
            else
            {
                foreach (var part in _mesh.MeshParts)
                {
                    if (!part.Enabled) continue;
                    _material.SetTextureIndex("BaseIndex", (uint)part.BaseIndex);
                    _material.Apply(cmd, Engine.Device);
                    cmd.DrawInstanced((uint)part.NumIndices, 1, 0, 0);
                }
            }

            // Finish headless RT (transition back)
            view.FinishHeadless(cmd);
        }

        Button AddButton(string label)
        {
            var btn = new Button
            {
                Dock = DockStyle.Right,
                Size = new Point(24, 24),
                Margin = new Margin(1, 0, 0, 0),
                Text = label,
                AutoSize = AutoSize.Horizontal,
            };
            header.Controls.Add(btn);
            return btn;
        }
    }
}
